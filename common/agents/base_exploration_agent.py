"""Shared depth-observed world model for every exploration agent (FrontierExplorationAgent, HybridFrontierAgent,
SearchAgent): an EstimatedMap built incrementally from the agent's own depth and pose, with DiscretePlanner-based
planning on top -- all three plan entirely from this belief map, HybridFrontierAgent via a goal_map of its own
object viewpoints/frontiers. Agents only ever see poses relative to their reference pose (ref_frame_state), as
habitat's episodic GPS + compass sensors give them: the map's origin is the start position, its axes the start heading
snapped to _REF_YAW_STEP_RAD."""
from typing import List, Optional, Tuple

import cv2
import habitat_sim
import matplotlib
import numpy as np
import quaternion
from habitat.tasks.nav.nav import TopDownMap  # type: ignore

from common.agents.abstract_agent import Agent, StopReason
from common.hybrid_exploration.discrete_planner import DiscreteNavigationAction, DiscretePlanner
from common.hybrid_exploration.exploration_map import EstimatedMap, MapConfig
from common.hybrid_exploration.plotting import draw_in_view
from common.utils.pose_utils import get_yaw, normalize_angle, quaternion_from_rpy
from common.utils.projection_utils import DepthCamera

_ACTION_NAMES = {
    DiscreteNavigationAction.MOVE_FORWARD: "move_forward",
    DiscreteNavigationAction.TURN_LEFT: "turn_right",
    DiscreteNavigationAction.TURN_RIGHT: "turn_left",
    DiscreteNavigationAction.STOP: "stop",
}

_GOAL_MARKER_COLOR = (200, 0, 200)  # BGR -- magenta/purple, matching EstimatedMap.belief_image's scheme
_FRONTIER_PIXEL_COLOR = (0, 165, 255)  # BGR -- orange, every current frontier cell (not just the chosen goal)
_FRONTIER_PIXEL_ALPHA = 0.5
_VISITED_COLOR = (0, 0, 255)
_CURRENT_POSITION_COLOR = (0, 255, 0)  # BGR -- green, the agent's current position on the map
_GOAL_SCORE_COLORMAP = matplotlib.colormaps["cool"]  # cyan (low) -> magenta (high), by goal_score_image() value
_STUCK_WINDOW_STEPS = 25
_STUCK_RADIUS_M = 0.1
_REF_YAW_STEP_RAD = np.pi / 2  # keeps the map grid on the world axes, along which scene walls run
_MAP_FIT_MARGIN = 0.9  # fraction of map_size_m the navmesh's own extent must fit within to be used as the map's center

class ProfileTimer:
    functions_names: List[str]
    timers: List[float]
    calls: List[int]

    def __init__(self):
        self.functions_names = []
        self.timers = []
        self.calls = []

    def update(self, function_name: str, elapsed_time: float) -> None:
        if function_name not in self.functions_names:
            self.functions_names.append(function_name)
            self.timers.append(elapsed_time)
            self.calls.append(1)
        else:
            idx = self.functions_names.index(function_name)
            self.timers[idx] += elapsed_time
            self.calls[idx] += 1
    
    def display(self) -> str:
        out = "Profiling results:\n"
        for name, time, calls in zip(self.functions_names, self.timers, self.calls):
            out += f"{name}: {time:.4f}s over {calls} calls ({time/calls:.4f}s per call)\n"
        return out

class BaseExplorationAgent(Agent):
    map_config: MapConfig
    rng: np.random.Generator
    planner: Optional[DiscretePlanner]
    estimated_map: Optional[EstimatedMap]
    trajectory: List[Tuple[np.ndarray, float]]
    _spin_action: Optional[str]
    _closest_goal_map: Optional[np.ndarray]
    _short_term_goal: Optional[Tuple[int, int]]
    _initial_spin_steps_left: int

    def __init__(self, map_config: MapConfig, rng: np.random.Generator, turn_angle_deg: float, planner: Optional[DiscretePlanner] = None):
        self.map_config = map_config
        self.rng = rng
        self.planner = planner
        self.floor_height = 0.0
        self.ref_state: Tuple[np.ndarray, quaternion.quaternion] = (np.zeros(3), quaternion.one)
        self.estimated_map: Optional[EstimatedMap] = None
        self.trajectory: List[Tuple[np.ndarray, float]] = []
        self._spin_action = None
        self._closest_goal_map = None
        self._short_term_goal = None
        self._initial_spin_steps = int(round(360.0 / turn_angle_deg))
        self._initial_spin_steps_left = self._initial_spin_steps

    def reset(
        self, start_state: habitat_sim.AgentState, floor_height: float, navmesh_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        """ref_state becomes start_state's position and its yaw snapped to _REF_YAW_STEP_RAD, the origin of every pose the
        agent sees (ref_frame_state); floor_height is a world y. navmesh_bounds (a scene's full pathfinder.get_bounds(), world-frame) centers EstimatedMap on the
        navmesh's own horizontal extent instead of the agent, when that extent fits within map_size_m -- so a long
        episode doesn't run the agent past the map's fixed edge; otherwise (or without navmesh_bounds) the map is
        centered on the agent's own start position, as before."""
        rotation = start_state.rotation
        snapped_yaw = round(get_yaw(rotation.w, rotation.x, rotation.y, rotation.z) / _REF_YAW_STEP_RAD) * _REF_YAW_STEP_RAD
        self.ref_state = (np.array(start_state.position), quaternion_from_rpy(0, 0, snapped_yaw))
        self.floor_height = floor_height - self.ref_state[0][1]
        self.estimated_map = EstimatedMap(self.map_config, self._map_center(navmesh_bounds))
        if self.planner is not None:
            self.planner.reset()
        self.trajectory = []
        self._spin_action = None
        self._closest_goal_map = None
        self._short_term_goal = None
        self._initial_spin_steps_left = self._initial_spin_steps

    def _map_center(self, navmesh_bounds: Optional[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """The navmesh's own horizontal center, in the reference frame, if its full extent fits within _MAP_FIT_MARGIN of
        map_size_m; np.zeros(3) (the agent's own start position) otherwise, or without navmesh_bounds."""
        if navmesh_bounds is None:
            return np.zeros(3)
        lower, upper = navmesh_bounds
        corners = np.array([[x, 0.0, z] for x in (lower[0], upper[0]) for z in (lower[2], upper[2])])
        corners_ref_frame = self.ref_frame_points(corners)
        extent = corners_ref_frame.max(axis=0) - corners_ref_frame.min(axis=0)
        if extent[0] > _MAP_FIT_MARGIN * self.map_config.map_size_m or extent[2] > _MAP_FIT_MARGIN * self.map_config.map_size_m:
            return np.zeros(3)
        return (corners_ref_frame.max(axis=0) + corners_ref_frame.min(axis=0)) / 2

    def ref_frame_points(self, points: np.ndarray) -> np.ndarray:
        """(N, 3) world points in the reference frame."""
        position, rotation = self.ref_state
        return quaternion.rotate_vectors(rotation.inverse(), np.asarray(points) - position)

    def ref_frame_state(self, agent_state: habitat_sim.AgentState) -> habitat_sim.AgentState:
        """agent_state, sensor poses included, relative to ref_state -- the only pose act() may use."""
        def relative(pose):
            return habitat_sim.SixDOFPose(
                position=self.ref_frame_points(np.asarray(pose.position)[None])[0], rotation=self.ref_state[1].inverse() * pose.rotation,
            )
        sensor_states = {uuid: relative(pose) for uuid, pose in agent_state.sensor_states.items()}
        state = relative(agent_state)
        return habitat_sim.AgentState(position=state.position, rotation=state.rotation, sensor_states=sensor_states)

    def initial_spin(self) -> Optional[str]:
        """spin()'s action for the episode's first full 360deg turn, None once that turn is complete -- lets each
        agent build initial belief-map coverage around itself before planning any movement."""
        if self._initial_spin_steps_left <= 0:
            return None
        self._initial_spin_steps_left -= 1
        return self.spin()

    def observe(self, agent_state: habitat_sim.AgentState, depth: np.ndarray, camera: DepthCamera) -> Tuple[np.ndarray, float]:
        assert self.estimated_map is not None, "Call reset() first"
        position, heading = np.array(agent_state.position), float(TopDownMap.get_polar_angle(agent_state))
        self.trajectory.append((position, heading))
        self.estimated_map.observe_obstacles(depth, agent_state.sensor_states["depth"], camera, self.floor_height)
        self.estimated_map.reveal(position, heading)
        return position, heading

    def sensor_pose(self, position: np.ndarray, heading: float) -> np.ndarray:
        """(7,) [x, y, o, gx1, gx2, gy1, gy2] for DiscretePlanner.plan(): x/y in meters derived from this
        agent's own EstimatedMap pixel grid (row -> y, col -> x, matching plan()'s own start=[row, col]
        derivation), planning window = the whole map (EstimatedMap has no local/global tiers).

        o is in degrees, in DiscretePlanner's own convention -- NOT habitat's polar angle (heading, as
        TopDownMap.get_polar_angle/observe() use for fog-of-war and point-turn logic elsewhere). o = 90 -
        degrees(heading) is the only formula verified to keep straight-line move_forward motion consistent
        with the planner's own atan2(row_delta, col_delta) goal-angle math (checked by moving a real agent
        forward at 8 headings spanning a full turn and confirming the resulting (row, col) displacement's
        atan2 equals this o at every one). Note _ACTION_NAMES swaps TURN_LEFT/TURN_RIGHT to compensate: under
        this o, habitat's own "turn_left" is what decreases o, matching what get_action's TURN_RIGHT branch
        expects -- using the unswapped action names with this o (or this o's negation with unswapped names)
        both produce a planner that oscillates turn_right/turn_left forever instead of converging, or that
        drives straight through walls away from its own stated goal; this pairing is the one verified
        correct end-to-end against real sim movement."""
        assert self.estimated_map is not None, "Call reset() first"
        row, col = self.estimated_map.to_pixel(position)
        size = self.estimated_map.map_pixels_size
        o = normalize_angle(90.0 - np.degrees(heading))
        return np.array([col * self.estimated_map.meters_per_pixel, row * self.estimated_map.meters_per_pixel, o, 0, size, 0, size], dtype=float)

    def is_stuck(self) -> bool:
        """True once the last _STUCK_WINDOW_STEPS trajectory positions all lie within _STUCK_RADIUS_M of each other --
        no net translation, even while the heading keeps oscillating between turns. Every agent's act() stops then."""
        if len(self.trajectory) < _STUCK_WINDOW_STEPS:
            return False
        recent = np.stack([position for position, _ in self.trajectory[-_STUCK_WINDOW_STEPS:]])
        return bool(np.all(np.linalg.norm(recent - recent[0], axis=1) < _STUCK_RADIUS_M))

    @staticmethod
    def action_name(action: DiscreteNavigationAction) -> str:
        return _ACTION_NAMES[action]

    @staticmethod
    def plan_stop_reason(action_name: str, no_feasible_plan: bool, no_frontier: bool) -> Optional[StopReason]:
        """The StopReason for a DiscretePlanner.plan() result, when action_name is "stop" (None otherwise)."""
        if action_name != "stop":
            return None
        if no_frontier:
            return StopReason.EXPLORATION_EXHAUSTED
        return StopReason.NO_FEASIBLE_PLAN if no_feasible_plan else StopReason.GOAL_REACHED

    def spin(self) -> str:
        if self._spin_action is None:
            self._spin_action = "turn_left" if self.rng.integers(2) else "turn_right"
        return self._spin_action

    def map_image(self) -> np.ndarray:
        """BGR top-down map: the belief map (EstimatedMap.belief_image), the current field of view tinted yellow,
        a blue goal-score heatmap when the agent defines goal_score_image() (e.g. SearchAgent's EstimatedMap.goal_score --
        drawn where each cell's value is > 0, colored by that value), the trajectory so far in
        red, every current frontier cell (EstimatedMap.frontier_pixels, not just the one picked as goal) as a
        half-transparent orange dot, and the planner's actual target this step -- plan()'s own
        closest_goal_map, the geodesically (not just Euclidean) nearest frontier cell or found-object goal --
        marked purple on top."""
        assert self.estimated_map is not None, "Call reset() first"

        image = self.estimated_map.belief_image()
        draw_in_view(image, self.estimated_map.in_view)
        if hasattr(self, "goal_score_image"):
            self._blend_goal_score(image, self.goal_score_image())
        trajectory_px = [(int(pixel[1]), int(pixel[0])) for pixel in (self.estimated_map.to_pixel(position) for position, _ in self.trajectory)]
        if len(trajectory_px) > 1:
            cv2.polylines(image, [np.array(trajectory_px)], isClosed=False, color=_VISITED_COLOR, thickness=1)
        for point in trajectory_px:
            cv2.circle(image, point, 1, _VISITED_COLOR, -1)
        cv2.circle(image, trajectory_px[-1], 2, _CURRENT_POSITION_COLOR, -1)  # last position
        
        if len(self.estimated_map.frontier_pixels):
            frontier_overlay = image.copy()
            for row, col in np.round(self.estimated_map.frontier_pixels).astype(int):
                cv2.circle(frontier_overlay, (int(col), int(row)), 1, _FRONTIER_PIXEL_COLOR, -1)
            cv2.addWeighted(frontier_overlay, _FRONTIER_PIXEL_ALPHA, image, 1 - _FRONTIER_PIXEL_ALPHA, 0, dst=image)
        
        if self._closest_goal_map is not None and self._closest_goal_map.any():
            goal_pixels = np.argwhere(self._closest_goal_map)
            for row, col in goal_pixels:
                cv2.circle(image, (int(col), int(row)), 3, _GOAL_MARKER_COLOR, 1)
        return image

    @staticmethod
    def _blend_goal_score(image: np.ndarray, goal_score: Optional[np.ndarray]) -> None:
        """Draws goal_score's matplotlib 'cool' colormap (cyan at 0, magenta at 1) into image in place,
        for every cell where goal_score > 0."""
        if goal_score is None or not goal_score.any():
            return
        mask = goal_score > 0
        value = np.clip(goal_score[mask], 0.0, 1.0)
        color_bgr = _GOAL_SCORE_COLORMAP(value)[:, 2::-1] * 255.0  # RGBA -> BGR, 0-255
        image[mask] = color_bgr.astype(np.uint8)

    def map_tile(self) -> Tuple[str, np.ndarray]:
        """(label, RGB map_image) mosaic tile of the current step."""
        return f"step{len(self.trajectory) - 1:03d}", self.map_image()[:, :, ::-1].copy()
