import itertools
import os
from dataclasses import dataclass
from typing import Any, Optional, cast

from omegaconf import read_write

import habitat # type: ignore
import habitat_sim
import magnum as mn
import numpy as np
from habitat import RLEnv # type: ignore
from habitat_baselines.common.baseline_registry import baseline_registry # type: ignore
from habitat.utils.visualizations import fog_of_war, maps # type: ignore
from gym import spaces
from hydra.core.config_store import ConfigStore
from habitat.config.default_structured_configs import ActionConfig # type: ignore
from habitat.core.embodied_task import SimulatorTaskAction # type: ignore
from habitat.tasks.nav.nav import NavigationGoal, NavigationTask, NavigationEpisode # type: ignore
from habitat_sim.utils.common import quat_rotate_vector

from common.env_utils.lighting import AI2THOR_BRIGHT_LIGHTS
from common.env_utils.object_annotations import ObjectAnnotation, get_all_objects
from common.planning.skeleton import grid_path
from common.utils.pose_utils import quaternion_from_rpy


_FLOOR_RAY_START_M = 0.3  # above the navmesh, below anything the agent can stand under


@dataclass
class SideStepActionConfig(ActionConfig):
    type: str = "SideStepAction"
    move_amount: float = 0.25
    angle_deg: float = 90.0


@habitat.registry.register_task_action
class SideStepAction(SimulatorTaskAction):
    """Moves the agent move_amount metres at angle_deg from its heading (90: to its left), sliding along the navmesh, without turning."""

    def step(self, *args: Any, **kwargs: Any) -> None:
        state = self._sim.get_agent_state()
        angle = np.deg2rad(self._config.angle_deg)
        offset = quat_rotate_vector(state.rotation, np.array([-np.sin(angle), 0.0, -np.cos(angle)])) * self._config.move_amount
        position = self._sim.pathfinder.try_step(state.position, state.position + offset)
        self._sim.set_agent_state(position, state.rotation, reset_sensors=False)


for _name, _angle_deg in (("move_left", 90.0), ("move_right", -90.0), ("move_backward", 180.0)):
    ConfigStore.instance().store(
        package=f"habitat.task.actions.{_name}", group="habitat/task/actions", name=_name, node=SideStepActionConfig(angle_deg=_angle_deg),
    )


@baseline_registry.register_task(name="ExplorationTask-v0")
class ExplorationTask(NavigationTask):
    def __init__(self, config, sim, dataset)  -> None:
        super().__init__(config=config, sim=sim, dataset=dataset)

    def _check_episode_is_active(self, *args: Any, **kwargs: Any) -> bool:
        return True

@baseline_registry.register_env(name="ExplorationEnv-v0")
class ExplorationEnv(RLEnv):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(config)
        self.kwargs = kwargs
        self._previous_action = None
        self.episode_over = False
        self._elapsed_steps = 0
        self._tdmap: Optional[np.ndarray] = None


    def reset(self, episode: Optional[NavigationEpisode] = None, rng: Optional[np.random.Generator] = None):
        r"""Resets the sim to `episode` if given, else re-initializes the current one --
        never consults `self._env._episode_iterator` (see `self.episodes` to enumerate
        candidates), so the caller always controls episode/scene selection explicitly.

        Places the agent at a random navigable point: an episode's own start_position/
        start_rotation are meaningless synthetic placeholders (see common/env_utils/dataset.py::
        ExplorationNavDataset), and agents (common/agents/) act from the agent's live pose."""
        self._elapsed_steps = 0
        self.episode_over = False
        self._tdmap = None

        self._env._reset_stats()

        if self._env._current_episode is not None:
            self._env._current_episode._shortest_path_cache = None

        if episode is not None:
            self._env.current_episode = episode

        self._env._episode_from_iter_on_reset = True
        self._env._episode_force_changed = False

        assert self._env._current_episode is not None, "Reset requires an episode"
        self._env.reconfigure(self._env._config)
        # Simulator.reset() (called above when the sim config is unchanged) unconditionally
        # rewrites DEFAULT_LIGHTING_KEY back to habitat-sim's stock rig, so any override has to
        # be re-applied every reset, not just once at construction.
        if self.get_env_name() == "ProcTHOR-hab":
            self.set_light_setup(AI2THOR_BRIGHT_LIGHTS)

        agent_state = habitat_sim.AgentState()
        agent_state.position = self.get_random_point(rng)
        agent_state.rotation = self.get_random_rotation(rng)
        self._env.sim.agents[0].set_state(agent_state)

        sim_obs = self._env.sim.get_sensor_observations()

        self._env.task.sensor_suite.get("bbsgt").setup_semantic_labels()
        sim_obs = self._env.sim.get_sensor_observations()

        task_obs = self._env.task.sensor_suite.get_observations(
            observations=sim_obs,
            episode=self._env._current_episode,
            task=self._env.task,
            should_time=True,
        )
        observations = {**sim_obs, **task_obs}

        for action_instance in self._env.task.actions.values():
            action_instance.reset(episode=self._env._current_episode, task=self._env.task)

        self._env._is_episode_active = True
        self._env._task.measurements.reset_measures(
            episode=self._env._current_episode,
            task=self._env.task,
            observations=observations,
        )
        return observations


    def set_goals(self, data):
        self._env.current_episode.goals = data.copy()

    def get_legal_actions(self) -> list[str]:
        legal_actions = []
        for action in self._env.action_space.spaces.keys():
            if self._env.task.is_action_allowed(action):
                legal_actions.append(action)
        return legal_actions

    def set_done(self, done) -> None:
        self._env.current_episode.episode_over = done

    def get_pathfinder(self) -> habitat_sim.PathFinder:
        return self._env.sim.pathfinder

    def get_sim(self) -> habitat_sim.Simulator:
        return self._env.sim

    def navmesh_floor_offset(self) -> float:
        """Height of the agent's navmesh position above the first collidable surface straight below it, the floor."""
        position = np.array(self.get_agent_state().position)
        ray = habitat_sim.geo.Ray(mn.Vector3(*(position + [0.0, _FLOOR_RAY_START_M, 0.0])), mn.Vector3(0.0, -1.0, 0.0))
        return float(position[1] - self._env.sim.cast_ray(ray).hits[0].point[1])

    def get_map_bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        l, u = self._env.sim.pathfinder.get_bounds()
        return l, u

    def get_last_reward(self) -> float:
        return self.last_reward

    def get_tdmap(self) -> np.ndarray:
        return self.get_object_annotation().object_occupancy_grid.topdown_view

    def get_object_annotation(self) -> ObjectAnnotation:
        return self._env.task.sensor_suite.get("bbsgt").annotation

    _GROUND_FLOOR_MPP = 0.125  # meters/cell, matches HabitatObjOccupancyGrid's own resolution
    _STEP_WALK_METERS = 0.25  # waypoint spacing along find_shortest_path_waypoints's route (embodied-active-learning-od's own grid spacing)

    def get_navmesh_grid(self) -> np.ndarray:
        r"""Navigable-cell mask of the floor's main navmesh island at _GROUND_FLOOR_MPP resolution,
        lazily built and cached per scene (reset on reset()/change_scene()) -- shared by
        get_random_point/find_shortest_path_waypoints below. Furniture tops baked into the navmesh
        (navmesh_include_static_objects) form small islands within the floor's height slab; only the
        largest island is kept so spawns and waypoints never land on them."""
        if self._tdmap is None:
            lower_bound, _ = self._env.sim.pathfinder.get_bounds()
            islands = self._env.sim.pathfinder.get_topdown_island_view(
                meters_per_pixel=self._GROUND_FLOOR_MPP, height=lower_bound[1]
            )
            self._floor_island = int(np.bincount(islands[islands >= 0]).argmax())
            self._tdmap = (islands == self._floor_island).astype(np.uint8)
        return self._tdmap

    def get_floor_island(self) -> int:
        r"""Navmesh island index of get_navmesh_grid()'s main floor island."""
        self.get_navmesh_grid()
        return self._floor_island

    def _snap_to_floor(self, x: float, z: float) -> np.ndarray:
        pathfinder = self._env.sim.pathfinder
        lower_bound, _ = pathfinder.get_bounds()
        return np.array(pathfinder.snap_point(mn.Vector3(x, lower_bound[1], z), island_index=self.get_floor_island()))

    def get_random_point(self, rng: Optional[np.random.Generator] = None, min_distance: float = 0.0, max_retries: int = 100) -> np.ndarray:
        pathfinder = self._env.sim.pathfinder
        tdmap = self.get_navmesh_grid()

        rows, cols = np.nonzero(tdmap)

        k = 0

        while k < max_retries:
            idx = rng.integers(len(rows)) if rng is not None else np.random.randint(len(rows))
            z, x = maps.from_grid(rows[idx], cols[idx], tdmap.shape, pathfinder=pathfinder)
            snapped = self._snap_to_floor(x, z)

            if min_distance > 0:
                agent_state = self._env.sim.agents[0].get_state()
                dist = np.linalg.norm(snapped - np.array(agent_state.position))
                if dist < min_distance:
                    continue
            k += 1

        return snapped
    
    def get_random_rotation(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        
        angle = self._env.sim.habitat_config.turn_angle * rng.integers(360 / self._env.sim.habitat_config.turn_angle) 
        return quaternion_from_rpy(0, 0, np.deg2rad(angle))

    def get_target_object_positions(self) -> list[tuple[int, int, np.ndarray]]:
        r"""(object_id, class_id, world position) per kept object (same filter as
        ObjectDetectorGTSensor.decompose_frame) -- feeds queryobjectsbaseline's navigation.
        Only ever non-empty for HSSD-HAB/ProcTHOR-hab: MP3D/Gibson-Semantic have no placed
        rigid/articulated objects for get_all_objects to enumerate."""
        sensor = self._env.task.sensor_suite.get("bbsgt")
        annotation = sensor.annotation
        classes = annotation.classes

        out = []
        for obj in get_all_objects(self._env.sim):
            class_id = annotation.obj_id_to_class_id.get(obj.object_id)
            if class_id is None:
                continue
            class_name = classes[class_id]
            if class_name == "unknown" or (sensor.filter_classes is not None and class_name not in sensor.filter_classes):
                continue

            if hasattr(obj, "collision_shape_aabb"):
                aabb = obj.collision_shape_aabb
                corners_local = itertools.product(
                    [aabb.min.x, aabb.max.x], [aabb.min.y, aabb.max.y], [aabb.min.z, aabb.max.z]
                )
                world_corners = [obj.rotation.transform_vector(mn.Vector3(*c)) + obj.translation for c in corners_local]
                position = np.mean([[c.x, c.y, c.z] for c in world_corners], axis=0)
            else:
                t = obj.translation
                position = np.array([t.x, t.y, t.z])

            out.append((obj.object_id, class_id, position))
        return out

    def get_point_near(self, position: np.ndarray, radius: float = 1.5, max_tries: int = 100) -> Optional[np.ndarray]:
        r"""Random navigable point of the floor island within `radius` of `position`, or None if none found."""
        point = np.array(self._env.sim.pathfinder.get_random_navigable_point_near(
            mn.Vector3(*position), radius, max_tries=max_tries, island_index=self.get_floor_island(),
        ))
        return None if np.isnan(point).any() else point

    def find_shortest_path_waypoints(self, start: np.ndarray, end: np.ndarray) -> Optional[list[np.ndarray]]:
        pathfinder = self._env.sim.pathfinder
        tdmap = self.get_navmesh_grid()

        start_row, start_col = maps.to_grid(start[2], start[0], tdmap.shape, pathfinder=pathfinder)
        end_row, end_col = maps.to_grid(end[2], end[0], tdmap.shape, pathfinder=pathfinder)

        path_px = grid_path(
            tdmap, (start_col, start_row), (end_col, end_row),
            step_size=self._STEP_WALK_METERS / self._GROUND_FLOOR_MPP,
        )
        if len(path_px) < 2:
            return None

        points = []
        for col, row in path_px:
            z, x = maps.from_grid(row, col, self._tdmap.shape, pathfinder=pathfinder)
            points.append(self._snap_to_floor(x, z))
        return points

    def get_env_name(self) -> str:
        return self._env.task.sensor_suite.get("bbsgt").env_name

    def get_vocab_name(self) -> str:
        return self._env.task.sensor_suite.get("bbsgt").vocab_name

    def get_agent_state(self) -> habitat_sim.AgentState:
        r"""Current position/rotation, including per-sensor states -- what an Agent (common/agents/) acts from."""
        return self._env.sim.agents[0].get_state()

    def step(self, action) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        self._elapsed_steps += 1
        self._previous_action = action
        obs = self._env.step(action)
        return  obs, 0, False, {}

    def teleport(self, agent_state: habitat_sim.AgentState) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        self._elapsed_steps += 1
        self._env.sim.agents[0].set_state(agent_state)
        sim_obs = self._env.sim.get_sensor_observations()
        task_obs = self._env.task.sensor_suite.get_observations(
            observations=sim_obs, episode=self._env._current_episode, task=self._env.task
        )
        obs = {**sim_obs, **task_obs}
        return  obs, 0, False, {}

    def get_step(self) -> int:
        return self._elapsed_steps

    def set_light_setup(self, lights: list, key: str = habitat_sim.gfx.DEFAULT_LIGHTING_KEY) -> None:
        self._env.sim.set_light_setup(lights, key)

    @property
    def original_action_space(self) -> spaces.space: # type: ignore
        return self.action_space

    def change_scene(self, scene: str) -> None:
        with read_write(self._env._config):
            self._env._config.simulator.scene = self._env._config.dataset.scenes_dir + "/" +  scene

        self._env._sim.reconfigure(self._env._config.simulator)
        self._env.task.sensor_suite.get("bbsgt").setup_semantic_labels()
        self._tdmap = None
        return

    def get_reward_range(self):
        return (-1.0, 1.0)
