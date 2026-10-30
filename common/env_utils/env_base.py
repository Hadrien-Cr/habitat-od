import itertools
import os
from dataclasses import dataclass
from typing import Any, Optional, cast

from omegaconf import read_write

import habitat # type: ignore
import habitat_sim
import magnum as mn
import numpy as np
import trimesh
from habitat import RLEnv # type: ignore
from habitat.config import read_write as habitat_read_write # type: ignore
from habitat_baselines.common.baseline_registry import baseline_registry # type: ignore
from habitat.utils.visualizations import fog_of_war, maps # type: ignore
from gym import spaces
from hydra.core.config_store import ConfigStore
from habitat.config.default_structured_configs import ActionConfig, ObjectDetectorGTSensorConfig # type: ignore
from habitat.core.embodied_task import SimulatorTaskAction # type: ignore
from habitat.tasks.nav.nav import NavigationGoal, NavigationTask, NavigationEpisode # type: ignore
from habitat.utils.geometry_utils import quaternion_from_coeff # type: ignore
from habitat_sim.utils.common import quat_rotate_vector

from common.agents.abstract_agent import Agent
from common.agents.random_agent import RandomAgent
from common.env_utils.dataset import NO_FIXED_START_POSITION, NO_FIXED_START_ROTATION, ExplorationNavDataset  # noqa: F401 - registers "ExplorationSynthetic"
from common.env_utils.env_registry import resolve_env
from common.env_utils.lighting import AI2THOR_BRIGHT_LIGHTS
from common.env_utils.object_annotations import ObjectAnnotation, get_all_objects, get_obj_from_id, object_world_mesh
from common.env_utils.object_detector_sensors import ObjectDetectorGTSensor  # noqa: F401 - registers "ObjectDetectorGTSensor"
from common.env_utils.sensors import AgentCollisionSensor, AgentPositionSensor, ColoredTDMapSensor  # noqa: F401 - registers agent_collision_sensor/position_sensor/colored_tdmap_sensor
from common.hybrid_exploration.exploration_map import MapConfig
from common.hybrid_exploration.frontier_exp_agent import DiscretePlannerConfig, FrontierExplorationAgent, FrontierExplorationConfig
from common.hybrid_exploration.hybrid_agent import HybridConfig, HybridFrontierAgent
from common.utils.plot_utils import plot_semantic_2d_map
from common.utils.pose_utils import quaternion_from_rpy
from common.utils.projection_utils import DepthCamera, TOPDOWN_MAP_SCALE, mesh_footprint_on_grid


_FLOOR_RAY_START_M = 0.3  # above the navmesh, below anything the agent can stand under


def gt_navigable_mask(pathfinder: habitat_sim.PathFinder, position: np.ndarray, resolution: int) -> np.ndarray:
    r"""Navigable-cell mask of the navmesh island reachable from position, at height position[1] -- ground
    truth, for visualization/comparison only (never used for planning). Usable directly (e.g. by a raw
    habitat_sim.Simulator with no ExplorationEnv) or via ExplorationEnv.get_gt_topdown_map."""
    meters_per_pixel = maps.calculate_meters_per_pixel(resolution, pathfinder=pathfinder)
    islands = pathfinder.get_topdown_island_view(meters_per_pixel, position[1])
    return np.ascontiguousarray(islands == pathfinder.get_island(mn.Vector3(*position)), dtype=np.uint8)


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
        self._gt_semantic_map_cache_key: Optional[tuple] = None


    def reset(self, episode: Optional[NavigationEpisode] = None, rng: Optional[np.random.Generator] = None):
        r"""Resets the sim to `episode` if given, else re-initializes the current one --
        never consults `self._env._episode_iterator` (see `self.episodes` to enumerate
        candidates), so the caller always controls episode/scene selection explicitly.

        Places the agent at the episode's own start_position/start_rotation when it has one
        (a real per-episode dataset, e.g. habitat_od_search's HSSD-rare episodes), or at a
        random navigable point/rotation when it carries ExplorationNavDataset's
        NO_FIXED_START_POSITION/NO_FIXED_START_ROTATION sentinel (its synthetic episodes --
        see common/env_utils/dataset.py)."""
        self._elapsed_steps = 0
        self.episode_over = False
        self._tdmap = None
        self._gt_semantic_map_cache_key = None

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

        episode = self._env._current_episode
        agent_state = habitat_sim.AgentState()
        has_fixed_start = (
            list(episode.start_position) != NO_FIXED_START_POSITION or list(episode.start_rotation) != NO_FIXED_START_ROTATION
        )
        if has_fixed_start:
            agent_state.position = np.array(episode.start_position)
            agent_state.rotation = quaternion_from_coeff(episode.start_rotation)
        else:
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

    def get_gt_topdown_map(self, position: np.ndarray, resolution: int) -> np.ndarray:
        r"""Navigable-cell mask of the navmesh island reachable from position, at height position[1] -- the
        ground-truth counterpart to an agent's own depth-built EstimatedMap, for visualization/comparison only
        (never used for planning)."""
        return gt_navigable_mask(self._env.sim.pathfinder, position, resolution)

    def gt_map_tracker(self, fov_deg: float, visibility_dist_m: float, resolution: int = 512) -> "GroundTruthMapTracker":
        r"""GroundTruthMapTracker of this env's current scene/agent position -- the ground-truth counterpart
        to an agent's own depth-built EstimatedMap, for a logged comparison panel only (never used for
        planning). Local import: common.hybrid_exploration.ground_truth_map itself imports gt_navigable_mask
        from this module, so importing it back at module scope here would cycle."""
        from common.hybrid_exploration.ground_truth_map import GroundTruthMapTracker
        position = np.array(self.get_agent_state().position)
        return GroundTruthMapTracker(self._env.sim.pathfinder, position, fov_deg, visibility_dist_m, resolution)

    def get_gt_meshes(self, classes: list) -> list[tuple[int, int, trimesh.Trimesh]]:
        r"""(object_id, class_id, world mesh) of every scene object the GT sensor annotates with one of
        classes -- env-owned, needs no agent."""
        annotation = self.get_object_annotation()
        class_ids = {name: idx for idx, name in enumerate(classes)}
        names = {object_id: annotation.classes[class_id] for object_id, class_id in annotation.obj_id_to_class_id.items()}
        objects = [
            (object_id, class_ids[name], get_obj_from_id(self._env.sim, object_id)) for object_id, name in names.items() if name in class_ids
        ]
        meshes = [(object_id, class_id, object_world_mesh(obj)) for object_id, class_id, obj in objects if obj is not None]
        return [(object_id, class_id, mesh) for object_id, class_id, mesh in meshes if mesh is not None]

    def get_gt_semantic_map(self, classes: list, colors: list, resolution: int = 256) -> np.ndarray:
        r"""RGB top-down silhouette map of every GT-annotated object of `classes`, on this env's own
        navmesh-shaped canvas (get_gt_topdown_map around the agent's current position) -- independent of
        any agent's belief map shape/resolution. Cached per (classes, colors, resolution), invalidated
        alongside _tdmap on reset()/change_scene()."""
        cache_key = (tuple(classes), tuple(map(tuple, colors)), resolution)
        if self._gt_semantic_map_cache_key != cache_key:
            origin = np.array(self.get_agent_state().position)
            canvas = self.get_gt_topdown_map(origin, resolution)
            lower_bound, upper_bound = self._env.sim.pathfinder.get_bounds()
            meters_per_pixel = maps.calculate_meters_per_pixel(resolution, pathfinder=self._env.sim.pathfinder)
            grid_size = (abs(upper_bound[2] - lower_bound[2]) / canvas.shape[0], abs(upper_bound[0] - lower_bound[0]) / canvas.shape[1])

            def to_pixels(points: np.ndarray) -> np.ndarray:
                rows = ((points[:, 2] - lower_bound[2]) / grid_size[0]).astype(int)
                cols = ((points[:, 0] - lower_bound[0]) / grid_size[1]).astype(int)
                return np.stack([rows, cols], axis=1)

            footprints = [(class_id, mesh_footprint_on_grid(mesh, canvas.shape, to_pixels)) for _, class_id, mesh in self.get_gt_meshes(classes)]
            image = plot_semantic_2d_map(
                canvas, footprints, dict(enumerate(colors)), dict(enumerate(classes)),
                meters_per_grid_pixel=meters_per_pixel, scale=TOPDOWN_MAP_SCALE,
            )
            self._gt_semantic_map_cache = np.array(image)
            self._gt_semantic_map_cache_key = cache_key
        return self._gt_semantic_map_cache

    def get_last_reward(self) -> float:
        return self.last_reward

    def get_tdmap(self) -> np.ndarray:
        return self.get_object_annotation().object_occupancy_grid.topdown_view

    def get_object_annotation(self) -> ObjectAnnotation:
        return self._env.task.sensor_suite.get("bbsgt").annotation

    _GROUND_FLOOR_MPP = 0.125  # meters/cell, matches HabitatObjOccupancyGrid's own resolution

    def get_navmesh_grid(self) -> np.ndarray:
        r"""Navigable-cell mask of the floor's main navmesh island at _GROUND_FLOOR_MPP resolution,
        lazily built and cached per scene (reset on reset()/change_scene()) -- shared by
        get_random_point below. Furniture tops baked into the navmesh (navmesh_include_static_objects)
        form small islands within the floor's height slab; only the largest island is kept so spawns
        never land on them."""
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
        self._gt_semantic_map_cache_key = None
        return

    def get_reward_range(self):
        return (-1.0, 1.0)


def configure_scene(habitat_cfg: Any, scenes: list, steps_per_episode: int, object_params: dict) -> None:
    """Points habitat_cfg at `scenes` (content_scenes -- see ExplorationNavDataset) and wires up
    the ObjectDetectorGTSensor from object_params, in place, before an ExplorationEnv is built
    from it."""
    scene_dataset_config = resolve_env(object_params["env_name"])
    with habitat_read_write(habitat_cfg):
        habitat_cfg.habitat.dataset.content_scenes = scenes
        habitat_cfg.habitat.dataset.scene_dataset_config = scene_dataset_config
        habitat_cfg.habitat.simulator.scene_dataset = scene_dataset_config
        habitat_cfg.habitat.task.lab_sensors = {
            "object_detector_gt": ObjectDetectorGTSensorConfig(**object_params),
            **habitat_cfg.habitat.task.lab_sensors,
        }


def set_sensor_resolution(habitat_cfg: Any, image_size: int) -> None:
    """Renders every agent sensor at image_size x image_size, in place."""
    with habitat_read_write(habitat_cfg):
        for sensor in habitat_cfg.habitat.simulator.agents.main_agent.sim_sensors.values():
            sensor.width = sensor.height = image_size


def build_env(scene_name: str, seed: int, max_steps: int, object_params: dict, image_size: Optional[int] = None) -> tuple[ExplorationEnv, Any]:
    """(ExplorationEnv on scene_name, its habitat config), not yet reset; its sensors render at image_size if given."""
    habitat_cfg = habitat.get_config(config_path="common/config/hssd-hab/default.yaml")
    if image_size is not None:
        set_sensor_resolution(habitat_cfg, image_size)
    with habitat_read_write(habitat_cfg):
        habitat_cfg.habitat.seed = seed
    configure_scene(habitat_cfg, [scene_name], max_steps, object_params)
    return ExplorationEnv(config=habitat_cfg), habitat_cfg


def build_agent(
    agent_name: str, rng: np.random.Generator, env: ExplorationEnv, habitat_cfg: Any, visibility_dist_m: float = 3.0,
    dump_location: str = "outputs/planner_dumps", exp_name: str = "exp",
) -> Agent:
    """Call reset(position, floor_height) on the returned FrontierExplorationAgent/HybridFrontierAgent
    (BaseExplorationAgent) once an episode is active, before the first act()."""
    if agent_name == "random":
        return RandomAgent(rng, [action for action in env.get_legal_actions() if action != "stop"])
    simulator_cfg = habitat_cfg.habitat.simulator
    fov = simulator_cfg.agents.main_agent.sim_sensors.rgb_sensor.hfov
    depth_cfg = simulator_cfg.agents.main_agent.sim_sensors.depth_sensor
    camera = DepthCamera(depth_cfg.hfov, depth_cfg.min_depth, depth_cfg.max_depth)
    if agent_name == "fbe":
        config = FrontierExplorationConfig(MapConfig(fov, visibility_dist_m=visibility_dist_m), DiscretePlannerConfig(simulator_cfg.turn_angle))
        return FrontierExplorationAgent(config, camera, rng, dump_location, exp_name)
    if agent_name == "hybrid":
        config = HybridConfig(MapConfig(fov), simulator_cfg.turn_angle, planner_config=DiscretePlannerConfig(simulator_cfg.turn_angle))
        return HybridFrontierAgent(config, camera, rng, dump_location, exp_name)
    raise ValueError(f"Unknown agent {agent_name!r}")
