import itertools
import os
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
from habitat.tasks.nav.nav import NavigationGoal, NavigationTask, NavigationEpisode # type: ignore

from common.env_utils.object_annotations import get_all_objects
from common.planning.skeleton import grid_path
from common.utils.pose_utils import quaternion_from_rpy


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
        self._follower = None


    def reset(self, episode: Optional[NavigationEpisode] = None, rng: Optional[np.random.Generator] = None):
        r"""Resets the sim to `episode` if given, else re-initializes the current one --
        never consults `self._env._episode_iterator` (see `self.episodes` to enumerate
        candidates), so the caller always controls episode/scene selection explicitly.

        Places the agent at a random navigable point: an episode's own start_position/
        start_rotation are meaningless synthetic placeholders (see common/env_utils/dataset.py::
        ExplorationNavDataset), historically safe only because every caller teleported before
        capturing anything -- RandomAgent (common/agents/random_agent.py) no longer does, since
        it plans from the agent's live pose (env.find_shortest_path), so it needs reset() to
        already have placed the agent somewhere real."""
        self._elapsed_steps = 0
        self.episode_over = False
        self._tdmap = None
        self._follower = None

        self._env._reset_stats()

        if self._env._current_episode is not None:
            self._env._current_episode._shortest_path_cache = None

        if episode is not None:
            self._env.current_episode = episode

        self._env._episode_from_iter_on_reset = True
        self._env._episode_force_changed = False

        assert self._env._current_episode is not None, "Reset requires an episode"
        self._env.reconfigure(self._env._config)

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

    def set_done(self, done) -> None:
        self._env.current_episode.episode_over = done

    def get_map_bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        l, u = self._env.sim.pathfinder.get_bounds()
        return l, u

    def get_last_reward(self) -> float:
        return self.last_reward

    def get_tdmap(self) -> np.ndarray:
        return self._env.task.sensor_suite.get("bbsgt").annotation.object_occupancy_grid.topdown_view

    def get_object_occupancy(self) -> np.ndarray:
        return self._env.task.sensor_suite.get("bbsgt").annotation.object_occupancy_grid

    _GROUND_FLOOR_MPP = 0.125  # meters/cell, matches HabitatObjOccupancyGrid's own resolution
    _STEP_WALK_METERS = 0.25  # waypoint spacing along find_shortest_path_waypoints's route (embodied-active-learning-od's own grid spacing)

    def get_random_point(self, rng: Optional[np.random.Generator] = None, min_distance: float = 0.0, max_retries: int = 100) -> np.ndarray:
        pathfinder = self._env.sim.pathfinder
        lower_bound, _ = pathfinder.get_bounds()
        
        if self._tdmap is None:
            self._tdmap = pathfinder.get_topdown_view(
                meters_per_pixel=self._GROUND_FLOOR_MPP, height=lower_bound[1]
            ).astype(np.uint8)

        rows, cols = np.nonzero(self._tdmap)

        k = 0

        while k < max_retries:
            idx = rng.integers(len(rows)) if rng is not None else np.random.randint(len(rows))
            z, x = maps.from_grid(rows[idx], cols[idx], self._tdmap.shape, pathfinder=pathfinder)
            snapped = pathfinder.snap_point(mn.Vector3(x, lower_bound[1], z))

            if min_distance > 0:
                agent_state = self._env.sim.agents[0].get_state()
                dist = np.linalg.norm(np.array([snapped.x, snapped.y, snapped.z]) - np.array(agent_state.position))
                if dist < min_distance:
                    continue
            k += 1

        return np.array(snapped)
    
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
        r"""Random navigable point within `radius` of `position`, or None if none found."""
        point = np.array(self._env.sim.pathfinder.get_random_navigable_point_near(mn.Vector3(*position), radius, max_tries=max_tries))
        return None if np.isnan(point).any() else point

    def find_shortest_path_waypoints(self, start: np.ndarray, end: np.ndarray) -> Optional[list[np.ndarray]]:
        pathfinder = self._env.sim.pathfinder
        lower_bound, _ = pathfinder.get_bounds()
        if self._tdmap is None:
            self._tdmap = pathfinder.get_topdown_view(
                meters_per_pixel=self._GROUND_FLOOR_MPP, height=lower_bound[1]
            ).astype(np.uint8)

        start_row, start_col = maps.to_grid(start[2], start[0], self._tdmap.shape, pathfinder=pathfinder)
        end_row, end_col = maps.to_grid(end[2], end[0], self._tdmap.shape, pathfinder=pathfinder)

        path_px = grid_path(
            self._tdmap, (start_col, start_row), (end_col, end_row),
            step_size=self._STEP_WALK_METERS / self._GROUND_FLOOR_MPP,
        )
        if len(path_px) < 2:
            return None

        points = []
        for col, row in path_px:
            z, x = maps.from_grid(row, col, self._tdmap.shape, pathfinder=pathfinder)
            points.append(np.array(pathfinder.snap_point(mn.Vector3(x, lower_bound[1], z))))
        return points

    def find_shortest_path(self, goal: np.ndarray) -> list:
        if self._follower is None:
            self._follower = self._env.sim.make_greedy_follower(
                0, stop_key="stop", forward_key="move_forward", left_key="turn_left", right_key="turn_right",
            )
        try:
            return self._follower.find_path(np.array(goal))
        except habitat_sim.errors.GreedyFollowerError:
            return []

    def get_env_name(self) -> str:
        return self._env.task.sensor_suite.get("bbsgt").env_name

    def get_vocab_name(self) -> str:
        return self._env.task.sensor_suite.get("bbsgt").vocab_name

    def get_agent_state(self) -> habitat_sim.AgentState:
        r"""Current position/rotation -- e.g. for an Agent (common/agents/) that needs to know
        where it is now to decide the next AgentState to move to."""
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

    @property
    def original_action_space(self) -> spaces.space: # type: ignore
        return self.action_space

    def change_scene(self, scene: str) -> None:
        with read_write(self._env._config):
            self._env._config.simulator.scene = self._env._config.dataset.scenes_dir + "/" +  scene

        self._env._sim.reconfigure(self._env._config.simulator)
        self._env.task.sensor_suite.get("bbsgt").setup_semantic_labels()
        self._tdmap = None
        self._follower = None
        return

    def get_reward_range(self):
        return (-1.0, 1.0)
