import os
from typing import Any, cast

from omegaconf import read_write

import habitat # type: ignore
import habitat_sim
import numpy as np
from habitat import RLEnv # type: ignore
from habitat_baselines.common.baseline_registry import baseline_registry # type: ignore
from habitat.utils.visualizations import fog_of_war, maps # type: ignore
from gym import spaces
from hydra.core.config_store import ConfigStore
from habitat.tasks.nav.nav import NavigationGoal, NavigationTask, NavigationEpisode # type: ignore


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
        self._max_episode_steps = self._env._max_episode_steps


    def reset(self):
        self._elapsed_steps = 0
        self.episode_over = False
        
        self._env._reset_stats()

        if self._env._current_episode is not None:
            self._env._current_episode._shortest_path_cache = None

        if (
            self._env._episode_iterator is not None
            and self._env._episode_from_iter_on_reset
        ):
            self._env._current_episode = next(self._env._episode_iterator)

        self._env._episode_from_iter_on_reset = True
        self._env._episode_force_changed = False

        assert self._env._current_episode is not None, "Reset requires an episode"
        self._env.reconfigure(self._env._config)
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

    def get_env_name(self) -> str:
        return self._env.task.sensor_suite.get("bbsgt").env_name

    def get_vocab_name(self) -> str:
        return self._env.task.sensor_suite.get("bbsgt").vocab_name
    
    def _past_limit(self) -> bool:
        if (
            self._max_episode_steps != 0
            and self._max_episode_steps <= self._elapsed_steps
        ):
            return True

        return False

    def step(self, action) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        self._elapsed_steps += 1
        self._previous_action = action
        
        if self._env._episode_over or self._past_limit():
            obs = self.reset()
            return  obs, 0, True, {}
        
        obs = self._env.step(action)
        return  obs, 0, False, {}
    
    def teleport(self, agent_state: habitat_sim.AgentState) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        self._elapsed_steps += 1
        
        if self._env._episode_over or self._past_limit():
            obs = self.reset()
            return  obs, 0, True, {}
        
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
        return
    
    def get_reward_range(self):
        return (-1.0, 1.0)
