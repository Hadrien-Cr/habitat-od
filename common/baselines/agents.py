import gc
import os
from typing import Any
import cv2
from tqdm import tqdm
import numpy as np
import math
from PIL import Image

from habitat.core.simulator import Observations # type: ignore
from habitat_sim import AgentState # type: ignore
from habitat.sims.habitat_simulator.actions import HabitatSimActions # type: ignore
from habitat.tasks.nav.nav import NavigationGoal, NavigationTask # type: ignore
from habitat_baselines.agents.simple_agents import GoalFollower, RandomAgent # type: ignore
from habitat_baselines.common.base_trainer import BaseRLTrainer # type: ignore
from habitat_baselines.common.baseline_registry import baseline_registry # type: ignore
from habitat.core.environments import get_env_class # type: ignore
from habitat.utils.geometry_utils import ( # type: ignore
    angle_between_quaternions,
    quaternion_from_coeff,
    quaternion_from_two_vectors,
    quaternion_rotate_vector,
)
import magnum as mn
from habitat.utils.visualizations import fog_of_war, maps # type: ignore
from habitat_sim.utils import common as sim_utils

from common.env_utils.habitat_utils import construct_envs, VectorEnv # type: ignore
from common.utils.data_utils import save_obs, load_data
from common.utils.plot_utils import plot_segmentation, plot_mask, plot_array, plot_semantic_2d_map
from common.planning.skeleton import do_plan
from common.utils.pose_utils import quaternion_from_rpy


SUCCESS_DISTANCE = 1.0



class Baseline(BaseRLTrainer):
    def __init__(self, config, agent_class, **kwargs) -> None:
        assert "baseline_params" in config
        super().__init__(config)

        self.config = config
        self.kwargs = kwargs
        self.rng_gen = np.random.default_rng(config.habitat.seed)   
        self.visualize = self.config.baseline_params.visualize  # visualize maps
        self.agent = agent_class(SUCCESS_DISTANCE, self.config.habitat.task.goal_sensor_uuid)

    def _init_train(self) -> VectorEnv:
        envs = self._init_envs()

        self.current_observations = envs.reset()
        self.last_observations = []
        self.current_dones = []
        self.current_infos = []
        self.current_rewards = []
        return envs

    def _step(self, envs: VectorEnv) -> None:
        actions = []
        for index_env in range(envs.num_envs):
            act = self.agent.act(self.current_observations[index_env])
            actions.append(act)
            envs.async_step_at(index_env, act)

        results = [envs.wait_step_at(index_env) for index_env in range(envs.num_envs)]

        self.last_observations = self.current_observations
        self.current_observations = [r[0] for r in results]
        self.current_rewards = [r[1] for r in results]
        self.current_dones = [r[2] for r in results]
        self.current_infos = [r[3] for r in results]

        self.current_steps += 1

    def _init_envs(self, config=None, kwargs=None) -> VectorEnv:
        if config is None:
            config = self.config
        if kwargs is None:
            kwargs = self.kwargs
        self.num_steps_done = 0
        env_class = get_env_class(config.habitat_baselines.env_name)
        assert env_class is not None, f"Environment class for {config.habitat_baselines.env_name} not found"
        envs = construct_envs(config, env_class, True, **kwargs) # type: ignore
        self.current_steps = np.zeros(envs.num_envs)
        return envs

    def train(self) -> None:
        pass

    def plan(self, idx) -> None:
        pass

    def init_collection(self) -> None:
        pass

    def collect(self, dataset_path: str, steps_per_episode: int) -> list[list[str]]:
        self.envs = self._init_train()
        os.makedirs(dataset_path, exist_ok=True)

        self.init_collection()
        collected_observations_paths = []
        
        nb_episodes: int = self.envs.number_of_episodes[0] # type: ignore
        pbar = tqdm(total=nb_episodes * steps_per_episode * self.envs.num_envs, desc="Generating data")
    
        for _ in range(nb_episodes * steps_per_episode): 
            for idx in range(self.envs.num_envs):
                self.plan(idx)

            self._step(self.envs)

            for idx in range(self.envs.num_envs):
                obs = self.current_observations[idx]
                done = self.current_dones[idx]
                episode = self.envs.current_episodes()[idx]
                step = self.current_steps[idx]

                if done:
                    self.current_steps[idx] = 0
                
                if self.visualize and idx == 0 and step % 100 == 1:
                    self.do_visualize(idx, dataset_path=dataset_path)

            paths = save_obs(
                dataset_path, 
                int(episode.episode_id),
                [self.current_observations[idx] for idx in range(self.envs.num_envs)], 
                int(step), 
                modalities=["rgb", "bbsgt"]
            )
            collected_observations_paths.append(paths)
            pbar.update(self.envs.num_envs)
            
        del self.current_steps
        del self.current_dones
        del self.current_observations

        self.envs.close()
        return sorted(collected_observations_paths)


    def do_visualize(self, env_idx: int, dataset_path: str):
        obs = self.current_observations[env_idx]
        episode = self.envs.current_episodes()[env_idx]
        step = self.current_steps[env_idx]
        gt_instances = obs["bbsgt"]["instances"]

        from detectron2.data import MetadataCatalog
        env_name = self.envs.call_at(0, "get_env_name")
        classes = MetadataCatalog.get(env_name.replace("HSSD-HAB/", "")).thing_classes
        colors = MetadataCatalog.get(env_name.replace("HSSD-HAB/", "")).thing_colors
        
        # save visualization of the segmentation gt
        fname = f"episode_{int(episode.episode_id):06d}_modality_vis_step_{int(step):05d}_id_{env_idx}.png"
        rgb = obs['rgb']
        im = plot_segmentation(rgb, pred_instances=gt_instances, gt_instances = None, classes=classes, colors=colors)

        os.makedirs("datadump/vis/" + dataset_path.split("/")[-1], exist_ok=True)
        im.save("datadump/vis/" + dataset_path.split("/")[-1] + "/" + fname)
    

        # save object_occupancy map
        tdmap = self.envs.call_at(0, "get_tdmap")
        object_occupancy_grid = self.envs.call_at(0, "get_object_occupancy")
        list_object_info = object_occupancy_grid.list_object_info

        obj_id_to_class = {object_info["object_id"]: object_info["class_name"] for object_info in list_object_info}
        obj_id_to_color = {object_info["object_id"]: colors[classes.index(object_info["class_name"])]  for object_info in list_object_info if object_info["class_name"] in classes}

        im = plot_semantic_2d_map(
            object_occupancy_grid.topdown_view, 
            object_occupancy_grid.obj_occupancy_td_view, 
            classes=obj_id_to_class, colors=obj_id_to_color, 
            meters_per_grid_pixel=object_occupancy_grid.meters_per_grid_pixel, 
            scale=10
        )
        fname = f"episode_{int(episode.episode_id):06d}_modality_obj_step_{int(step):05d}_id_{env_idx}.png"
        os.makedirs("datadump/maps/" + dataset_path.split("/")[-1], exist_ok=True)
        im.save("datadump/maps/" + dataset_path.split("/")[-1] + "/" + fname)

        # save tdmap
        tdmap = self.envs.call_at(0, "get_tdmap")
        lower_bound, upper_bound = self.envs.call_at(env_idx, "get_map_bounds")

        agent_position = obs['position']['position']
        agent_rotation = obs['position']['orientation']

        recolor_map = np.array(
            [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
        )

        top_down_map = recolor_map[tdmap]

        tdmap_resolution = (
            abs(upper_bound[2] - lower_bound[2]) / tdmap.shape[0],
            abs(upper_bound[0] - lower_bound[0]) / tdmap.shape[1],
        )
        agent_pixel_pos = (
            int((agent_position[2] - lower_bound[2]) / tdmap_resolution[0]),
            int((agent_position[0] - lower_bound[0]) / tdmap_resolution[1])
        )
        
        agent_forward = sim_utils.quat_to_magnum(agent_rotation).transform_vector(
            mn.Vector3(0, 0, -1.0) # type: ignore
        ) 
        agent_orientation = math.atan2(agent_forward[0], agent_forward[2])
        top_down_map = maps.draw_agent(
            top_down_map,
            agent_center_coord=agent_pixel_pos,
            agent_rotation=agent_orientation,
            agent_radius_px=4,
        )
        im = plot_array(top_down_map)
        fname = f"episode_{int(episode.episode_id):06d}_modality_tdmap_step_{int(step):05d}_id_{env_idx}.png"
        os.makedirs("datadump/maps/" + dataset_path.split("/")[-1], exist_ok=True)
        im.save("datadump/maps/" + dataset_path.split("/")[-1] + "/" + fname)


@baseline_registry.register_trainer(name="randombaseline")
class RandomBaseline(Baseline):
    def __init__(self, config, **kwargs)  -> None:
        super().__init__(config, agent_class=RandomAgent, **kwargs)


class BounceAgent(RandomAgent):
    def __init__(self, success_distance: float, goal_sensor_uuid: str) -> None:
        super().__init__(success_distance, goal_sensor_uuid)
        self.turn_count = 0

    def act(self, observations: Observations) -> dict[str, int]:
        action = HabitatSimActions.move_forward

        # if collision with navmesh and not turning already
        if observations['agent_collision_sensor'] and self.turn_count == 0:
            self.turn_count = 6

        if self.turn_count > 1:
            action = (
                HabitatSimActions.turn_left
            )  # TODO: choose turning side based on tangent angle wrt the obstacle

            self.turn_count -= 1
        elif self.turn_count == 1:
            action = HabitatSimActions.turn_right
            self.turn_count -= 1

        return {"action": action}


@baseline_registry.register_trainer(name="bouncebaseline")
class BounceBaseline(Baseline):
    def __init__(self, config, **kwargs) -> None:
        super().__init__(config, agent_class=BounceAgent, **kwargs)


class RotateAgent(RandomAgent):
    def __init__(self, success_distance: float, goal_sensor_uuid: str) -> None:
        super().__init__(success_distance, goal_sensor_uuid)

    def act(self, observations: Observations) -> dict[str, int]:
        return {"action": HabitatSimActions.turn_left}


@baseline_registry.register_trainer(name="rotatebaseline")
class RotateBaseline(Baseline):
    def __init__(self, config, **kwargs) -> None:
        super().__init__(config, agent_class=RotateAgent, **kwargs)


@baseline_registry.register_trainer(name="randomgoalsbaseline")
class RandomGoalsBaseline(Baseline):
    def __init__(self, config, **kwargs) -> None:
        super().__init__(config, agent_class=GoalFollower, **kwargs)     

    def goto_next_subgoal(self, idx) -> None:
        dist_subgoal, angle_subgoal = self.current_observations[idx]['pointgoal_with_gps_compass']
        
        update_subgoal = (
            self.got_new_plan[idx] or dist_subgoal < SUCCESS_DISTANCE or self.current_steps[idx] % 10 == 0
        ) 

        if update_subgoal and len(self.sub_goals[idx]) > 0:
            new_sub_goal = self.sub_goals[idx][0]
            self.sub_goals[idx] = self.sub_goals[idx][1:]
            self.got_new_plan[idx] = False
            self.envs.call_at(
                idx, "set_goals", {"data": [NavigationGoal(position=new_sub_goal)]}
            )

    def compute_new_goals(self, idx: int) -> None:
        dist_subgoal, angle_subgoal = self.current_observations[idx]['pointgoal_with_gps_compass']
        
        if (
            not (len(self.sub_goals[idx]) == 0 and dist_subgoal < SUCCESS_DISTANCE)
            and not self.path_step[idx] == -1
            and self.path_step[idx] % self.config.baseline_params.replanning_steps != 0
        ):
            return
            
        self.path_step[idx] = 0

        # get the current td map
        tdmap = self.envs.call_at(idx, "get_tdmap")
        lower_bound, upper_bound = self.envs.call_at(idx, "get_map_bounds")
        agent_position = self.current_observations[idx]['position']['position']

        tdmap_resolution = (
            abs(upper_bound[2] - lower_bound[2]) / tdmap.shape[0],
            abs(upper_bound[0] - lower_bound[0]) / tdmap.shape[1],
        )
        start_pixel_pos = [
            int((agent_position[0] - lower_bound[0]) / tdmap_resolution[1]),
            int((agent_position[2] - lower_bound[2]) / tdmap_resolution[0]),
        ]

        # compute goal proposal
        dilation = 1 + int(tdmap_resolution[0] * SUCCESS_DISTANCE)
        if dilation > 1:
            dilated_map = cv2.dilate((255 * tdmap).astype(np.uint8), np.ones((dilation, dilation), dtype=np.uint8), iterations=1)
        else:
            dilated_map = (255 * tdmap).astype(np.uint8)

        pixel_path, skel = do_plan(dilated_map, start_pixel_pos, start_pixel_pos, random_goal=True)
        plot_mask(skel == 255).save("skel.jpg")
        plot_mask(dilated_map == 255).save("dilated_map.jpg")
        plot_mask(tdmap).save("tdmap.jpg")

        assert len(pixel_path) > 0, "No path found by planner, check the map and the start position"
        path = [
            [lower_bound[0] + p[0] * tdmap_resolution[1], agent_position[1], lower_bound[2] + p[1] * tdmap_resolution[0]]
            for p in pixel_path
        ]
        goal_pixel_pos = pixel_path[-1]

        self.sub_goals[idx] = path

        while len(self.sub_goals[idx]) > 1 and np.linalg.norm(np.array(self.sub_goals[idx][0]) - np.array(agent_position)) < SUCCESS_DISTANCE:
            self.sub_goals[idx] = self.sub_goals[idx][1:]

        if len(self.sub_goals[idx]) > 0:
            self.got_new_plan[idx] = True
        else:
            self.compute_new_goals(idx)

        if self.visualize and idx == 0:
            out_img = (tdmap * 255).astype(np.uint8).copy()
            out_img = cv2.cvtColor(out_img, cv2.COLOR_GRAY2BGR)
            cv2.circle(out_img, start_pixel_pos, 20, (255, 0, 0), 4)
            cv2.circle(out_img, goal_pixel_pos, 20, (0, 0, 255), 4)

            pairs = list(zip(path[:-1], path[1:]))

            for (s, g) in pairs:
                p1 = [
                    int((s[0] - lower_bound[0]) / tdmap_resolution[1]),
                    int((s[2] - lower_bound[2]) / tdmap_resolution[0]),
                ]
                p2 = [
                    int((g[0] - lower_bound[0]) / tdmap_resolution[1]),
                    int((g[2] - lower_bound[2]) / tdmap_resolution[0]),
                ]
                cv2.line(out_img, p1, p2, (0, 255, 0), 2)

            os.makedirs(f"datadump/goal_env", exist_ok=True)
            cv2.imwrite(os.path.abspath(f"datadump/goal_env/step{self.num_steps_done}.jpg"), out_img)
    
    def init_collection(self) -> None:
        self.sub_goals = [[] for _ in range(self.envs.num_envs)]
        self.path_step = [-1 for _ in range(self.envs.num_envs)]

    def plan(self, idx) -> None:
        self.compute_new_goals(idx)
        self.goto_next_subgoal(idx)
        
    


@baseline_registry.register_trainer(name="randomteleportbaseline")
class RandomTeleport(Baseline):
    def __init__(self, config, **kwargs) -> None:
        super().__init__(config, agent_class=RotateAgent, **kwargs)

    def get_random_agent_state(self, idx, obs) -> AgentState:
        tdmap = self.envs.call_at(idx, "get_tdmap")
        lower_bound, upper_bound = self.envs.call_at(idx, "get_map_bounds")
        tdmap_resolution = (
            abs(upper_bound[2] - lower_bound[2]) / tdmap.shape[0],
            abs(upper_bound[0] - lower_bound[0]) / tdmap.shape[1],
        )
        valid_positions = np.argwhere(tdmap == 1)
        assert np.sum(tdmap) > 0, "No valid positions found in the top-down map"
        random_pixel = valid_positions[self.rng_gen.integers(0, len(valid_positions))]
        random_position = np.array([
            lower_bound[0] + random_pixel[1] * tdmap_resolution[1],
            obs['position']['position'][1],
            lower_bound[2] + random_pixel[0] * tdmap_resolution[0],
        ])
        random_yaw = self.rng_gen.uniform(0, 2 * np.pi)
        agent_state = AgentState()
        agent_state.position = random_position
        agent_state.rotation = quaternion_from_rpy(0, 0, random_yaw)
        return agent_state

    def _step(self, envs: VectorEnv) -> None:
        results = []
        for index_env in range(envs.num_envs):
            agent_state = self.get_random_agent_state(index_env, self.current_observations[index_env])
            r = envs.call_at(index_env, "teleport", {"agent_state": agent_state})
            results.append(r)

        self.last_observations = self.current_observations
        self.current_observations = [r[0] for r in results]
        self.current_rewards = [r[1] for r in results]
        self.current_dones = [r[2] for r in results]
        self.current_infos = [r[3] for r in results]

        self.current_steps += 1