import os
import cv2
import numpy as np
import magnum as mn
import itertools

import habitat_sim
from habitat.core.env import Env # type: ignore
from habitat.config import read_write # type: ignore
from habitat.config.default import get_agent_config # type: ignore
from habitat.utils.visualizations import maps # type: ignore
from habitat.config.default_structured_configs import HabitatSimSemanticSensorConfig # type: ignore
from scipy.spatial.transform import Rotation as R

import habitat.sims.habitat_simulator.sim_utilities as sutils # type: ignore
from habitat_sim.agent.agent import AgentState

from common.hssd_od_open_voc.hssd_object_annotations import ObjectSemanticsHSSD
from common.utils.grid_utils import HabitatObjOccupancyGrid
from common.interfaces import DiscreteNavigationAction, Observations, Labels
from common.utils.pose_utils import get_yaw

def get_obj_from_id(
    sim: habitat_sim.Simulator,
    obj_id: int,
):
    rom = sim.get_rigid_object_manager()
    if rom.get_library_has_id(obj_id):
        return rom.get_object_by_id(obj_id)

    return None


def object_shortname_from_handle( object_handle: str) -> str:
    return object_handle.split("/")[-1].split(".")[0].split("_:")[0].split("_")[0]


class HSSD_OpenVoc_Env(Env):
    obj_id_to_objname: dict[int, str]

    def __init__(self, config, vocab_name="HSSD500"):
        super().__init__(config)
        self.object_annotations = ObjectSemanticsHSSD(vocab_name)
        self.goal_image = None
        self.update_scene()

    def get_scene_name(self,) -> str:
        return self.current_episode.scene_id.split("/")[-1]

    def get_episode_goal(self,) -> dict:
        return {
            "object_id": self.current_episode.goals[0].object_id,
            "object_shortname":  object_shortname_from_handle(
                self.current_episode.goals[0].object_name
            ),
            "view_points": self.current_episode.goals[0].view_points,
        }


    def change_scene(self, scene: str) -> None:
        scenes_dir = self._config.dataset.scenes_dir

        with read_write(self._config):
            self._config.simulator.scene = scenes_dir + "/" +  scene

        self._sim.reconfigure(self._config.simulator)
        self.update_scene()

    def set_goal_image(self, obs_rgb: np.ndarray):
        self.goal_image = obs_rgb

    def get_agent_state(self, ) -> AgentState:
        return self.sim.agents[0].state


    def reset(self) -> Observations:
        self._reset_stats()

        if self._current_episode is not None:
            self._current_episode._shortest_path_cache = None

        if (
            self._episode_iterator is not None
            and self._episode_from_iter_on_reset
        ):
            self._current_episode = next(self._episode_iterator)

        self._episode_from_iter_on_reset = True
        self._episode_force_changed = False

        assert self._current_episode is not None, "Reset requires an episode"

        old_scene_id = self.sim.config.sim_cfg.scene_id

        self._config = self._task.overwrite_sim_config(
            self._config, self.current_episode
        )
        self._sim.reconfigure(self._config.simulator, self.current_episode)

        if self._current_episode.scene_id != old_scene_id:
            self.update_scene()

        observations = self.task.reset(episode=self.current_episode)

        self._task.measurements.reset_measures(
            episode=self.current_episode,
            task=self.task,
            observations=observations,
        )

        return observations

    def get_episode_viewpoints(self) -> list[AgentState]:
        return [
            AgentState(
                position=vp["position"],
                rotation=vp["rotation"]
            )
            for vp in self.current_episode.info["viewpoints"]
        ]
    
    def teleport(self, agent_state: AgentState) -> None:
        self.sim.agents[0].set_state(agent_state)

    def get_obs_gt(self, agent_state: AgentState)-> tuple[Observations, Labels]:
        self.teleport(agent_state)

        sensor_obs = self.sim.get_sensor_observations()
        labels = self.decompose_frame(sensor_obs["semantic"])
        semantic_frame = self.colorize(sensor_obs["semantic"])

        rot_x, rot_y, rot_z, rot_w = (
            float(agent_state.rotation.x),  #type: ignore
            float(agent_state.rotation.y),  #type: ignore
            float(agent_state.rotation.z),  #type: ignore
            float(agent_state.rotation.w), #type: ignore
        )

        quat = np.array([rot_w, rot_x, rot_y, rot_z], dtype=np.float32)

        rotation_matrix = R.from_quat(quat).as_matrix()
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = rotation_matrix
        camera_pose[:3, 3] = agent_state.position.copy()
        

        yaw = get_yaw(rot_w, rot_x, rot_y, rot_z)

        undefined_depth = (sensor_obs["depth"] == 0)
        sensor_obs["depth"][undefined_depth] = self._config.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth
        depth = np.clip(
            sensor_obs["depth"], 
            self._config.simulator.agents.main_agent.sim_sensors.depth_sensor.min_depth, 
            self._config.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth, 
        )

        metrics = self.get_metrics()
        x, y, z = agent_state.position
        gps = np.array([-z, -x])

        observation = Observations(
            gps = gps,
            compass = np.array([yaw]),
            rgb = sensor_obs["rgb"][:,:,:3],
            depth = depth,
            semantic = semantic_frame,
            camera_pose = camera_pose,
            task_observations= {
                "collisions": {"is_collision": 0},
                "top_down_map": metrics["top_down_map"],
                "goal_image": self.goal_image if self.goal_image is not None else sensor_obs["rgb"][:,:,:3],
            }
        )
        return observation, labels

    
    def get_scenes_names(self,) -> list[str]:
        scenes_dir = self._config.dataset.scenes_dir
        content_scenes = self._config.dataset.content_scenes

        if content_scenes == ["*"]:
            return  [ x.replace(".scene_instance.json", "") for x in os.listdir(scenes_dir)]
        
        return content_scenes


    def get_oracle_object_occupancy_grid(self, meters_per_grid_pixel) -> HabitatObjOccupancyGrid:
        return HabitatObjOccupancyGrid(
            self.sim,
            meters_per_grid_pixel=meters_per_grid_pixel,
            class_mapping=self.get_class_mapping(),
            list_object_info=self.get_objects()
        )

    def get_class_mapping(self) -> dict[str, int]:
        return {class_name: i for i, class_name in enumerate(self.get_classes())}

    def get_objects(self,) -> list[dict]:
        out = []
        objid_to_class = self.get_objid_to_class()
        
        for obj_id, obj_name in self.obj_id_to_objname.items():
            obj = get_obj_from_id(self.sim, obj_id)
            class_name = objid_to_class[obj_id]

            aabb = obj.collision_shape_aabb # type: ignore
            min_v = aabb.min
            max_v = aabb.max

            # 8 local box corners
            corners_local = [
                mn.Vector3(c) # type: ignore
                for c in itertools.product(
                    [min_v.x, max_v.x],
                    [min_v.y, max_v.y],
                    [min_v.z, max_v.z],
                )
            ]

            # Transform to world space (preserves orientation)
            corners_world = [
                obj.rotation.transform_vector(c) + obj.translation # type: ignore
                for c in corners_local
            ]

            out.append({
                "object_id": obj_id,
                "obj_name": obj_name,
                "class_name": class_name,

                "position": obj.translation, # type: ignore
                "rotation": obj.rotation, # type: ignore
                "corners": [
                    (c.x, c.y, c.z)
                    for c in corners_world
                ],
            })
        
        return out
    
    def get_objid_to_class(self) -> dict[int, str]:
        return {obj_id: self.object_annotations.mapping_objname_class[obj_name] for obj_id, obj_name in self.obj_id_to_objname.items()} 

    def update_scene(self) -> None:
        self.setup_obj_id_to_objname()
        self.setup_semantic_labels()

    def get_class(self, obj_name: str):
        return self.object_annotations.mapping_objname_class[obj_name]

    def get_classes(self) -> list[str]:
        return sorted(set(self.object_annotations.mapping_objname_class.values()))
    
    def colorize(self, semantic_obs: np.ndarray) -> np.ndarray:
        class2color = self.object_annotations.class2color
        objid_to_class = self.get_objid_to_class()

        color_map = np.array(
            [(0,0,0)] + [class2color[class_name] for obj_id, class_name in sorted(objid_to_class.items())]
        ).astype(np.uint8)
        colorized = color_map[semantic_obs].astype(np.uint8)

        return colorized


    def decompose_frame(self, semantic_obs: np.ndarray) -> Labels:
        instances = []
        
        values = np.unique(semantic_obs)
        objid_to_class = self.get_objid_to_class()

        def flatten_contour(countour_of_pairs):
            out = []
            for i in range(len(countour_of_pairs)):
                out.append(countour_of_pairs[i,0])
                out.append(countour_of_pairs[i,1])
            return out
        
        for semantic_id in values:
            if semantic_id == 0:
                continue

            mask = (semantic_obs == semantic_id).astype("uint8")

            obj_id = semantic_id - 1
            obj_name = self.obj_id_to_objname[obj_id]
            class_name = objid_to_class[obj_id]

            if class_name == "unknown":
                continue

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                continue

            # list of polygons (one per contour)
            list_mask_polygons = [
                flatten_contour(contour.reshape(-1, 2))
                for contour in contours
                if len(contour) >= 3  # valid polygon
            ]

            if not list_mask_polygons:
                continue

            # bounding box over all contours / full object
            x, y, w, h = cv2.boundingRect(
                np.vstack(contours)
            )

            bbx_area = w * h
            mask_area = int(np.sum(mask))

            instances.append({
                "class_name": class_name,
                "object_id": obj_id,
                "obj_name": obj_name,
                "source_class_name": self.object_annotations.source_mapping_objname_class.get(obj_name, "unknown"),
                "mask_polygons": list_mask_polygons,
                "bounding_box": (x, y, w, h),
                "bbx_area": bbx_area,
                "mask_area": mask_area
            })

        return Labels(instances=instances)

    def setup_semantic_labels(self,)  -> None:
        rom = self.sim.get_rigid_object_manager()

        for _, handle in enumerate(rom.get_object_handles()):
            obj = rom.get_object_by_handle(handle)
            for node in obj.visual_scene_nodes:
                node.semantic_id = 1 + obj.object_id
    
    def setup_obj_id_to_objname(self,) -> None:
        def objname_from_handle( object_handle: str) -> str:
            return object_handle.split("/")[-1].split(".")[0].split("_:")[0].split("_")[0]

        # setup the dictionnary obj_id_to_objname
        self.obj_id_to_objname = {}

        for obj_id, obj_handle in sutils.get_all_object_ids(self.sim).items():
            objname = objname_from_handle(obj_handle)
            self.obj_id_to_objname[obj_id] = objname
            assert objname in self.object_annotations.mapping_objname_class, f"Object name {objname} not found in annotations mapping. Please check the object config files and the annotations csv file."
