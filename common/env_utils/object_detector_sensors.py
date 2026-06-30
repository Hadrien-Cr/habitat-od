import itertools
import logging
from typing import Any

from attr import dataclass
import cv2
from habitat_sim.agent.agent import AgentState
import habitat_sim # type: ignore
import torch
import habitat # type: ignore
import numpy as np
from gym import spaces
from habitat.core.registry import registry # type: ignore[import]
import habitat.sims.habitat_simulator.sim_utilities as sutils # type: ignore
from habitat.config.default_structured_configs import LabSensorConfig, ObjectDetectorGTSensorConfig  # type: ignore[import]
from detectron2.structures import Boxes, Instances
import magnum as mn

from common.env_utils.hssd_object_annotations import ObjectSemanticsHSSD
from common.env_utils.vocab_constants import *
from common.utils.grid_utils import HabitatObjOccupancyGrid

log = logging.getLogger(__name__)


def get_obj_from_id(sim: habitat_sim.Simulator,obj_id: int,):
    rom = sim.get_rigid_object_manager()
    if rom.get_library_has_id(obj_id):
        return rom.get_object_by_id(obj_id)
    return None

def object_shortname_from_handle(object_handle: str) -> str:
    return object_handle.split("/")[-1].split(".")[0].split("_:")[0].split("_")[0]

def get_all_objects(
    sim: habitat_sim.Simulator,
):
    managers = [
        sim.get_rigid_object_manager(),
        sim.get_articulated_object_manager(),
    ]
    all_objects = []
    for mngr in managers:
        all_objects.extend(mngr.get_objects_by_handle_substring().values())
    return all_objects


def get_objects_info(sim, obj_name_to_class: dict[str, str], fallback_obj_name_to_class: dict[str, str]) -> list[dict]:
    out = []

    for obj_id, obj_handle in sutils.get_all_object_ids(sim).items():
        obj_name = object_shortname_from_handle(obj_handle)

        obj = get_obj_from_id(sim, obj_id)

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

        corners_world = [
            obj.rotation.transform_vector(c) + obj.translation # type: ignore
            for c in corners_local
        ]

        out.append({
            "object_id": obj_id,
            "obj_name": obj_name,
            "class_name": obj_name_to_class.get(obj_name, "/u:" + fallback_obj_name_to_class[obj_name]),
            "position": obj.translation, # type: ignore
            "rotation": obj.rotation, # type: ignore
            "center": np.array(obj.collision_shape_aabb.center), # type: ignore
            "corners": [
                (c.x, c.y, c.z)
                for c in corners_world
            ],
        })
    return out


@registry.register_sensor
class ObjectDetectorGTSensor(habitat.Sensor):
    scene: str
    env_name: str
    object_info_list: list[dict]
    object_occupancy_grid: HabitatObjOccupancyGrid
    area_thr: float
    filter_occluded: bool # whether to filter out objects that are occluded 50% occluded

    
    def __init__(self, sim, config: ObjectDetectorGTSensorConfig, **kwargs: Any) -> None:
        super().__init__(config=config)
        self._sim = sim
        self.scene = ""
        self.object_info_list = None
        self.object_occupancy_grid = None

        self.env_name = config.env_name
        self.area_thr = config.area_thr
        self.filter_occluded = config.filter_occluded

        if self.env_name.startswith("HSSD-HAB"):
            self.object_annotations = ObjectSemanticsHSSD(self.env_name.replace("HSSD-HAB/", ""))
        else:
            raise NotImplementedError(f"Environment {self.env_name} not supported for object detector gt sensor")

    def get_classes(self) -> list[str]:
        if self.env_name.startswith("HSSD-HAB"):
            return self.object_annotations.classes
        else:
            raise NotImplementedError(f"Environment {self.env_name} not supported for object detector gt sensor")

    def semantic_id_to_classid(self, semantic_id: int) -> int:
        if semantic_id < 1000:
            raise ValueError("Semantic id is not a valid object id")

        if self.env_name.startswith("HSSD-HAB"):
            obj_id = semantic_id - 1000
            return self.objid_to_class_id[obj_id]
        else:
            raise NotImplementedError(f"Environment {self.env_name} not supported for object detector gt sensor")

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "bbsgt"

    # Defines the type of the sensor
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return habitat.SensorTypes.MEASUREMENT

    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(3,),
            dtype=np.float32,
        )

    # This is called whenver reset is called or an action is taken
    def get_observation(self, *args: Any, **kwargs: Any):
        semantic_obs = kwargs['observations']['semantic']
        assert self.scene == self._sim.curr_scene_name, "Scene has changed since last setup of semantic labels. This should not happen, make sure to call setup_semantic_labels on the environment wrapper when changing scenes."
        return self.decompose_frame(semantic_obs)    

    def decompose_frame(self, semantic_obs: np.ndarray) -> dict:
        detections = []
        classes = self.get_classes()
        values = [v for v in set(semantic_obs.ravel().tolist()) if v >= 1000]

        for semantic_id in values:
            if semantic_id < 1000:
                assert semantic_id == 0, f"Unexpected semantic id {semantic_id} in semantic observation"
                continue
            
            mask = (semantic_obs == semantic_id).astype("uint8")
            # mask = cv2.morphologyEx(mask, op=cv2.MORPH_OPEN, kernel=np.ones((3, 3), dtype=np.uint8), iterations=1)
            if self.filter_occluded and not self.object_occupancy_grid.object_is_visible(semantic_id - 1000, agent_state=self._sim.get_agent_state(), min_depth=0.0, max_depth=5.0, camera_hfov=90, n_rays=5, min_object_fov=5.0):
                continue
            
            class_id = self.semantic_id_to_classid(semantic_id)

            assert class_id < len(classes), f"Class id {class_id} for semantic id {semantic_id} is out of range for vocabulary {self.get_classes()}"

            if classes[class_id] == "undefined":
                continue
            
            x, y, w, h = cv2.boundingRect(mask)
            mask_area = np.sum(mask)
            bbx_area = w * h

            if min(mask_area*2, bbx_area) < self.area_thr:
                continue

            detections.append({
                "class_id": torch.tensor(class_id).unsqueeze(0),
                "mask": torch.from_numpy(mask.squeeze()).bool().unsqueeze(0),
                "bounding_box": torch.tensor([x, y, x + w, y + h]).unsqueeze(0),
                "info": {
                    "object_id": semantic_id - 1000,
                    "env_name": self.env_name,
                }
            })
        
        if not detections:
            return {'instances': Instances(
                image_size=(semantic_obs.shape[0], semantic_obs.shape[1]),
                pred_boxes=Boxes(torch.zeros((0,4))),
                pred_classes=torch.zeros((0,)).long(),
                scores=torch.zeros((0,)),
                pred_masks=torch.zeros((0, semantic_obs.shape[0], semantic_obs.shape[1])).bool(),
            )}
        get_center = lambda bbx: (bbx[0] + 0.5 * (bbx[2] - bbx[0]), bbx[1] + 0.5 * (bbx[3] - bbx[1]))
        sorted_detections = sorted(detections, key=lambda d: get_center(d["bounding_box"][0])[0] + get_center(d["bounding_box"][0])[1], reverse=False)
        pred_boxes = torch.cat([d["bounding_box"] for d in sorted_detections])
        pred_classes = torch.cat([d["class_id"] for d in sorted_detections])
        pred_masks = torch.cat([d["mask"] for d in sorted_detections])
        infos = np.array([d["info"] for d in sorted_detections])

        return {'instances': Instances(
            image_size=(semantic_obs.shape[0], semantic_obs.shape[1]),
            pred_boxes=Boxes(pred_boxes),
            pred_classes=pred_classes,
            scores=torch.ones(len(sorted_detections)),
            pred_masks=pred_masks,
            infos=infos,
        )}

    def setup_semantic_labels(self,)  -> None:
        self.scene = self._sim.curr_scene_name

        if self.env_name.startswith("HSSD-HAB"):
            class2int = {c: i for i, c in enumerate(self.get_classes())}

            self.objid_to_class_id = {}

            for obj in get_all_objects(self._sim):
                obj_name = object_shortname_from_handle(obj.handle)

                if obj_name not in self.object_annotations.mapping_objname_class:
                    for node in obj.visual_scene_nodes:
                        node.semantic_id = 0
                    continue
                
                class_name = self.object_annotations.mapping_objname_class[obj_name]
                class_id = class2int[class_name]

                for node in obj.visual_scene_nodes:
                    node.semantic_id = 1000 + obj.object_id

                self.objid_to_class_id[obj.object_id] = class_id

            self.object_info_list = get_objects_info(self._sim, self.object_annotations.mapping_objname_class, self.object_annotations.source_mapping_objname_class)
            self.object_occupancy_grid = HabitatObjOccupancyGrid(self._sim, meters_per_grid_pixel=0.125, list_object_info=self.object_info_list)
        
        else:
            raise NotImplementedError
        
        
    def decompose_scene(self) -> dict:
        assert self.scene == self._sim.curr_scene_name, "Scene has changed since last setup of semantic labels. This should not happen, make sure to call setup_semantic_labels on the environment wrapper when changing scenes."

        classes = self.get_classes()
        rom = self._sim.get_rigid_object_manager()

        mapping = {}
        for _, obj_handle in enumerate(rom.get_object_handles()):
            obj_name = object_shortname_from_handle(obj_handle)

            if obj_name not in self.object_annotations.mapping_objname_class:
                continue
            
            obj = rom.get_object_by_handle(obj_handle)
            class_name = classes[self.semantic_id_to_classid(obj.semantic_id)]
            mapping[obj.object_id] = class_name

        return {
            "objects": mapping,
        }
    
