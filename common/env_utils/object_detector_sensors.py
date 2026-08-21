import itertools
import logging
from typing import Any, Optional

import cv2
from habitat_sim.agent.agent import AgentState
import habitat_sim # type: ignore
import torch
import habitat # type: ignore
import numpy as np
from gym import spaces
from habitat.core.registry import registry # type: ignore[import]
import habitat.sims.habitat_simulator.sim_utilities as sutils # type: ignore
from habitat.config.default_structured_configs import ObjectDetectorGTSensorConfig  # type: ignore[import]
from detectron2.structures import Boxes, Instances
import magnum as mn

from common.env_utils.hssd_object_annotations import ObjectSemanticsHSSD
from common.env_utils.vocab_constants import *
from common.env_utils.visibility_utils import mesh_visibility_fraction, compute_obj_dimensions
from common.utils.grid_utils import HabitatObjOccupancyGrid

log = logging.getLogger(__name__)


def get_obj_from_id(sim: habitat_sim.Simulator,obj_id: int,):
    rom = sim.get_rigid_object_manager()
    if rom.get_library_has_id(obj_id):
        return rom.get_object_by_id(obj_id)
    return None

def object_shortname_from_handle(object_handle: str) -> str:
    """removes :_xxxx suffix from object that helps distinguish between multiple instances of the same object in the scene"""
    return object_handle.split("_:")[0]

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
        center_world = obj.rotation.transform_vector(aabb.center()) + obj.translation # type: ignore

        if obj_name not in obj_name_to_class:
            class_name = "/u:" + fallback_obj_name_to_class.get(obj_name, "undefined")
        else:
            class_name = obj_name_to_class[obj_name]

        out.append({
            "object_id": obj_id,
            "obj_name": obj_name,
            "class_name": class_name,
            "position": obj.translation, # type: ignore
            "rotation": obj.rotation, # type: ignore
            "center": np.array(center_world),
            "corners": [
                (c.x, c.y, c.z)
                for c in corners_world
            ],
        })
    return out


@registry.register_sensor
class ObjectDetectorGTSensor(habitat.Sensor): # type: ignore
    scene: str
    env_name: str
    camera_hfov: float
    object_info_list: list[dict]
    object_occupancy_grid: HabitatObjOccupancyGrid
    area_thr: float
    filter_low_visibility: bool
    min_visibility_fraction: float
    filter_out_classes: set[str]
    dimensions_by_obj_id: dict[int, np.ndarray]

    def __init__(self, sim, config: ObjectDetectorGTSensorConfig, **kwargs: Any) -> None:
        super().__init__(config=config)
        self._sim = sim
        self.scene = ""
        self.object_info_list = None # type: ignore
        self.object_occupancy_grid = None # type: ignore
        self.dimensions_by_obj_id = {}

        self.env_name = config.env_name
        self.area_thr = config.area_thr
        self.filter_low_visibility = config.filter_low_visibility
        self.min_visibility_fraction = config.min_visibility_fraction
        self.filter_out_classes = set(config.filter_out_classes)
        self.camera_hfov = next(
            float(spec.hfov)
            for spec in self._sim.config.agents[0].sensor_specifications
            if spec.uuid == "rgb"
        )

        if self.env_name.startswith("HSSD-HAB"):
            self.object_annotations = ObjectSemanticsHSSD(self.env_name.replace("HSSD-HAB/", ""))
        else:
            raise NotImplementedError(f"Environment {self.env_name} not supported for object detector gt sensor")

    def get_classes(self) -> list[str]:
        if self.env_name.startswith("HSSD-HAB"):
            return self.object_annotations.classes
        else:
            raise NotImplementedError(f"Environment {self.env_name} not supported for object detector gt sensor")

    def semantic_id_to_classid_obj_id(self, semantic_id: int) -> tuple[int, int]:
        if self.env_name.startswith("HSSD-HAB"):
            class_id = semantic_id % 1000
            obj_id = semantic_id // 1000
            return class_id, obj_id
        else:
            raise NotImplementedError(f"Environment {self.env_name} not supported for object detector gt sensor")

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "bbsgt"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return habitat.SensorTypes.MEASUREMENT # type: ignore

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(3,),
            dtype=np.float32,
        )

    def get_observation(self, *args: Any, **kwargs: Any):
        semantic_obs = kwargs['observations']['semantic']
        depth_obs = kwargs['observations']['depth']
        assert self.scene == self._sim.curr_scene_name, "Scene has changed since last setup of semantic labels. This should not happen, make sure to call setup_semantic_labels on the environment wrapper when changing scenes."
        return self.decompose_frame(semantic_obs, self._sim.get_agent_state(), depth_obs=depth_obs)

    def decompose_frame(
        self, semantic_obs: np.ndarray, agent_state: AgentState, depth_obs: np.ndarray
    ) -> dict:
        detections = []
        classes = self.get_classes()
        values = [v for v in set(semantic_obs.ravel().tolist()) if v >= 1000]

        for semantic_id in values:
            if semantic_id < 1000:
                assert semantic_id == 0, f"Unexpected semantic id {semantic_id} in semantic observation"
                continue

            mask = (semantic_obs == semantic_id).astype("uint8")
            class_id, obj_id = self.semantic_id_to_classid_obj_id(semantic_id)

            assert class_id < len(classes), f"Class id {class_id} for semantic id {semantic_id} is out of range for vocabulary {self.get_classes()}"

            if classes[class_id] == "unknown" or classes[class_id] in self.filter_out_classes:
                continue

            x, y, w, h = cv2.boundingRect(mask)
            mask_area = np.sum(mask)
            bbx_area = w * h

            filtered_low_area = False
            filtered_low_visibility = False
            visibility_fraction = 1.0

            if min(mask_area*2, bbx_area) < self.area_thr:
                filtered_low_area = True

            elif self.filter_low_visibility:
                visibility_fraction = mesh_visibility_fraction(
                    self.dimensions_by_obj_id[obj_id],
                    mask.astype(bool),
                    agent_state,
                    depth_obs,
                )
                if visibility_fraction < self.min_visibility_fraction:
                    filtered_low_visibility = True

            detections.append({
                "class_id": torch.tensor(class_id).unsqueeze(0),
                "mask": torch.from_numpy(mask.squeeze()).bool().unsqueeze(0),
                "bounding_box": torch.tensor([x, y, x + w, y + h]).unsqueeze(0),
                "info": {
                    "object_id": obj_id,
                    "env_name": self.env_name,
                    "filtered_low_area": filtered_low_area,
                    "filtered_low_visibility": filtered_low_visibility,
                    "visibility_fraction": visibility_fraction
                }
            })

        if not detections:
            return {'instances': Instances(
                image_size=(semantic_obs.shape[0], semantic_obs.shape[1]),
                pred_boxes=Boxes(torch.zeros((0,4))),
                pred_classes=torch.zeros((0,)).long(),
                scores=torch.zeros((0,)),
                pred_masks=torch.zeros((0, semantic_obs.shape[0], semantic_obs.shape[1])).bool(),
                infos=np.zeros((0,), dtype=object),
            )}

        get_center = lambda bbx: ((bbx[2] + bbx[0]) / 2.0, (bbx[3] + bbx[1]) / 2.0)
        sorted_detections = sorted(detections, key=lambda d: get_center(d["bounding_box"][0])[0] + get_center(d["bounding_box"][0])[1], reverse=False)
        pred_boxes = torch.cat([d["bounding_box"] for d in sorted_detections])
        pred_classes = torch.cat([d["class_id"] for d in sorted_detections])
        pred_masks = torch.cat([d["mask"] for d in sorted_detections])
        infos = np.array([d["info"] for d in sorted_detections], dtype=object)

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

            self.obj_id_to_class_id = {}
            self.dimensions_by_obj_id = {}
            mesh_cache_by_handle: dict[str, Optional[tuple[np.ndarray, np.ndarray]]] = {}

            for obj in get_all_objects(self._sim):
                obj_name = object_shortname_from_handle(obj.handle)

                if obj_name not in self.object_annotations.target_vocab_object_annotations:
                    for node in obj.visual_scene_nodes:
                        node.semantic_id = 0
                    continue

                class_name = self.object_annotations.target_vocab_object_annotations[obj_name]
                class_id = class2int[class_name]

                for node in obj.visual_scene_nodes:
                    node.semantic_id = obj.object_id * 1000 + class_id

                self.obj_id_to_class_id[obj.object_id] = class_id
                self.dimensions_by_obj_id[obj.object_id] = compute_obj_dimensions(obj)
                
            self.object_info_list = get_objects_info(self._sim, self.object_annotations.target_vocab_object_annotations, self.object_annotations.hssd400_object_annotations)
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

            if obj_name not in self.object_annotations.target_vocab_object_annotations:
                continue

            obj = rom.get_object_by_handle(obj_handle)
            class_id, obj_id = self.semantic_id_to_classid_obj_id(obj.semantic_id)
            class_name = classes[class_id]
            mapping[obj.object_id] = class_name

        return {
            "objects": mapping,
        }

