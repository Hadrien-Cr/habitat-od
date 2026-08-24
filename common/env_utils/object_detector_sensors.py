import logging
from typing import Any

import cv2
from habitat_sim.agent.agent import AgentState
import torch
import habitat # type: ignore
import numpy as np
from gym import spaces
from habitat.core.registry import registry # type: ignore[import]
from habitat.config.default_structured_configs import ObjectDetectorGTSensorConfig  # type: ignore[import]
from detectron2.structures import Boxes, Instances

from common.env_utils import object_annotations
from common.env_utils.object_annotations import ObjectAnnotation
from common.env_utils.visibility_utils import mesh_visibility_fraction

log = logging.getLogger(__name__)


@registry.register_sensor
class ObjectDetectorGTSensor(habitat.Sensor): # type: ignore
    scene: str
    env_name: str
    vocab_name: str
    camera_hfov: float
    area_thr: float
    filter_low_visibility: bool
    min_visibility_fraction: float
    filter_out_classes: set[str]
    annotation: ObjectAnnotation

    def __init__(self, sim, config: ObjectDetectorGTSensorConfig, **kwargs: Any) -> None:
        super().__init__(config=config)
        self._sim = sim
        self.scene = ""
        self.annotation = None # type: ignore - populated by setup_semantic_labels()

        self.env_name = config.env_name
        self.vocab_name = config.vocab_name
        self.area_thr = config.area_thr
        self.filter_low_visibility = config.filter_low_visibility
        self.min_visibility_fraction = config.min_visibility_fraction
        self.filter_out_classes = set(config.filter_out_classes)
        self.camera_hfov = next(
            float(spec.hfov)
            for spec in self._sim.config.agents[0].sensor_specifications
            if spec.uuid == "rgb"
        )

    def get_classes(self) -> list[str]:
        return self.annotation.classes

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
        classes = self.annotation.classes
        values = [v for v in set(semantic_obs.ravel().tolist()) if v != 0]

        for semantic_id in values:
            decoded = self.annotation.semantic_id_to_classid_obj_id(semantic_id)
            if decoded is None:
                continue
            class_id, obj_id = decoded

            mask = (semantic_obs == semantic_id).astype("uint8")

            assert class_id < len(classes), f"Class id {class_id} for semantic id {semantic_id} is out of range for vocabulary {classes}"

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
                    self.annotation.dimensions_by_obj_id[obj_id],
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
        self.annotation = object_annotations.setup_semantic_labels(self._sim, self.env_name, self.vocab_name)
