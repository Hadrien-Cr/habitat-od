import abc
import dataclasses
from dataclasses import dataclass
import os
from pathlib import WindowsPath

import cv2
import numpy as np
import quaternion
import torch
from detectron2.data import MetadataCatalog
from detectron2.structures import BitMasks
from detectron2.structures.instances import Instances
from detectron2.utils.visualizer import ColorMode, Visualizer
from habitat.utils.visualizations import maps # type: ignore[import]
from typing import Any


def _get_info_from_string(path, info, split_symbol="_"):
    filename = os.path.split(os.path.splitext(path)[0])[1]
    return filename[filename.find(info) :].split(split_symbol)[1]

def get_sense_info(path):
    base_path = os.path.dirname(path)
    episode = int(_get_info_from_string(path, "episode"))
    mod = _get_info_from_string(path, "modality")
    idx = int(_get_info_from_string(path, "id"))
    step = int(_get_info_from_string(path, "step"))
    return SenseInfo(base_path, mod, episode, idx, step)




def get_class_from_modality_code(code: str):
    switch = {
        "rgb": RGBSense,
        "depth": DepthSense,
        "semantic": SemanticSense,
        "semanticinstances": SemanticInstancesSense,
        "bbs": BBSense,
        "bbsgt": BBSense,
        'position': AgentPoseSense,
        'egomap': EgomapSense,
    }
    return switch[code]

@dataclass
class SenseInfo:
    """Class for keeping track of an item in inventory."""
    base_path: str
    mod: str
    episode: int = 0
    camera_id: int = 0
    step: int = 0

    def get_path(self) -> str:
        return os.path.join(
            self.base_path,
            f"episode_{self.episode:06d}_modality_{self.mod}_step_{self.step:05d}_id_{self.camera_id}.npy",
        )
    
class Sense(abc.ABC):
    def __init__(self, path: str, sense_info: SenseInfo):
        if sense_info is None and path is not None:
            self.sense_info = get_sense_info(path)
        elif sense_info is not None:
            self.sense_info = sense_info
        else:
            raise ValueError("Either path or sense_info must be provided")

        if self.sense_info is not None:
            self.name = f"{self.sense_info.episode}-{self.sense_info.mod}-{self.sense_info.camera_id}"
        else:
            self.name = ""

    @staticmethod
    def load(path):
        return Sense(path, sense_info=get_sense_info(path))



class Pose(Sense):
    AGENT_TO_SENSOR_TRANSLATION = np.array([0, 0.88, 0])

    def __init__(
        self,
        position: np.ndarray,
        orientation,
        reference: str,
        path: str,
        sense_info: SenseInfo,
    ):
        super().__init__(path, sense_info)
        self.position = position
        self.orientation = orientation
        self.reference = reference

    def get_T(self):
        """
        Get pose_world transformation matrix for pose
        """
        rotation_0 = quaternion.as_rotation_matrix(self.orientation)
        T = np.eye(4)
        T[0:3, 0:3] = rotation_0
        T[0:3, 3] = self.position
        return T

    def get_transformation_to_pose(self, pose2):
        T_world_pose1 = self.get_T()
        T_world_pose2 = pose2.get_T()

        T_pose2_world = np.linalg.inv(T_world_pose2)

        T_pose2_pose1 = np.matmul(T_pose2_world, T_world_pose1)
        return T_pose2_pose1


class AgentPoseSense(Pose):

    CODE = "position"

    def __init__(
        self, position: np.ndarray, orientation: quaternion.quaternion, path: str, sense_info: SenseInfo
    ):
        super().__init__(
            position, orientation, "agent", path=path, sense_info=sense_info
        )

    def get_T_world_agent(self):
        """
        Get pose_world transformation matrix for pose
        """
        rotation_0 = quaternion.as_rotation_matrix(self.orientation)
        T = np.eye(4)
        T[0:3, 0:3] = rotation_0
        T[0:3, 3] = self.position
        return T

    def get_cam_pose(self):
        rot_mat = quaternion.as_rotation_matrix(self.orientation)
        translation = np.matmul(rot_mat, AgentPoseSense.AGENT_TO_SENSOR_TRANSLATION)
        position = self.position + translation
        return CamPoseSense(
            position=position, orientation=self.orientation, sense_info=self.sense_info, path=None
        )

    @staticmethod
    def load(path):
        location_data = np.load(path, allow_pickle=True)

        try:
            position = location_data.item()['position']
            orientation = location_data.item()['orientation']

        except Exception as ex:  # type: ignore[F841]
            position = location_data[0]
            orientation = location_data[1]

        return AgentPoseSense(position, orientation, path, sense_info=get_sense_info(path)).get_cam_pose()


class CamPoseSense(Pose):
    def __init__(
        self, position: np.ndarray, orientation: quaternion.quaternion, path: str, sense_info: SenseInfo
    ):
        super().__init__(position, orientation, "cam", path=path, sense_info=sense_info)


@dataclass
class Intrinsics:
    xc: float
    yc: float
    focal_length: float
    width: int
    height: int

    def get_mat(self) -> np.ndarray:
        return np.array(
            [
                [self.focal_length, 0, self.xc],
                [0.0, self.focal_length, self.yc],
                [0.0, 0, 1],
            ]
        )


class VisualSense(Sense):
    HFOV_DEG = 90

    def get_camera_matrix(self, fov=HFOV_DEG):
        """
        From Object-Goal-Navigation
        Returns a camera matrix from image size and fov.
        """
        width = height = self.get_width()
        xc = (width - 1.0) / 2.0
        yc = (height - 1.0) / 2.0
        f = (width / 2.0) / np.tan(np.deg2rad(fov) / 2.0)

        return Intrinsics(xc, yc, f, width, height)

    def __init__(self, data: Any, path: str, sense_info: SenseInfo):
        super().__init__(path, sense_info)

        self.data = data

    def get_width(self):
        return self.data.shape[0]


class DepthSense(VisualSense):
    CODE = "depth"

    def __init__(self, data: np.ndarray, path: str, sense_info: SenseInfo):
        super().__init__(data, path, sense_info)

    @staticmethod
    def load(path):
        depth_image = np.load(path)

        if "neuralslam" in path:
            depth_image = depth_image * 10  # only for neuralslam

        return DepthSense(depth_image, path, sense_info=get_sense_info(path))


class RGBSense(VisualSense):
    CODE = "rgb"
    INPUT_FORM = "RGB"  # RGB

    def __init__(self, data: np.ndarray, path: str, sense_info: SenseInfo):
        super().__init__(data, path, sense_info)

    @staticmethod
    def load(path):
        rgb_image = np.load(path)
        rgb_image = rgb_image[:, :, 0:3]  # remove alpha channel
        return RGBSense(rgb_image, path, sense_info=get_sense_info(path))


class SemanticSense(VisualSense):
    CODE = "semantic"

    def __init__(self, data: np.ndarray, path: str, sense_info: SenseInfo):
        super().__init__(data, path, sense_info)

    @staticmethod
    def load(path):
        semantic_image = np.load(path).astype("uint8")
        return SemanticSense(semantic_image, path, sense_info=get_sense_info(path))



class SemanticInstancesSense(VisualSense):
    CODE = "semantic"

    def __init__(self, data: np.ndarray, mapping: dict, path: str, sense_info: SenseInfo):
        super().__init__(data, path, sense_info)
        self.mapping = mapping

    @staticmethod
    def load(path: str):
        data = np.load(path, allow_pickle=True).item()

        semantic_image = data['semantic_instances']
        mapping = data['mapping']
        return SemanticInstancesSense(semantic_image, mapping, path, sense_info=get_sense_info(path))



class EgomapSense(VisualSense):
    CODE = "egomap"

    def __init__(self, data: np.ndarray, path: str, sense_info: SenseInfo):
        super().__init__(data, path, sense_info)

    @staticmethod
    def load(path: str):
        egomap = np.load(path)
        return EgomapSense(egomap, path, sense_info=get_sense_info(path))


class BBSense(VisualSense):
    CODE = "bbs"

    def __init__(self, bbs: Instances, frame: RGBSense, path: str, sense_info: SenseInfo):
        super().__init__(bbs, path, sense_info)
        self.bbs = bbs
        rgb_sense_info = dataclasses.replace(self.sense_info, mod=RGBSense.CODE)

        try:
            if frame is None and rgb_sense_info is not None:
                frame = RGBSense.load(rgb_sense_info.get_path())
            self.frame = frame
        except Exception as ex:
            self.frame = None

    @staticmethod
    def load(path_bb):
        res = np.load(path_bb, allow_pickle=True).item()
        return BBSense(path=path_bb, bbs=res["instances"], frame=None, sense_info=get_sense_info(path_bb))

    def get_bbs_as_gt(self):
        target = Instances(self.bbs.image_size)
        target.gt_boxes = self.bbs.pred_boxes
        target.gt_classes = self.bbs.pred_classes

        if hasattr(self.bbs, "pred_masks"):
            target.gt_masks = self.bbs.pred_masks

        if hasattr(self.bbs, "infos"):
            target.infos = self.bbs.infos
            for t in target.infos:
                t['episode'] = self.sense_info.episode

        return target

    def get_bounding_boxes(self):
        if 'pred_boxes' in self.bbs:
            return self.bbs.pred_boxes
        else:
            return []
