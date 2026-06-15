import logging
import math
import os, glob

import cv2
import numpy as np
import torch
import tqdm
from albumentations.pytorch import ToTensorV2
from detectron2.data import detection_utils as du
from detectron2.structures.boxes import Boxes, BoxMode
from detectron2.structures.instances import Instances
from detectron2.structures.masks import BitMasks
from torch.utils.data import Dataset



from common.env_utils.sense import (
    AgentPoseSense,
    BBSense,
    DepthSense,
    RGBSense,
    SemanticSense,
    Sense,
    VisualSense,
    get_class_from_modality_code,
    _get_info_from_string, get_sense_info
)


def _mask_more_n(arr, n) -> np.ndarray:
    mask = np.ones(arr.shape, np.bool_)

    current = arr[0]
    count = 0
    for idx, item in enumerate(arr):
        if item == current:
            count += 1
        else:
            current = item
            count = 1
        mask[idx] = count <= n
    return mask


class SampleLoader:
    paths: dict
    episode_list: np.ndarray

    def __init__(self, dataset_path, samples_path=None) -> None:
        self._load_paths(dataset_path, samples_path)

    def __len__(self) -> int:
        return len(self.get_episode_and_steps_dense_list()[0])

    def _load_paths(self, load_path, samples_paths=None) -> None:
        if samples_paths is None:
            samples_paths = glob.glob(load_path + "/*.npy")

        samples_paths = sorted(samples_paths, key=lambda x: (int(_get_info_from_string(x, "episode")), int(_get_info_from_string(x, "step")), int(_get_info_from_string(x, "id")), _get_info_from_string(x, "modality")))
        
        episode_list = [int(_get_info_from_string(s, "episode")) for s in samples_paths]
        mod_list = [_get_info_from_string(s, "modality") for s in samples_paths]
        idx_list = [int(_get_info_from_string(s, "id")) for s in samples_paths]
        steps_list = [int(_get_info_from_string(s, "step")) for s in samples_paths]

        paths = {}
        for sample_path, episode_id, input_id, mod, step in zip(
            samples_paths, episode_list, idx_list, mod_list, steps_list
        ):
            if episode_id not in paths:
                paths[episode_id] = {}
            if input_id not in paths[episode_id]:
                paths[episode_id][input_id] = {}
            if mod not in paths[episode_id][input_id]:
                paths[episode_id][input_id][mod] = {}

            paths[episode_id][input_id][mod][step] = sample_path

        self.paths = paths
        self.episode_list = np.array(episode_list)
        self.steps_list = np.array(steps_list)

    @staticmethod
    def _load_data(path: str) -> Sense:
        sense_info = get_sense_info(path)
        return get_class_from_modality_code(sense_info.mod).load(path)

    def get_episode_length(self, episode_id) -> int:
        return len(self.paths[episode_id][0][RGBSense.CODE])

    def get_sample(self, episode_id, input_id, mod, step) -> Sense:
        data_path = self.paths[episode_id][input_id][mod][step]
        return SampleLoader._load_data(data_path)

    def get_sample_multimodality(self, episode_id, id_camera, modalities, step) -> dict[str, Sense]:
        results = {}
        for mod in modalities:
            data = self.get_sample(episode_id, id_camera, mod, step)
            results[mod] = data
        return results

    def get_episode_and_steps_dense_list(self, filter_episodes=None, *args, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """
        Get list of episodes and of steps
        """
        mask = _mask_more_n(self.steps_list, 1)

        if filter_episodes is not None:
            mask_episodes = np.array(
                [li in filter_episodes for li in self.episode_list]
            )
            mask *= mask_episodes
        return self.episode_list[mask], self.steps_list[mask]


# A logger for this file
log = logging.getLogger(__name__)


class HabitatDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        inputs=None,
        sampler=None,
        transform=None,
        modalities=None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        if modalities is None:
            modalities = ["rgb", "bbsgt"]
        if sampler is None:
            sampler = SampleLoader(dataset_path)
        self.sampler = sampler
        if inputs is None:
            (
                episode_list,
                steps_list,
            ) = self.sampler.get_episode_and_steps_dense_list(*args, **kwargs)
            inputs = np.array([x for x in zip(episode_list, steps_list)])

        self.inputs = inputs

        indices = np.arange(len(self.inputs))

        self.index = indices

        if transform is None:
            transform = ToTensorV2()
        self.transform = transform

        self.camera_id = 0
        self.modalities = modalities

    def __len__(self) -> int:
        return 5
        # return len(self.index)

    def _transform_batch(self, x, y) -> tuple[torch.Tensor, Instances]:
        if len(y.gt_boxes.tensor) == 0:
            x = self.transform.transforms[0](image=x)["image"] # type: ignore
            size = x.shape[1:]
            y = Instances(
                gt_boxes=Boxes(torch.Tensor()),
                image_size=size,
                gt_masks=BitMasks(torch.Tensor(size=[0, *size])),
                gt_classes=torch.Tensor(),
                infos=[],
            )
            return x, y

        transformed = self.transform(
            image=x,
            bboxes=y.gt_boxes.tensor.numpy(),
            class_labels=y.gt_classes,
            masks=[m for m in y.gt_masks.numpy().astype(np.uint8)],
            infos=y.infos,
        )

        transformed_image = transformed["image"]
        transformed_bboxes = transformed["bboxes"]
        transformed_class_labels = transformed["class_labels"]
        transformed_infos = transformed["infos"]
        transformed_mask = transformed["masks"]

        size = transformed_image.shape[1:]

        x = transformed_image
        y = Instances(
            image_size=size,
            gt_boxes=Boxes(transformed_bboxes),
            gt_classes=torch.stack(transformed_class_labels),
            gt_masks=BitMasks(
                torch.stack(transformed_mask)
            ),
            infos=transformed_infos,
        )
        return x, y

    def __getitem__(self, idx) -> dict:
        episode, step = self.inputs[self.index[idx]]

        data = self.sampler.get_sample_multimodality(
            episode, self.camera_id, self.modalities, step
        )
        
        rgb_image = data["rgb"].data # type: ignore

        assert rgb_image.shape[-1] == 3

        x = np.ascontiguousarray(rgb_image[:, :, ::-1])
        y = data["bbsgt"].get_bbs_as_gt() # type: ignore

        x, y = self._transform_batch(x, y)

        size = x.shape[1:]

        return {
            "episode": episode,
            "image": x,
            "rgb_image": rgb_image,
            "episode": episode,
            "image_id": idx,
            "instances": y,
            "width": size[1],
            "height": size[0],
        }

    def get_coco_item_dict(self, idx) -> dict:
        ind = self.index[idx].item()
        episode, step = self.inputs[ind]

        data = self.sampler.get_sample_multimodality(
            episode, self.camera_id, ["bbsgt"], step
        )

        gt = data["bbsgt"]
        file_name = gt.frame.sense_info.get_path()  # type: ignore
        y = data["bbsgt"].get_bbs_as_gt()  # type: ignore
        class_labels = y.gt_classes

        annotations = [
            {
                "bbox": y[id_instance].gt_boxes.tensor[0].tolist(),
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": class_labels[id_instance],
                # "segmentation": y[id_instance].gt_masks,
                "iscrowd": 0,
            }
            for id_instance in range(len(y))
        ]
        rgb_image = data["rgb"].data # type: ignore
        assert rgb_image.shape[-1] == 3

        instance_dict = {
            "file_name": file_name,
            "image_id": ind,
            "rgb_image": rgb_image,
            "height": y.image_size[0],
            "width": y.image_size[1],
            "annotations": annotations,
            "episode": episode,
        }

        return instance_dict


class HabitatFullDataset(HabitatDataset):
    def __init__(self, dataset_path, *args, **kwargs) -> None:
        super().__init__(
            dataset_path, modalities=["rgb", "depth", "position", "bbsgt"], *args, **kwargs
        )


    def __getitem__(self, idx) -> dict:
        episode, step = self.inputs[self.index[idx]]

        data = self.sampler.get_sample_multimodality(
            episode, self.camera_id, self.modalities, step
        )
        rgb_image = data["rgb"].data # type: ignore
        assert rgb_image.shape[-1] == 3


        x = np.ascontiguousarray(rgb_image[:, :, ::-1])

        depth = data["depth"].data # type: ignore
        location = data["position"].get_T() # type: ignore
        y = data["bbsgt"].get_bbs_as_gt() # type: ignore

        rgbd = np.concatenate((x, depth), -1)
        transformed_rgbd, y = self._transform_batch(rgbd, y)
        x = transformed_rgbd[:3]
        depth = transformed_rgbd[-1].unsqueeze(0)
        size = x.shape[1:]

        return {
            "image": x,
            "rgb_image": rgb_image,
            "depth": depth,
            "location": torch.tensor(location),
            "instances": y,
            "width": size[1],
            "height": size[0],
            "info": f"episode_{episode}_step_{step}",
            "episode": episode,
        }


class HabitatSequentialDataset(HabitatDataset):
    def __init__(
        self, dataset_path, consecutive_obs=4, subsample_factor=2, *args, **kwargs
    ) -> None:
        sampler = SampleLoader(dataset_path)
        (
            self.episode_list,
            self.steps_list,
        ) = sampler.get_episode_and_steps_dense_list()

        num_sample = len(self.steps_list) // consecutive_obs * subsample_factor
        episode_list = np.resize(self.episode_list, (num_sample, consecutive_obs))
        steps_list = np.resize(self.steps_list, (num_sample, consecutive_obs))

        inputs = np.array([x for x in zip(episode_list, steps_list)])

        self.window_size = consecutive_obs
        modalities = ["rgb", "bbsgt"]

        super().__init__(
            dataset_path=dataset_path,
            sampler=sampler,
            inputs=inputs,
            modalities=modalities,
            *args,
            **kwargs,
        )
        self.subsample_factor = subsample_factor

    def __getitem__(self, idx) -> list[dict]:
        result = []
        episodes, steps = self.inputs[self.index[idx]]
        length_mask = len(episodes)
        sub_randomized_mask = np.random.choice(
            range(length_mask), length_mask // self.subsample_factor
        )
        for index in sub_randomized_mask:
            episode = episodes[index]
            step = steps[index]

            data = self.sampler.get_sample_multimodality(
                episode, self.camera_id, self.modalities, step
            )

            rgb_image = data["rgb"].data # type: ignore
            assert rgb_image.shape[-1] == 3

    

            x = np.ascontiguousarray(rgb_image[:, :, ::-1])
            y = data["bbsgt"].get_bbs_as_gt() # type: ignore

            x, y = self._transform_batch(x, y)

            size = x.shape[1:]
            result.append(
                {
                    "image": x,
                    "rgb_image": rgb_image,
                    "instances": y,
                    "width": size[1],
                    "height": size[0],
                    "episode": episode,
                    "info": f"episode_{episode}_step_{step}",
                }
            )
        return result


class HabitatFullSequentialDataset(HabitatDataset):
    def __init__(
        self,
        dataset_path,
        sampler=None,
        consecutive_obs=1,
        subsample_factor=1,
        index_mask=None,
        *args,
        **kwargs,
    ) -> None:
        if sampler is None:
            sampler = SampleLoader(dataset_path)
        (
            self.episode_list,
            self.steps_list,
        ) = sampler.get_episode_and_steps_dense_list(*args, **kwargs)

        if index_mask:
            self.episode_list = self.episode_list[index_mask]
            self.steps_list = self.steps_list[index_mask]
        num_sample = len(self.steps_list) // consecutive_obs * subsample_factor
        episode_list = np.resize(self.episode_list, (num_sample, consecutive_obs))
        steps_list = np.resize(self.steps_list, (num_sample, consecutive_obs))

        inputs = np.array([x for x in zip(episode_list, steps_list)])

        modalities = ["rgb", "bbsgt"]

        super().__init__(
            dataset_path=dataset_path,
            sampler=sampler,
            inputs=inputs,
            modalities=modalities,
            *args,
            **kwargs,
        )
        self.window_size = consecutive_obs

        self.subsample_factor = subsample_factor

    def __getitem__(self, idx) -> list[dict]:

        result = []
        episodes, steps = self.inputs[self.index[idx]]
        length_mask = len(episodes)
        sub_randomized_mask = np.random.choice(
            range(length_mask), length_mask // self.subsample_factor, replace=False
        )
        for index in sub_randomized_mask:

            episode = episodes[index]
            step = steps[index]

            data = self.sampler.get_sample_multimodality(
                episode, self.camera_id, self.modalities, step
            )

            rgb_image = data["rgb"].data # type: ignore
            assert rgb_image.shape[-1] == 3

    

            x = np.ascontiguousarray(rgb_image[:, :, ::-1])
            y = data["bbsgt"].get_bbs_as_gt()  # type: ignore

            x, y = self._transform_batch(x, y)

            size = x.shape[1:]

            result.append(
                {
                    "image_id": int(idx * self.window_size + index),
                    "rgb_image": rgb_image,
                    "image": x,
                    "instances": y,
                    "width": size[1],
                    "height": size[0],
                    "episode": torch.tensor(episode),
                    "step": torch.tensor(step),
                }
            )
        return result

    def get_coco_item_dict(self, index) -> dict:

        idx = index // self.window_size
        sub_id = index % self.window_size

        episodes, steps = self.inputs[self.index[idx]]
        episode = episodes[sub_id]
        step = steps[sub_id]

        data = self.sampler.get_sample_multimodality(
            episode, self.camera_id, ["bbsgt"], step
        )

        gt = data["bbsgt"]
        file_name = gt.frame.sense_info.get_path()  # type: ignore
        y = data["bbsgt"].get_bbs_as_gt()  # type: ignore
        class_labels = y.gt_classes

        annotations = [
            {
                "bbox": y[id_instance].gt_boxes.tensor[0].tolist(),
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": class_labels[id_instance],
                # "segmentation": y[id_instance].gt_masks,
                "iscrowd": 0,
            }
            for id_instance in range(len(y))
        ]

        instance_dict = {
            "file_name": file_name,
            "image_id": index,
            "height": y.image_size[0],
            "width": y.image_size[1],
            "annotations": annotations,
        }

        return instance_dict


def _transform_batch_with_logits(transform, x, y) -> tuple[torch.Tensor, Instances]:
    if isinstance(transform, ToTensorV2):
        out = transform(image=x)
        x = out["image"]

    else:
        transformed = transform(
            image=x,
            bboxes=y.gt_boxes.tensor.numpy(),
            class_labels=y.gt_classes,
            masks=[m for m in y.gt_masks.numpy().astype(np.uint8)],
            infos=y.infos,
            gt_logits=[l for l in y.gt_logits],
        )

        transformed_image = transformed["image"]
        transformed_bboxes = transformed["bboxes"]

        transformed_class_labels = transformed["class_labels"]
        transformed_infos = transformed["infos"]
        transformed_mask = transformed["masks"]

        transformed_gt_logits = transformed["gt_logits"]
        size = transformed_image.shape[1:]

        x = transformed_image

        min_area = -1  # transform._to_dict()["bbox_params"]["min_area"]

        if len(transformed_class_labels) > 0:
            semantic_masks_boxes = [cv2.boundingRect(x.numpy() if not isinstance(x, np.ndarray) else x) for x in transformed_mask]
            semantic_masks_area = [x[-1] * x[-2] for x in semantic_masks_boxes]
            mask =  [m for m, area in zip(transformed_mask, semantic_masks_area) if area > min_area]
            gt_masks = BitMasks(torch.stack(mask))

            y = Instances(
                image_size=size,
                gt_boxes=Boxes(transformed_bboxes),
                gt_classes=torch.stack(transformed_class_labels),
                gt_logits=torch.stack(transformed_gt_logits),
                gt_masks=gt_masks,
                infos=transformed_infos,
            )
        else:
            y = Instances(
                gt_boxes=Boxes(torch.Tensor()),
                image_size=size,
                gt_masks=BitMasks(torch.Tensor(size=[0, *size])),
                gt_classes=torch.Tensor(),
                gt_logits=torch.Tensor(),
                infos=[],
            )

    return x, y
