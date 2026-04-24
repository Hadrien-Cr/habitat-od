from pathlib import Path
import os
import yaml
from habitat_od.utils.data_utils import agent_state2fname, enumerate_fnames, make_colors, save_img, save_ground_truth, load_img, load_bounding_boxes, load_segmentation_masks
import numpy as np
from torch.utils.data import Dataset
from datasets import DatasetDict
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from habitat_sim.agent.agent import AgentState


class ObjectDetectionDataset(Dataset):
    def __init__(self, data_dir: Path, class_mapping: dict[str, int], transform=None, target_transform=None):
        self.data_dir = data_dir
        self.fnames = enumerate_fnames(data_dir)
        self.class_mapping = class_mapping
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        image = load_img(self.data_dir / "images" / fname.with_suffix(".jpg"))
        label = load_bounding_boxes(self.data_dir / "labels" / fname.with_suffix(".txt"), image.shape[:2])

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label


class SegmentationDataset(Dataset):
    def __init__(self, data_dir: Path, class_mapping: dict[str, int], transform=None, target_transform=None):
        self.data_dir = data_dir
        self.fnames = enumerate_fnames(data_dir)
        self.class_mapping = class_mapping
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        image = load_img(self.data_dir / "images" / fname.with_suffix(".jpg"))
        label = load_segmentation_masks(self.data_dir / "labels" / fname.with_suffix(".txt"), image.shape[:2])

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label


def save_data(list_fname_images_masks: list[tuple[Path, np.ndarray, list]], class_mapping: dict[str, int], data_dir: Path, segmentation: bool):
    """
    data: list of (fname, image, labels) to store
    """
    for (fname, img, object_detection_info) in tqdm(list_fname_images_masks, desc = "Saving data"):
        save_img(img, data_dir, fname)
        if segmentation:
            save_ground_truth(object_detection_info, data_dir, fname, img.shape, class_mapping, save_bounding_boxes=False, save_segmentations_masks=True)
        else:
            save_ground_truth(object_detection_info, data_dir, fname, img.shape, class_mapping, save_bounding_boxes=True, save_segmentations_masks=False)


def save_dataset(config, splits: dict[str, list], class_mapping: dict[str, int]) -> None:
    os.makedirs(Path(config.DATASET.data_root) / config.DATASET.dataset_name, exist_ok=True)

    ds_path = Path(config.DATASET.data_root) / config.DATASET.dataset_name
    content = dict(
        path= ds_path,
        names={i: name for i, name in enumerate(class_mapping.keys())},
    )

    with open( ds_path / f"{config.DATASET.dataset_name}.yaml", "w") as f:
        yaml.dump(content, f)

    for split_name, list_fname_images_masks in splits.items():
        save_data(
            list_fname_images_masks, 
            class_mapping, 
            ds_path / split_name, 
            config.DATASET.segmentation
        )

        content = {
            "name": split_name,
            "path": ds_path / split_name
        }

        with open(ds_path / split_name / f"{config.DATASET.dataset_name}-{split_name}.yaml", "w") as f:
            yaml.dump(content, f)


def load_dataset(
    data_dir: Path,
) -> tuple[DatasetDict, dict[int, str], dict[int, tuple[int, int, int]]]:
    
    dataset_yaml_path = None
    for file in data_dir.iterdir():
        if file.suffix == ".yaml":
            dataset_yaml_path = file
            break
    if dataset_yaml_path is None:
        raise ValueError("No YAML configuration file found in the dataset directory.")

    with open(dataset_yaml_path) as f:
        main_config = yaml.safe_load(f)
    
    datasets_splits = {}

    for split_name in list(main_config["splits"]):

        dataset_yaml_path = None
        for file in (data_dir/split_name).iterdir():
            if file.suffix == ".yaml":
                dataset_yaml_path = file
                break
        if dataset_yaml_path is None:
            raise ValueError("No YAML configuration file found in the dataset directory.")

        with open(dataset_yaml_path) as f:
            dataset_split_config = yaml.safe_load(f)

        if dataset_split_config["segmentation"]:
            datasets_splits[split_name] = SegmentationDataset(
                data_dir=data_dir/split_name,
                class_mapping={name: i for i, name in enumerate(main_config["names"])}
            )
        else:
            datasets_splits[split_name] = ObjectDetectionDataset(
                data_dir=data_dir/split_name,
                class_mapping={name: i for i, name in enumerate(main_config["names"])}
            )

    return DatasetDict(datasets_splits), main_config["names"], make_colors(len(main_config["names"]))


