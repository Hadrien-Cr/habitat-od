from pathlib import Path
import os
import yaml
from common.utils.data_utils import fname2pose, pose2fname, enumerate_fnames, make_colors, save_img, save_ground_truth, load_img, load_bounding_boxes, load_segmentation_masks
import numpy as np
from torch.utils.data import Dataset
from datasets import DatasetDict
from tqdm import tqdm
from torchvision.io import decode_image

from dataclasses import dataclass

@dataclass
class DatasetGenerationConfig:
    """Configuration for generating the dataset."""
    env: str
    """The environment for the dataset."""
    scenes: list[str]
    """The scenes used for generation."""
    num_samples: int
    """The number of samples to generate."""
    seed: int
    """The random seed for reproducibility."""
    segmentation: bool
    """Whether to include segmentation masks."""

    min_pixel_area: int
    """The minimum area (in pixels) of objects to be included in the dataset."""
    class_mapping: dict[str, int]
    """A mapping from class names (AI2THOR object types) to class IDs (integers). Objects not in this mapping will be ignored."""
    img_height: int
    img_width: int
    grid_size: float
    """The size of the grid cells."""
    visibility_distance: float
    """The maximum distance at which objects are visible."""
    yaw_bins: int
    """The number of yaw bins for discretizing the agent's orientation."""

    downsampling: str
    """The downsampling method to use."""
    downsampling_factor: float
    """The factor by which to downsample the dataset. If det to 4, 1 on 4 images will be saved."""


    def to_dict(self):
        return {
            "env": self.env,
            "scenes": self.scenes,
            "num_samples": self.num_samples,
            "seed": self.seed,
            "segmentation": self.segmentation,
            "min_pixel_area": self.min_pixel_area,
            "img_height": self.img_height,
            "img_width": self.img_width,
            "grid_size": self.grid_size,
            "visibility_distance": self.visibility_distance,
            "yaw_bins": self.yaw_bins,
            "downsampling": self.downsampling,
            "downsampling_factor": self.downsampling_factor
        }
    

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


def save_data(data: list[tuple[Path, np.ndarray, dict]], class_mapping: dict[str, int], data_dir: Path, segmentation: bool):
    """
    data: list of (fname, image, labels) to store
    """
    for (fname, img, label) in tqdm(data, desc = "Saving data"):
        save_img(img, data_dir, fname)
        
        if segmentation:
            save_ground_truth(label, data_dir, fname, img.shape, class_mapping, save_bounding_boxes=False, save_segmentations_masks=True)
        else:
            save_ground_truth(label, data_dir, fname, img.shape, class_mapping, save_bounding_boxes=True, save_segmentations_masks=False)


def save_dataset(
    data_root: Path, dataset_name: Path, class_mapping: dict[str, int], splits: dict[str, tuple[DatasetGenerationConfig, list]]
) -> None:
    
    os.makedirs(data_root / dataset_name, exist_ok=True)

    content = dict(
        path=str(data_root / dataset_name),
        names={i: name for i, name in enumerate(class_mapping.keys())},
        splits=list(splits.keys())
    )

    with open(data_root / dataset_name / f"{dataset_name}.yaml", "w") as f:
        yaml.dump(content, f)

    for split_name, (config,data) in splits.items():
        save_data(data, class_mapping, data_root / dataset_name / split_name, config.segmentation)

        content = config.to_dict()
        content["path"] = str(data_root / dataset_name / split_name)

        with open(data_root / dataset_name / split_name /f"{dataset_name}-{split_name}.yaml", "w") as f:
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


