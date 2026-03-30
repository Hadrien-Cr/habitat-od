from pathlib import Path
import os
import yaml
from common.utils.data_utils import fname2pose, pose2fname, enumerate_fnames, make_colors, save_img, save_label, load_img, load_label, DiscretizedAgentPose
import numpy as np
from datasets import DatasetDict, Dataset
from tqdm import tqdm
from torchvision.io import decode_image

from dataclasses import dataclass

@dataclass
class DetectionConfig:
    min_pixel_area: int
    class_names: list[str]


class ObjectDetectionDataset(Dataset):
    def __init__(self, data_dir: Path, transform=None, target_transform=None):
        self.data_dir = data_dir
        self.fnames = enumerate_fnames(data_dir)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = data_dir / self.names[idx]
        image = load_img(fname)
        label = load_label(fname)
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label

def save_data(data: list[tuple[Path, np.ndarray, list[dict]]], data_dir: Path):
    """
    data: list of (fname, image, labels) to store
    """
    for (fname, img, label) in tqdm(data, desc = "Saving data"):
        save_img(img, data_dir, fname)
        save_label(label, data_dir, fname, img.shape)


def save_dataset(
    data_root: Path, dataset_name: Path, class_names, splits: dict[str, list[tuple[Path, np.ndarray, list[dict]]]]
) -> None:

    for split_name, data in splits.items():
        save_data(data, data_root / dataset_name / split_name)

    content = dict(
        path=data_root / dataset_name,
        train="train",  # training images relative to 'path'
        val="val",  # validation images relative to path
        nc=len(class_names),  # number of class_names
        names=class_names,
        colors=make_colors(len(class_names), seed=1),
    )
    
    with open(os.path.join(data_root, dataset_name, f"{dataset_name}.yaml"), "w") as f:
        yaml.dump(content, f)


def load_dataset(
    data_dir: Path,
) -> DatasetDict:
    dataset_yaml_path = None

    for file in data_dir.parent.iterdir():
        if file.suffix == ".yaml":
            dataset_yaml_path = file
            break

    if dataset_yaml_path is None:
        raise ValueError("No YAML configuration file found in the dataset directory.")

    with open(dataset_yaml_path) as f:
        config = yaml.safe_load(f)

    class_names = list(config["names"])
    colors = list(config["colors"])
    splits = list(config["splits"])
    
    return DatasetDict({
        split_name: ObjectDetectionDataset(data_dir/split_name) for split_name in splits
    })


