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
class ObjectDetectionDatasetGenerationConfig:
    env: str
    scenes: list[str]
    num_samples: int
    seed: int

    min_pixel_area: int
    class_names: list[str]
    img_shape: tuple[int,int]
    grid_size: float
    visibility_distance: float
    yaw_bins: int

    downsampling: str
    downsampling_factor: float


class ObjectDetectionDataset(Dataset):
    def __init__(self, data_dir: Path, class_names: list[str], transform=None, target_transform=None):
        self.data_dir = data_dir
        self.fnames = enumerate_fnames(data_dir)
        self.class_names = class_names
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        image = load_img(self.data_dir / "images" / fname.with_suffix(".jpg"))
        label = load_label(self.data_dir / "labels" / fname.with_suffix(".txt"), self.class_names, image.shape[:2])

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label

def save_data(data: list[tuple[Path, np.ndarray, list[dict]]], class_names: list[str], data_dir: Path):
    """
    data: list of (fname, image, labels) to store
    """
    for (fname, img, label) in tqdm(data, desc = "Saving data"):
        save_img(img, data_dir, fname)
        save_label(label, data_dir, fname, img.shape, class_names)


def save_dataset(
    data_root: Path, dataset_name: Path, class_names: list[str], splits: dict[str, tuple[ObjectDetectionDatasetGenerationConfig, list]]
) -> None:
    
    os.makedirs(data_root / dataset_name, exist_ok=True)

    content = dict(
        path=str(data_root / dataset_name),
        num_classes=len(class_names),
        class_names=class_names,
        splits=list(splits.keys())
    )

    with open(data_root / dataset_name / f"{dataset_name}.yaml", "w") as f:
        yaml.dump(content, f)

    for split_name, (config,data) in splits.items():
        save_data(data, class_names, data_root / dataset_name / split_name)

        content = dict(
            path=str(data_root / dataset_name / split_name),
            scenes=config.scenes,
            num_samples=config.num_samples,
            seed=config.seed,
            min_pixel_area=config.min_pixel_area,
            img_shape=config.img_shape,
            grid_size=config.grid_size,
            visibility_distance=config.visibility_distance,
            yaw_bins=config.yaw_bins,
            downsampling=config.downsampling,
            downsampling_factor=config.downsampling_factor,
        )
        
        with open(data_root / dataset_name / split_name /f"{dataset_name}-{split_name}.yaml", "w") as f:
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
        main_config = yaml.safe_load(f)
    
    datasets_splits = {}

    for split_name in list(main_config["splits"]):
        for file in (data_dir/split_name).parent.iterdir():
            if file.suffix == ".yaml":
                dataset_yaml_path = file
                break

        with open(dataset_yaml_path) as f:
            dataset_split_config = yaml.safe_load(f)

        datasets_splits[split_name] = ObjectDetectionDataset(
            data_dir=data_dir/split_name,
            class_names=main_config["class_names"]
        )

    return DatasetDict(datasets_splits)


