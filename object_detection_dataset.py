from common.utils.object_utils import load_object_classes
from common.utils.scene_utils import enumerate_scenes
from thor_od.data_gen import DatasetGenerationConfig, random_teleport_collection, save_dataset, load_dataset

from pathlib import Path


if __name__ == "__main__":
    all_classes = load_object_classes("procthor-all")
    class_mapping = {c: i for i, c in enumerate(all_classes)}

    train_split_config = DatasetGenerationConfig(
        env="ProcTHOR",
        scenes=enumerate_scenes("procthor-train"),
        num_samples=10000,
        seed=0,
        segmentation=True,
        img_height=640,
        img_width=640,
        min_pixel_area=0,
        grid_size=0.25,
        visibility_distance=8.0,
        yaw_bins=12,
        class_mapping=class_mapping,
        downsampling="covisibility",
        downsampling_factor=4,
    )
    
    val_split_config = DatasetGenerationConfig(
        env="ProcTHOR",
        scenes=enumerate_scenes("procthor-val"),
        num_samples=2000,
        seed=0,
        segmentation=True,
        img_height=640,
        img_width=640,
        min_pixel_area=0,
        grid_size=0.25,
        visibility_distance=8.0,
        yaw_bins=12,
        class_mapping=class_mapping,
        downsampling="covisibility",
        downsampling_factor=4,
    )

    test_split_config = DatasetGenerationConfig(
        env="ProcTHOR",
        scenes=enumerate_scenes("procthor-test"),
        num_samples=2000,
        seed=0,
        segmentation=True,
        img_height=640,
        img_width=640,
        min_pixel_area=0,
        grid_size=0.25,
        visibility_distance=8.0,
        yaw_bins=12,
        class_mapping=class_mapping,
        downsampling="covisibility",
        downsampling_factor=4,
    )

    train_samples = random_teleport_collection(train_split_config)
    val_samples = random_teleport_collection(val_split_config)
    test_samples = random_teleport_collection(test_split_config)

    save_dataset(
        data_root=Path("datasets"), 
        dataset_name=Path("procthor-segmentation"), 
        class_mapping=class_mapping,
        splits={
            "train": (train_split_config,train_samples), 
            "val": (val_split_config, val_samples), 
            "test": (test_split_config, test_samples) 
        }
    )

    loaded_dataset = load_dataset(Path("datasets/procthor-segmentation"))