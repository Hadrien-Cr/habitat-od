from pathlib import Path
import os
import yaml
from common.utils.data_utils import save_img
from tqdm import tqdm
import json

def save_dataset(
    config_ds, 
    splits, 
    class_mapping: dict[str, int],
    class_frequency_mapping: dict[str, str]
):
    ds_path = Path(config_ds.data_root) / config_ds.dataset_name

    content = dict(
        path=str(ds_path),
        classes={i:name for i,name in enumerate(class_mapping)},
        classes_frequent = {i:name for i,name in enumerate(class_mapping) if class_frequency_mapping[name] == "frequent"},
        classes_common   = {i:name for i,name in enumerate(class_mapping) if class_frequency_mapping[name] == "common"},
        classes_rare     = {i:name for i,name in enumerate(class_mapping) if class_frequency_mapping[name] == "rare"}
    )

    with open(ds_path / "dataset.yaml","w") as f:
        yaml.dump(content,f)

    for split_name, list_samples in splits.items():

        images=[]
        annotations=[]
        ann_id=1

        for img_id, (fname, object_detection_info) in enumerate(
            tqdm(list_samples, desc=f"Saving data to {ds_path / split_name} ")
        ):
            
            images.append({
                "id": img_id,
                "file_name": "images/" +str(fname.with_suffix(".jpg")),
                "width":  640,
                "height": 480,
                "not_exhaustive_category_ids": [],
                "neg_category_ids": [],
            })

            for obj in object_detection_info:
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": class_mapping[obj["class_name"]] + 1,
                    "bbox": obj["bounding_box"],
                    "segmentation": [[float(x) for x in p ] for p in obj["mask_polygons"]],
                    "area": obj["bbx_area"],
                    "iscrowd": 0
                })
                ann_id += 1

        categories=[
            {
              "id":i+1,
              "name":name,
              "frequency": class_frequency_mapping[name][0]
            }
            for i,name in enumerate(class_mapping.keys())
        ]

        lvis_json={
            "images":images,
            "annotations":annotations,
            "categories":categories
        }
        with open(
            ds_path / f"{config_ds.dataset_name}_{split_name}.json",
            "w"
        ) as f:
            json.dump(lvis_json,f)