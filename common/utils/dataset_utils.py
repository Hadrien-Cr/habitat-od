from pathlib import Path
import os
import yaml
from tqdm import tqdm
import json
from common.interfaces import Labels
import numpy as np
from collections import defaultdict
from detectron2.structures import BoxMode


def make_dataset_dict(img, labels: Labels, classes: list[str]):
    h, w = img.shape[:2]
    class_mapping = {name: i for i, name in enumerate(classes)}

    return {
        "image": img,
        "height": h,
        "width": w,
        "annotations": [
            {
                "bbox": obj["bounding_box"],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": class_mapping[obj["class_name"]],
                "segmentation": [
                    [float(x) for x in p]
                    for p in obj["mask_polygons"]
                ],
                "iscrowd": 0,
            }
            for obj in labels.instances
        ],
    }


def save_lvis_dataset(
    dataset_root,
    dataset_name,
    list_samples: list[tuple[str, Labels]],
    img_size: tuple[int, int],
    classes: list[str]
):
    ds_path = Path(dataset_root) / dataset_name
    ds_path.mkdir(parents=True, exist_ok=True)

    per_class_occurences = defaultdict(int)

    for (fname, labels) in tqdm(list_samples, desc=f"Saving dataset '{dataset_name}'"):
        object_set = set()

        for inst in labels.instances:
            if (inst["object_id"]) not in object_set:
                object_set.add(inst["object_id"])
                per_class_occurences[inst["class_name"]] += 1
    
    k = len(per_class_occurences)
    occ = sorted(per_class_occurences.values())
    one_third_value, two_third_value = occ[k//3], occ[(2*k)//3]
    class_frequency_mapping = {
        name: "rare" if f <= one_third_value else ("common" if f <= two_third_value else "frequent")
        for name, f in per_class_occurences.items()
    }

    for name in classes:
        if name not in class_frequency_mapping:
            class_frequency_mapping[name] = "rare"

    class_mapping = {name: i for i, name in enumerate(classes)}

    images = []
    annotations = []
    ann_id = 1

    for img_id, (fname, labels) in enumerate(
        tqdm(list_samples, desc=f"Saving dataset '{dataset_name}' to {ds_path}")
    ):
        rel_path = f"images/{Path(fname).with_suffix('.jpg')}"
        images.append({
            "id": img_id,
            "file_name": rel_path,
            "width": img_size[1],
            "height": img_size[0],
            "not_exhaustive_category_ids": [],
            "neg_category_ids": [],
        })
        
        annot = [
            {
                "bbox": obj["bounding_box"],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": class_mapping[obj["class_name"]] + 1,
                "segmentation": [
                    [float(x) for x in p]
                    for p in obj["mask_polygons"]
                ],
                "area": obj.get("mask_area", 0),
                "iscrowd": 0,
            }
            for obj in labels.instances
        ]
        
        for ann in annot:
            ann["image_id"] = img_id
            ann["id"] = ann_id
            annotations.append(ann)
            ann_id += 1

    categories = [
        {
            "id": i + 1,
            "name": name,
            "frequency": class_frequency_mapping[name][0],
        }
        for name, i in sorted(class_mapping.items(), key=lambda kv: kv[1])
    ]

    lvis_json = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    json_path = ds_path / ".." / f"{dataset_name}.json"
    with open(json_path, "w") as f:
        json.dump(lvis_json, f)


    content = dict(
        path=str(ds_path),
        classes={
            i: name for name, i in class_mapping.items()
        },
        classes_frequent={
            i: name for name, i in class_mapping.items()
            if class_frequency_mapping[name] == "frequent"
        },
        classes_common={
            i: name for name, i in class_mapping.items()
            if class_frequency_mapping[name] == "common"
        },
        classes_rare={
            i: name for name, i in class_mapping.items()
            if class_frequency_mapping[name] == "rare"
        },
    )

    with open(ds_path / ".." / f"{dataset_name}.yaml", "w") as f:
        yaml.dump(content, f)



def load_custom_lvis_json(json_file, image_root, id_map):
    with open(json_file) as f:
        data = json.load(f)

    img_id_to_info = {img["id"]: img for img in data["images"]}

    # Group annotations by image
    ann_by_img = {}
    for ann in data.get("annotations", []):
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    dataset_dicts = []
    for img_id, img_info in img_id_to_info.items():
        record = {
            "file_name": str(Path(image_root) / img_info["file_name"]),
            "image_id": img_id,
            "height": img_info["height"],
            "width": img_info["width"],
            "annotations": [],
        }
        for ann in ann_by_img.get(img_id, []):
            obj = {
                "bbox": ann["bbox"],                          # [x, y, w, h]
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": id_map[ann["category_id"]],
                "iscrowd": ann.get("iscrowd", 0),
                "segmentation": ann.get("segmentation", []),
                "area": ann.get("area", 0),
            }
            record["annotations"].append(obj)

        dataset_dicts.append(record)

    return dataset_dicts
