import os
import yaml
import json
from pathlib import Path
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

data_dir = Path("data_od")

# register all datasets that are place in data_dir 
for dataset_name in [x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x))]:

    with open(data_dir / dataset_name / "dataset.yaml") as f:
        yaml_cfg = yaml.safe_load(f)

    thing_classes = list(yaml_cfg["classes"].values())
    frequent_classes = list(yaml_cfg["classes_frequent"].values())
    common_classes = list(yaml_cfg["classes_common"].values())
    rare_classes = list(yaml_cfg["classes_rare"].values())

    def load_custom_lvis_json(json_file, image_root, id_map):
        """
        Minimal LVIS/COCO-format loader that doesn't call get_lvis_instances_meta.
        """
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
                    "category_id": id_map[ann["category_id"]],   # remap to contiguous
                    "iscrowd": ann.get("iscrowd", 0),
                }
                if "segmentation" in ann:
                    obj["segmentation"] = ann["segmentation"]
                record["annotations"].append(obj)

            dataset_dicts.append(record)

        return dataset_dicts


    for split_name in ["test"]:
        full_name = f"{dataset_name}_{split_name}"
        json_file  = str(data_dir / dataset_name / f"{full_name}.json")
        image_root = str(data_dir / dataset_name / split_name)

        # Build id map from the JSON itself — no LVIS built-in lookup
        with open(json_file) as f:
            raw = json.load(f)
        cat_ids = sorted(c["id"] for c in raw["categories"])
        id_map = {v: i for i, v in enumerate(cat_ids)}

        # Register metadata FIRST
        meta = MetadataCatalog.get(full_name)
        meta.thing_classes = thing_classes
        meta.thing_dataset_id_to_contiguous_id = id_map
        meta.json_file = json_file
        meta.image_root = image_root
        meta.evaluator_type = "lvis"
        class_freq = {}

        for c in rare_classes:
            class_freq[c] = "r"

        for c in common_classes:
            class_freq[c] = "c"

        for c in frequent_classes:
            class_freq[c] = "f"

        meta.set(
            thing_classes=thing_classes,
            class_frequency=class_freq
        )

        # Register loader (captures id_map, never calls get_lvis_instances_meta)
        if full_name not in DatasetCatalog:
            DatasetCatalog.register(
                full_name,
                lambda j=json_file, r=image_root, m=id_map: load_custom_lvis_json(j, r, m)
            )