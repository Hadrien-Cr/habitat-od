import os
import yaml
import json
from pathlib import Path
from common.utils.dataset_utils import load_custom_lvis_json
from detectron2.data import MetadataCatalog, DatasetCatalog

data_dir = Path("datasets")

# register all datasets that are place in data_dir 
for dataset_name in [x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x))]:

    split_names = [x for x in os.listdir(data_dir / dataset_name) if os.path.isdir(os.path.join(data_dir / dataset_name, x))]

    for split_name in split_names:
        
        json_file  = str(data_dir / dataset_name / f"{split_name}.json")
        image_root = str(data_dir / dataset_name / split_name)

        if not Path(json_file).exists():
            continue

        with open(data_dir / dataset_name / f"{split_name}.yaml") as f:
            yaml_cfg = yaml.safe_load(f)

        thing_classes = list(yaml_cfg["classes"].values())
        frequent_classes = list(yaml_cfg["classes_frequent"].values())
        common_classes = list(yaml_cfg["classes_common"].values())
        rare_classes = list(yaml_cfg["classes_rare"].values())

        # Build id map from the JSON itself — no LVIS built-in lookup
        with open(json_file) as f:
            raw = json.load(f)

        cat_ids = sorted(c["id"] for c in raw["categories"])
        id_map = {v: i for i, v in enumerate(cat_ids)}

        meta = MetadataCatalog.get(split_name)
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
            frequent_classes=frequent_classes,
            common_classes=common_classes,
            rare_classes=rare_classes,
            class_frequency=class_freq
        )

        if split_name not in DatasetCatalog:
            DatasetCatalog.register(
                split_name,
                lambda j=json_file, r=image_root, m=id_map: load_custom_lvis_json(j, r, m)
            )