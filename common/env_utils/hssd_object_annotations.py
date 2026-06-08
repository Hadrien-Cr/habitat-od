import os
import json
import pandas as pd
import numpy as np
from common.env_utils.vocab_constants import VOCABULARIES, CLASS_LABELS_HSSD40, CLASS_LABELS_HSSD500

HABITAT_DATA = os.environ.get("HABITAT_DATA")

def load_hssd_object_annotations() -> dict:
    if HABITAT_DATA is None:
        raise ValueError("HABITAT_DATA environment variable is not set")
    
    objects_csv_path = os.path.join(
        HABITAT_DATA, "scene_datasets/hssd-hab/semantics/objects.csv"
    )

    object_info_ds = pd.read_csv(objects_csv_path)

    object_info_ds["main_category"] = object_info_ds["main_category"].fillna("undefined")
    object_info_ds["wnsynsetkey"] = (
        object_info_ds["wnsynsetkey"].fillna("undefined").map(lambda x: x.split(".")[0])
    )
    object_info_ds["name"] = object_info_ds["wnsynsetkey"].map(lambda x: x.split(".")[0])

    mapping_objname_category = dict(zip(object_info_ds["id"], object_info_ds["main_category"]))
    mapping_objname_wnsynsetkey = dict(zip(object_info_ds["id"], object_info_ds["wnsynsetkey"]))

    return {
        "mapping_objname_category": mapping_objname_category,
        "mapping_objname_wnsynsetkey": mapping_objname_wnsynsetkey,
    }


class ObjectSemanticsHSSD:
    source_mapping_objname_class: dict[str, str] 
    mapping_objname_class: dict[str, str] 
    classes: list[str]
    colors: list[tuple[int,int,int]]
    int2color: dict[int, tuple[int,int,int]]
    class2color: dict[str, tuple[int,int,int]]
    palette_colors: list[tuple[int,int,int]]

    def __init__(self, vocab_name: str) -> None:
        annotations = load_hssd_object_annotations()

        self.source_mapping_objname_class = annotations["mapping_objname_wnsynsetkey"]
        if vocab_name not in VOCABULARIES.keys():
            raise ValueError(f"Vocabulary {vocab_name} not recognized. Must be one of {VOCABULARIES.keys()}")
        
        target_class_labels, mapping_to_hssd500_synset, colors = VOCABULARIES[vocab_name]

        self.classes = target_class_labels
        self.colors = colors

        if vocab_name == "HSSD500":
            self.mapping_objname_class = {o: n for o, n in annotations["mapping_objname_wnsynsetkey"].items() if n != "undefined"}
        elif vocab_name == "HSSD40":
            self.mapping_objname_class = {o: n for o, n in annotations["mapping_objname_category"].items() if n != "undefined"}
        else:
            self.mapping_objname_class = {}

            for obj_name in annotations["mapping_objname_wnsynsetkey"]:
                hssd500_wnsynsetkey = annotations["mapping_objname_wnsynsetkey"][obj_name]

                mapped_class = "undefined"
                
                if hssd500_wnsynsetkey != "undefined":
                    mapped_class = mapping_to_hssd500_synset[hssd500_wnsynsetkey]

                if mapped_class != "undefined":
                    self.mapping_objname_class[obj_name] = mapped_class

        self.int2color = {i: color for i, color in enumerate(self.colors)}
        self.class2color = {cls: self.int2color[i] for i, cls in enumerate(self.classes)}
        self.palette_colors = palette_colors + self.colors


class PaletteIndices:
    """
    Indices of different types of maps maintained in the agent's map state.
    """
    EMPTY_SPACE = 0
    OBSTACLES = 1
    EXPLORED = 2
    VISITED = 3
    CLOSEST_GOAL = 4
    REST_OF_GOAL = 5
    BEEN_CLOSE = 6
    SHORT_TERM_GOAL = 7
    BLACKLISTED_TARGETS_MAP = 8
    INSTANCE_BORDER = 9
    SEM_START = 10

palette_colors = [
    (255,255,255),
    (153,153,153),
    (242,242,242),
    (245,91,66),
    (31,117,178),
    (161,199,242),
    (153,222,138),
    (0,255,0),
    (153,43,138),
    (0,0,0)
]