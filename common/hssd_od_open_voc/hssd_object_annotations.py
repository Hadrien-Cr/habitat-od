import os
import json
import pandas as pd
from dataclasses import dataclass, field
import numpy as np
from common.hssd_od_open_voc.vocab_constants import VOCABULARIES, CLASS_LABELS_HSSD40, CLASS_LABELS_HSSD500

HABITAT_DATA = os.environ.get("HABITAT_DATA")


def rgb_to_hex(rgb):
    r,g,b = rgb
    return '#%02x%02x%02x' % (int(r), int(g), int(b))

def hex_to_rgb(hx):
    """hx is a string, begins with #. ASSUME len(hx)=7."""
    if len(hx) != 7:
        raise ValueError("Hex must be #------")
    hx = hx[1:]  # omit the '#'
    r = int('0x'+hx[:2], 16)
    g = int('0x'+hx[2:4], 16)
    b = int('0x'+hx[4:6], 16)
    return (r,g,b)

def make_colors(num, seed=1, ctype=1) -> list[tuple[int,int,int]]:
    """Return `num` number of unique colors in a list,
    where colors are [r,g,b] lists."""
    rng_gen = np.random.default_rng(seed)
    colors = []

    def random_unique_color(colors, ctype, rng_gen):
        """
        ctype=1: completely random
        ctype=2: red random
        ctype=3: blue random
        ctype=4: green random
        ctype=5: yellow random
        """
        if ctype == 1:
            color = "#%06x" % rng_gen.integers(0x444444, 0x999999)
            while color in colors:
                color = "#%06x" % rng_gen.integers(0x444444, 0x999999)
        elif ctype == 2:
            color = "#%02x0000" % rng_gen.integers(0xAA, 0xFF)
            while color in colors:
                color = "#%02x0000" % rng_gen.integers(0xAA, 0xFF)
        elif ctype == 4:  # green
            color = "#00%02x00" % rng_gen.integers(0xAA, 0xFF)
            while color in colors:
                color = "#00%02x00" % rng_gen.integers(0xAA, 0xFF)
        elif ctype == 3:  # blue
            color = "#0000%02x" % rng_gen.integers(0xAA, 0xFF)
            while color in colors:
                color = "#0000%02x" % rng_gen.integers(0xAA, 0xFF)
        elif ctype == 5:  # yellow
            h = rng_gen.integers(0xAA, 0xFF)
            color = "#%02x%02x00" % (h, h)
            while color in colors:
                h = rng_gen.integers(0xAA, 0xFF)
                color = "#%02x%02x00" % (h, h)
        else:
            raise ValueError("Unrecognized color type %s" % (str(ctype)))
        return color

    while len(colors) < num:
        colors.append(hex_to_rgb(random_unique_color(colors,ctype=ctype,rng_gen=rng_gen)))
    return colors


def load_hssd_object_annotations() -> dict:
    if HABITAT_DATA is None:
        raise ValueError("HABITAT_DATA environment variable is not set")

    semantic_lexicon_path = os.path.join(
        HABITAT_DATA, "scene_datasets/hssd-hab/semantics/hssd-hab_semantic_lexicon.json"
    )
    objects_csv_path = os.path.join(
        HABITAT_DATA, "scene_datasets/hssd-hab/semantics/objects.csv"
    )

    with open(semantic_lexicon_path, "r") as f:
        lexicon = json.load(f)

    class2int = {x["name"]: x["id"] for x in lexicon["classes"]}
    int2class = {x["id"]: x["name"] for x in lexicon["classes"]}

    object_info_ds = pd.read_csv(objects_csv_path)

    object_info_ds["main_category"] = object_info_ds["main_category"].fillna("unknown")
    object_info_ds["main_wnsynsetkey"] = (
        object_info_ds["main_wnsynsetkey"].fillna("unknown").map(lambda x: x.split(".")[0])
    )
    object_info_ds["wnsynsetkey"] = (
        object_info_ds["wnsynsetkey"].fillna("unknown").map(lambda x: x.split(".")[0])
    )
    object_info_ds["name"] = object_info_ds["wnsynsetkey"].map(lambda x: x.split(".")[0])

    mapping_objname_category = dict(zip(object_info_ds["id"], object_info_ds["main_category"]))
    mapping_objname_wnsynsetkey = dict(zip(object_info_ds["id"], object_info_ds["wnsynsetkey"]))
    mapping_objname_fullname = dict(zip(object_info_ds["id"], object_info_ds["name"]))

    mapping_objname_semantic_class = {
        obj_name: "undefined" for obj_name in mapping_objname_category
    }
    
    objects_root = os.path.join(HABITAT_DATA, "scene_datasets/hssd-hab/objects")

    for root, _, files in os.walk(objects_root):
        for file in files:
            if file.endswith(".object_config.json"):
                file_target = os.path.join(root, file)
                obj_name = file.replace(".object_config.json", "")

                if obj_name not in mapping_objname_category:
                    continue

                with open(file_target, "r") as f:
                    obj_data = json.load(f)

                semantic_id = obj_data.get("semantic_id")
                if semantic_id is not None and semantic_id in int2class:
                    mapping_objname_semantic_class[obj_name] = int2class[semantic_id]

    return {
        "mapping_objname_category": mapping_objname_category,
        "mapping_objname_wnsynsetkey": mapping_objname_wnsynsetkey,
        "mapping_objname_fullname": mapping_objname_fullname,
        "mapping_objname_semantic_class": mapping_objname_semantic_class, # this one looks inconsistent
    }


class ObjectSemanticsHSSD:
    source_mapping_objname_class: dict[str, str] 
    mapping_objname_class: dict[str, str] 
    colors: list[tuple[int,int,int]]
    int2color: dict[int, tuple[int,int,int]]
    class2color: dict[str, tuple[int,int,int]]
    palette_colors: list[tuple[int,int,int]]

    def __init__(self, vocab_name: str) -> None:
        annotations = load_hssd_object_annotations()

        self.source_mapping_objname_class = annotations["mapping_objname_wnsynsetkey"]
        if vocab_name not in VOCABULARIES.keys():
            raise ValueError(f"Vocabulary {vocab_name} not recognized. Must be one of {VOCABULARIES.keys()}")
        
        if vocab_name == "HSSD500":
            self.mapping_objname_class = annotations["mapping_objname_wnsynsetkey"]
        elif vocab_name == "HSSD40":
            self.mapping_objname_class = annotations["mapping_objname_category"]

        else:
            target_class_labels, mapping_to_hssd500_synset, mapping_to_hssd500_cat = VOCABULARIES[vocab_name]
            self.mapping_objname_class = {}

            for obj_name in annotations["mapping_objname_wnsynsetkey"]:
                hssd500_wnsynsetkey = annotations["mapping_objname_wnsynsetkey"][obj_name]
                hssd500_wnsynsetkey_id = CLASS_LABELS_HSSD500.index(hssd500_wnsynsetkey)
                mapped_class_id = mapping_to_hssd500_synset[hssd500_wnsynsetkey_id]
                mapped_class = target_class_labels[mapped_class_id]
                
                if mapped_class == "unknown":
                    hssd40_cat = annotations["mapping_objname_category"][obj_name]
                    hssd40_cat_id = CLASS_LABELS_HSSD40.index(hssd40_cat)
                    mapped_class_id = mapping_to_hssd500_cat[hssd40_cat_id]
                    mapped_class = target_class_labels[mapped_class_id]

                self.mapping_objname_class[obj_name] = mapped_class



        classes = sorted(set(self.mapping_objname_class.values()))
        self.colors = make_colors(len(classes), seed=0, ctype=1)
        self.colors[classes.index("unknown")] = (0,0,0)  # make unknown class black
        self.int2color = {i: color for i, color in enumerate(self.colors)}
        self.class2color = {cls: self.int2color[i] for i, cls in enumerate(classes)}
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