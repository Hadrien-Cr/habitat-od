import glob
import os
import json
import pandas as pd
from common.env_utils.vocab_constants import VOCABULARIES, CLASS_LABELS_HSSD400, HSSD400_TO_VOCAB

HABITAT_DATA = os.environ.get("HABITAT_DATA")
BASE_DIR = os.environ.get("BASE_DIR")
assert BASE_DIR is not None, "BASE_DIR environment variable must be set to the repo root"

class ObjectSemanticsHSSD:
    hssd400_object_annotations: dict[str, str] # mapping from object name to hssd400 class, including "unknown" objects. should list every object that is present in condensed_hssd400_vocab
    target_vocab_object_annotations: dict[str, str] # mapping from object name to target vocabulary class, excluding "unknown" objects
    classes: list[str]
    colors: list[tuple[int,int,int]]
    int2color: dict[int, tuple[int,int,int]]
    class2color: dict[str, tuple[int,int,int]]
    palette_colors: list[tuple[int,int,int]]

    def __init__(self, vocab_name: str) -> None:
        if vocab_name not in VOCABULARIES:
            raise ValueError(f"Vocabulary {vocab_name} not recognized. Must be one of {VOCABULARIES.keys()}")

        target_class_labels, _, colors = VOCABULARIES[vocab_name]
        self.classes = target_class_labels
        self.colors = colors

        self.target_vocab_object_annotations: dict[str, str] = {}
        self.hssd400_object_annotations: dict[str, str] = {}

        df_hssd400 = pd.read_csv(os.path.join(str(BASE_DIR), "common", "env_utils", "hssd_obj_semantics_condensed.csv")).set_index("Object Hash")
        # raw_hssd400_vocab = df_hssd400_vocab.iloc[:,3].to_dict()
        condensed_hssd400_vocab = df_hssd400.iloc[:,2].to_dict()
        pickable = dict(zip(df_hssd400.index, map(lambda x: x == "Yes", df_hssd400.iloc[:,1].to_dict().values())))

        for obj_name, s in condensed_hssd400_vocab.items():
            hssd400_class = str(s).replace("/", "_")

            if "unknown" in hssd400_class or hssd400_class == "nan":
                hssd400_class = "unknown"

            assert hssd400_class in CLASS_LABELS_HSSD400, (hssd400_class,s)
    
            self.hssd400_object_annotations[str(obj_name)] = hssd400_class

            if vocab_name == "HSSD400":
                if hssd400_class != "unknown":
                    self.target_vocab_object_annotations[str(obj_name)] = hssd400_class
            
            else:
                if hssd400_class != "unknown" and HSSD400_TO_VOCAB[vocab_name][hssd400_class] != "unknown":
                    self.target_vocab_object_annotations[str(obj_name)] = HSSD400_TO_VOCAB[vocab_name][hssd400_class]


        assert "unknown" in self.hssd400_object_annotations.values(), "HSSD400 annotations should contain 'unknown' class"
        assert "unknown" not in self.target_vocab_object_annotations.values(), "Target vocabulary annotations should not contain 'unknown' class"
        
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