import os
import json
import csv
import pandas as pd
from dataclasses import dataclass

@dataclass
class ClassMappingHSSD:
    mapping_obj_name_category: dict[str, str]
    mapping_obj_name_wnsynsetkey: dict[str, str]
    mapping_obj_name_fullname: dict[str, str]
    mapping_obj_name_semantic_class: dict[str, str]
    class2int: dict[str, int]
    int2class: dict[str, int]


    def __init__(self):
        HABITAT_DATA = os.environ["HABITAT_DATA"]
        semantic_lexicon_path = HABITAT_DATA + "/scene_datasets/hssd-hab/semantics/hssd-hab_semantic_lexicon.json"
        objects_csv_path =  HABITAT_DATA + "/scene_datasets/hssd-hab/semantics/objects.csv"

        with open(semantic_lexicon_path) as f:
            f = json.load(f)
            self.class2int = {x["name"]: x["id"] for x in f["classes"]}
            self.int2class = {x["id"]: x["name"] for x in f["classes"]}

        object_info_ds = pd.read_csv(objects_csv_path)
        object_info_ds['main_category'] = object_info_ds['main_category'].fillna('unknown')
        object_info_ds['main_wnsynsetkey'] = object_info_ds['main_wnsynsetkey'].fillna('unknown').map(lambda x: x.split(".")[0])
        object_info_ds['wnsynsetkey'] = object_info_ds['wnsynsetkey'].fillna('unknown').map(lambda x: x.split(".")[0])
        object_info_ds['name'] = object_info_ds['wnsynsetkey'].fillna('unknown').map(lambda x: x.split(".")[0])
    
        self.mapping_obj_name_category = dict(zip(object_info_ds['id'], object_info_ds['main_category']))
        self.mapping_obj_name_wnsynsetkey = dict(zip(object_info_ds['id'], object_info_ds['wnsynsetkey']))
        self.mapping_obj_name_fullname = dict(zip(object_info_ds['id'], object_info_ds['name']))
        self.mapping_obj_name_semantic_class = {obj_name: "undefined" for obj_name in self.mapping_obj_name_category}

        for root, dirs, files in os.walk(HABITAT_DATA + "/scene_datasets/hssd-hab/objects"):    
            for file in files: 
                if file.endswith(".object_config.json"):
                    file_target = os.path.join(root, file)
                
                    obj_name = file.replace(".object_config.json","")
                    if obj_name not in self.mapping_obj_name_category:
                        continue

                    with open(file_target) as f:
                        f = json.load(f)
                        if "semantic_id" in f:
                            self.mapping_obj_name_semantic_class[obj_name] = self.int2class[f["semantic_id"]]
