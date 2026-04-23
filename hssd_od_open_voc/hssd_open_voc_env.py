import habitat_sim
from habitat.core.env import Env
import numpy as np
import habitat.sims.habitat_simulator.sim_utilities as sutils
from hssd_od_open_voc.hssd_class_mapping import ClassMappingHSSD
from common.utils.data_utils import make_colors

import cv2

import magnum as mn
import itertools

def get_obj_from_id(
    sim: habitat_sim.Simulator,
    obj_id: int,
):
    rom = sim.get_rigid_object_manager()
    if rom.get_library_has_id(obj_id):
        return rom.get_object_by_id(obj_id)

    return None


def object_shortname_from_handle( object_handle: str) -> str:
    return object_handle.split("/")[-1].split(".")[0].split("_:")[0].split("_")[0]

HSSD_class_mapping = ClassMappingHSSD()

class HSSD_OpenVoc_Env(Env):
    class_mapping: ClassMappingHSSD
    obj_id_to_obj_shortname: dict[int, str]
    obj_id_to_obj_group: dict[int, int]
    obj_group_to_semantic_class: dict[int, str]
    obj_group_to_full_name: dict[int, str]
    obj_group_to_wnsynsetkey: dict[int, str]
    obj_group_to_category: dict[int, str]
    vocab: str = "wnsynsetkey" # which key to use to refer object class

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.class_mapping = HSSD_class_mapping

    def get_scene_name(self,) -> str:
        return self.current_episode.scene_id.split("/")[-1]

    def get_episode_goal(self,) -> dict:
        return {
            "object_id": self.current_episode.goals[0].object_id,
            "object_shortname":  object_shortname_from_handle(
                self.current_episode.goals[0].object_name
            ),
            "view_points": self.current_episode.goals[0].view_points,
        }


    def get_colors(self):
        int2color =  make_colors(len(self.get_classes()))
        return {class_name : int2color[i] for i, class_name in enumerate(self.get_classes())}
    
    def get_int2color(self):
        return make_colors(len(self.get_classes()))

    def get_class_mapping(self):
        return {class_name: i for i, class_name in enumerate(self.get_classes())}


    def get_objects(self,) -> list[dict]:
        out = []

        for obj_id, obj_group in self.obj_id_to_obj_group.items():
            obj = get_obj_from_id(self.sim, obj_id)
            obj_name = self.obj_id_to_obj_shortname[obj_id]
            class_name = self.get_class(obj_group)
            
            aabb = obj.collision_shape_aabb
            min_v = aabb.min
            max_v = aabb.max

            corners_local = [
                mnmx
                for mnmx in itertools.product(
                    [min_v.x, max_v.x],
                    [min_v.y, max_v.y],
                    [min_v.z, max_v.z],
                )
            ]
            corners_local = [mn.Vector3(c) for c in corners_local]
            corners_world = [obj.rotation.transform_vector(c) +  obj.translation for c in corners_local]
            xs = [c.x for c in corners_world]
            ys = [c.y for c in corners_world]
            zs = [c.z for c in corners_world]

            world_min = mn.Vector3(min(xs), min(ys), min(zs))
            world_max = mn.Vector3(max(xs), max(ys), max(zs))
            
            corners_global = [
                mnmx
                for mnmx in itertools.product(
                    [world_min.x, world_max.x],
                    [world_min.y, world_max.y],
                    [world_min.z, world_max.z],
                )
            ]
            corners_global = [mn.Vector3(c) for c in corners_global]

            out.append({
                "object_id": obj_id,
                "object_group": obj_group,
                "obj_name": obj_name,
                "class_name": class_name,
                "local_corners": corners_local,
                "global_corners": corners_global,
                "position": obj.translation,
                "rotation": obj.rotation
            })
        
        return out

    def update_scene(self):
        self.setup_group_semantic_labels()

        self.obj_group_to_semantic_class  = {}
        self.obj_group_to_full_name       = {}
        self.obj_group_to_wnsynsetkey     = {}
        self.obj_group_to_category        = {}

        for obj_id, obj_group in self.obj_id_to_obj_group.items():
            obj_name = self.obj_id_to_obj_shortname[obj_id]

            self.obj_group_to_semantic_class[obj_group] = self.class_mapping.mapping_obj_name_semantic_class[obj_name]
            self.obj_group_to_full_name[obj_group]      = self.class_mapping.mapping_obj_name_fullname[obj_name]
            self.obj_group_to_wnsynsetkey[obj_group]    = self.class_mapping.mapping_obj_name_wnsynsetkey[obj_name]
            self.obj_group_to_category[obj_group]       = self.class_mapping.mapping_obj_name_category[obj_name]
        
    def get_scene_annotations(self):
        if self.vocab == "semantic_class":
            return self.obj_group_to_semantic_class
        
        elif self.vocab == "full_name":
            return self.obj_group_to_full_name
        
        elif self.vocab == "wnsynsetkey":
            return self.obj_group_to_wnsynsetkey

        elif self.vocab == "category":
            return self.obj_group_to_category
        
        raise ValueError   

    def get_vocab(self):
        if self.vocab == "semantic_class":
            return self.class_mapping.mapping_obj_name_semantic_class
        
        elif self.vocab == "full_name":
            return self.class_mapping.mapping_obj_name_fullname
        
        elif self.vocab == "wnsynsetkey":
            return self.class_mapping.mapping_obj_name_wnsynsetkey

        elif self.vocab == "category":
            return self.class_mapping.mapping_obj_name_category
        
        raise ValueError   
    
    def get_class(self, obj_group: int):
        if self.vocab == "semantic_class":
            return self.obj_group_to_semantic_class[obj_group]
        
        elif self.vocab == "full_name":
            return self.obj_group_to_full_name[obj_group]
        
        elif self.vocab == "wnsynsetkey":
            return self.obj_group_to_wnsynsetkey[obj_group]

        elif self.vocab == "category":
            return self.obj_group_to_category[obj_group]
        
        raise ValueError   

    
    def get_classes(self):
        return sorted(set(self.get_vocab().values()))

    def get_mask(self, semantic_obs) -> list[tuple[str, np.ndarray]]:
        out = []
        
        values = np.unique(semantic_obs)

        for label in values:
            if label == 0:  # optional: skip background
                continue

            mask = (semantic_obs == label).astype("uint8")
            obj_group = label - 100

            class_name = self.get_class(obj_group)

            _mask_polygon, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask_polygon = np.reshape(_mask_polygon[0], (-1, 2))
            xmin, ymin, xmax, ymax = cv2.boundingRect(mask)
            out.append((class_name, {"mask": mask, "mask_polygon": mask_polygon, "bounding_box": (xmin, ymin, xmax, ymax)}))

        return out

    def setup_group_semantic_labels(self,):
        """
        This function modifies the semantic sensor such that it outputs segmentations based object groups instead of semantic info.
        To do so, we overwrite obj.semantic_id. 
        """        
        def object_shortname_from_handle( object_handle: str) -> str:
            return object_handle.split("/")[-1].split(".")[0].split("_:")[0].split("_")[0]

        rom = self._sim.get_rigid_object_manager()

        self.obj_id_to_obj_group = {}
        self.obj_id_to_obj_shortname = {}

        for obj_id, obj_handle in sutils.get_all_object_ids(self._sim).items():
            obj = rom.get_object_by_id(obj_id)
            shortname = object_shortname_from_handle(obj_handle)
            self.obj_id_to_obj_shortname[obj_id] = shortname
            assert shortname in self.class_mapping.mapping_obj_name_category
        
        self.obj_id_to_obj_group = {obj_id: obj_group for obj_group, (obj_id, obj_name) in enumerate(sorted(self.obj_id_to_obj_shortname.items()))}
        
        for _, handle in enumerate(rom.get_object_handles()):
            obj = rom.get_object_by_handle(handle)
            for node in obj.visual_scene_nodes:
                node.semantic_id = self.obj_id_to_obj_group[obj.object_id] + 100




