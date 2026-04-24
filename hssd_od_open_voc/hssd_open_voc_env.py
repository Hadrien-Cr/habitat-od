import os
import cv2
import numpy as np
import magnum as mn
import itertools

import habitat_sim
from habitat.core.env import Env
from habitat.config import read_write
import habitat.sims.habitat_simulator.sim_utilities as sutils
from hssd_od_open_voc.hssd_object_annotations import ObjectAnnotationHSSD
from habitat_od.utils.data_utils import make_colors
from habitat_od.utils.grid_utils import HabitatSemGrid

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

HSSD_object_annot = ObjectAnnotationHSSD()

class HSSD_OpenVoc_Env(Env):
    object_annotations: ObjectAnnotationHSSD
    obj_id_to_obj_shortname: dict[int, str]
    vocab: str = "wnsynsetkey" # which key to use to refer object class

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.object_annotations = HSSD_object_annot

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


    def change_scene(self, scene: str) -> None:
        scenes_dir = self._config.dataset.scenes_dir

        with read_write(self._config):
            self._config.simulator.scene = scenes_dir + "/" +  scene

        self._sim.reconfigure(self._config.simulator)


    def get_scenes_names(self,) -> list[str]:
        scenes_dir = self._config.dataset.scenes_dir
        content_scenes = self._config.dataset.content_scenes

        if content_scenes == ["*"]:
            return  [ x.replace(".scene_instance.json", "") for x in os.listdir(scenes_dir)]
        
        return content_scenes


    def get_oracle_semantic_grid(self, meters_per_grid_pixel) -> HabitatSemGrid:
        return HabitatSemGrid(
            self.sim,
            meters_per_grid_pixel=meters_per_grid_pixel,
            class_mapping=self.get_class_mapping(),
            list_object_info=self.get_objects()
        )

    def get_name2color(self) -> dict[str, tuple]:
        int2color =  make_colors(len(self.get_classes()))
        return {class_name : int2color[i] for i, class_name in enumerate(self.get_classes())}
    
    def get_int2color(self) -> dict[int, str]:
        int2color =  make_colors(len(self.get_classes()))
        return {i : int2color[i] for i, class_name in enumerate(self.get_classes())}

    def get_class_mapping(self) -> dict[str, int]:
        return {class_name: i for i, class_name in enumerate(self.get_classes())}

    def get_objects(self,) -> list[dict]:
        out = []

        for obj_id, obj_name in self.obj_id_to_obj_shortname.items():
            obj = get_obj_from_id(self.sim, obj_id)
            class_name = self.get_class(obj_name)

            aabb = obj.collision_shape_aabb # type: ignore
            min_v = aabb.min
            max_v = aabb.max

            # 8 local box corners
            corners_local = [
                mn.Vector3(c) # type: ignore
                for c in itertools.product(
                    [min_v.x, max_v.x],
                    [min_v.y, max_v.y],
                    [min_v.z, max_v.z],
                )
            ]

            # Transform to world space (preserves orientation)
            corners_world = [
                obj.rotation.transform_vector(c) + obj.translation # type: ignore
                for c in corners_local
            ]

            out.append({
                "object_id": obj_id,
                "obj_name": obj_name,
                "class_name": class_name,

                "position": obj.translation, # type: ignore
                "rotation": obj.rotation, # type: ignore
                "corners": [
                    (c.x, c.y, c.z)
                    for c in corners_world
                ],
            })
        
        return out


    def update_scene(self) -> None:
        def object_shortname_from_handle( object_handle: str) -> str:
            return object_handle.split("/")[-1].split(".")[0].split("_:")[0].split("_")[0]

        # setup the dictionnary obj_id_to_obj_shortname
        self.obj_id_to_obj_shortname = {}
        vocab = self.get_vocab()

        for obj_id, obj_handle in sutils.get_all_object_ids(self.sim).items():
            shortname = object_shortname_from_handle(obj_handle)
            self.obj_id_to_obj_shortname[obj_id] = shortname
            assert shortname in vocab

        self.setup_semantic_labels()


    def get_scene_annotations(self) -> dict[int, str]:
        vocab = self.get_vocab()
        return {obj_id: vocab[obj_name] for obj_id, obj_name in self.obj_id_to_obj_shortname.items()} 

    def get_vocab(self) -> dict[str, str]:
        if self.vocab == "semantic_class":
            return self.object_annotations.mapping_obj_name_semantic_class
        
        elif self.vocab == "full_name":
            return self.object_annotations.mapping_obj_name_fullname
        
        elif self.vocab == "wnsynsetkey":
            return self.object_annotations.mapping_obj_name_wnsynsetkey

        elif self.vocab == "category":
            return self.object_annotations.mapping_obj_name_category
        
        raise ValueError   
    
    def get_class(self, obj_name: str):
        return self.get_vocab()[obj_name]

    def get_classes(self):
        return sorted(set(self.get_vocab().values()))

    def decompose_frame(self, semantic_obs) -> list[dict]:
        out = []
        
        values = np.unique(semantic_obs)

        annotations = self.get_scene_annotations()

        for label in values:
            if label == 0:  # optional: skip background
                continue

            mask = (semantic_obs == label).astype("uint8")
            
            obj_id = label - 100
            class_name = annotations[obj_id]

            _mask_polygon, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask_polygon = np.reshape(_mask_polygon[0], (-1, 2))
            xmin, ymin, xmax, ymax = cv2.boundingRect(mask)

            bbx_area = (xmax - xmin) * (ymax -ymin)
            mask_area = np.sum(mask)

            out.append( 
                {
                    "class_name": class_name,
                    "mask": mask, 
                    "mask_polygon": mask_polygon, 
                    "bounding_box": (xmin, ymin, xmax, ymax),
                    "bbx_area": bbx_area,
                    "mask_area": mask_area
                }
            )

        return out

    def setup_semantic_labels(self,):
        """
        This function modifies the semantic sensor such that it outputs segmentations based object instead of scene semantic info.
        To do so, we overwrite obj.semantic_id. 
        """        
        rom = self.sim.get_rigid_object_manager()

        for _, handle in enumerate(rom.get_object_handles()):
            obj = rom.get_object_by_handle(handle)
            for node in obj.visual_scene_nodes:
                node.semantic_id = obj.object_id + 100