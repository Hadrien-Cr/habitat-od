import csv
import pandas as pd
from .scene_utils import enumerate_scenes
from .controller import setup_controller
from tqdm import tqdm
from collections import Counter
from ai2thor.server import Event
import numpy as np


def get_objects(scene_types: list[str], save_name: str):
    scenes = {"train": [], "val": []}

    counter = Counter()
    scene_counter = Counter()
    properties = ["pickupable", "moveable", "toggleable","openable"]
    attributes = {}

    for scene_type in scene_types:
        scenes[scene_type] = enumerate_scenes(scene_type)

        for scene_name in tqdm(scenes[scene_type], desc=f"Processing scenes of type {scene_type}"):
            controller = setup_controller(
                scene_name=scene_name,
                img_shape=(640, 640),
                grid_size=0.25,
                visibility_distance=1.0,
                rotate_step_degrees=30,
                render_instance_segmentation=False,
                render_depth_image = False,
                snap_to_grid = False,
                continuous = False,
                cloud_rendering = True,
                id = 0,
                quality = "Ultra",
            )  

            event = controller.last_event
            objects = event.metadata["objects"] # type:ignore

            for obj in objects:
                obj_type = obj["objectType"]
                if obj_type not in attributes:
                    attributes[obj_type] = {p: obj[p] for p in properties}

            counter.update([obj["objectType"] for obj in objects])
            scene_counter.update(list(set([obj["objectType"] for obj in objects])))
            controller.stop()

    with open(f'common/objectdata/{save_name}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        data = [["objectType", "#items", "#scenes"] + properties]

        for obj_type, count in sorted(counter.items()):
            data.append([obj_type, count, scene_counter[obj_type]] + [attributes[obj_type][p] for p in properties])

        writer.writerows(data)


def load_object_classes(save_name: str) -> list[str]:
    data = pd.read_csv(f'common/objectdata/{save_name}.csv')
    return list(data["objectType"])


def get_object_class_position(object_custom_id: str) -> tuple[str, float, float, float]:
    parts = object_custom_id.split('_')
    return parts[0], float(parts[1]), float(parts[2]), float(parts[3])


def get_ground_truth_bbx(
    event: Event, 
    min_pixel_area: int,
) -> list:
    
    def item_is_a_child(obj_id: str) -> bool:
        return "___" in obj_id

    seen_obj_ids = set()
    bounding_boxes = []
    
    visible_objects = {
        obj["objectId"]: obj["visible"] or obj["parentReceptacles"] is None
        for obj in event.metadata["objects"]
    }

    assert event.instance_detections2D is not None

    for objid in event.instance_detections2D:
        if objid in seen_obj_ids or not visible_objects.get(objid, False) or item_is_a_child(objid):
            continue

        xmin, ymin, xmax, ymax = event.instance_detections2D[objid]
        if (xmax - xmin) * (ymax - ymin) < min_pixel_area:
            continue
        
        seen_obj_ids.add(objid)

        object_info = None

        for obj in event.metadata["objects"]:
            if obj["objectId"] == objid:
                object_info = obj
                break
        
        assert isinstance(object_info, dict)
        object_class = object_info["objectType"]
        object_center = object_info ["axisAlignedBoundingBox"]["center"]

        x, y, z = object_center["x"], object_center["y"], object_center["z"]
        object_custom_id = object_class+'_'+str(round(x, 3))+'_'+str(round(y, 3))+'_'+str(round(z, 3))
        
        bounding_boxes.append(
            dict(
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
                class_name=object_class,
                object_id=object_custom_id
            )
        )
    return bounding_boxes

if __name__ == "__main__":
    get_objects(["procthor-train"], "procthor-train")
    get_objects(["procthor-val"], "procthor-val")
    get_objects(["procthor-test"], "procthor-test")
    get_objects(["procthor-train", "procthor-val", "procthor-test"], "procthor-all")
    print(load_object_classes("procthor-all"))