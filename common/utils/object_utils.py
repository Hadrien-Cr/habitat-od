import csv
import pandas as pd
from .scene_utils import enumerate_scenes
from .controller import setup_controller
from tqdm import tqdm
from collections import Counter
from ai2thor.server import Event

import cv2
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


def get_object_custom_id(object_info: dict) -> str:
    object_class = object_info["objectType"]
    object_center = object_info["axisAlignedBoundingBox"]["center"]
    x, y, z = object_center["x"], object_center["y"], object_center["z"]
    return object_class+'_'+str(round(x, 3))+'_'+str(round(y, 3))+'_'+str(round(z, 3))


def get_ground_truth(
    event: Event, 
    min_pixel_area: int,
) -> dict[str, dict]:
    """
    For all objects in event.instance_masks, generate a mask and a bounding box.
    """
    assert event.instance_detections2D is not None, "Instance detections are not available"

    detection_info = {} # dictionnary of {objiid: {mask: np.ndarray, bbx: tuple}}
    object_ids = [object["objectId"] for object in event.metadata["objects"]]

    for id in event.instance_masks:
        if id not in object_ids:
            continue

        mask = event.instance_masks[id].astype(np.uint8)
        _mask_polygon, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_polygon = np.reshape(_mask_polygon[0], (-1, 2))
        xmin, ymin, xmax, ymax = cv2.boundingRect(mask)

        if (xmax - xmin) * (ymax - ymin) < min_pixel_area:
            continue
        
        object_info = None

        for obj in event.metadata["objects"]:
            if obj["objectId"] == id:
                object_info = obj
                break

        assert isinstance(object_info, dict)

        object_custom_id = get_object_custom_id(object_info)
        detection_info[object_custom_id] = dict(
            dict(
                mask_polygon=mask_polygon,
                bounding_box=(xmin, ymin, xmax, ymax),
                class_name=object_info["objectType"],
            )
        )

    return detection_info

if __name__ == "__main__":
    get_objects(["procthor-train"], "procthor-train")
    get_objects(["procthor-val"], "procthor-val")
    get_objects(["procthor-test"], "procthor-test")
    get_objects(["procthor-train", "procthor-val", "procthor-test"], "procthor-all")
    print(load_object_classes("procthor-all"))