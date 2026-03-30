import csv
import pandas as pd
from .scene_utils import enumerate_scenes
from .controller import setup_controller
from tqdm import tqdm
from collections import Counter


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
    

if __name__ == "__main__":
    get_objects(["procthor-train"], "procthor")
    get_objects(["kitchen"], "kitchen")
    get_objects(["living_room"], "living_room")
    get_objects(["bathroom"], "bathroom")
    get_objects(["bedroom"], "bedroom")
    get_objects(["kitchen", "living_room", "bathroom", "bedroom", "procthor-train"], "all")
    
    print(load_object_classes("procthor"))
    print(load_object_classes("kitchen"))
    print(load_object_classes("living_room"))
    print(load_object_classes("bathroom"))
    print(load_object_classes("bedroom"))
    print(load_object_classes("all"))