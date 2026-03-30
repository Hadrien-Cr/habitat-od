from thor_od.utils.dataset_utils import save_dataset, load_dataset, DetectionConfig

from common.utils.data_utils import DiscretizedAgentPose, pose2fname
from common.utils.pose_utils import teleport_agent_pose, get_ground_truth_bbx
from common.utils.controller import setup_controller
from common.utils.object_utils import load_object_classes
import numpy as np
from tqdm import tqdm
from pathlib import Path


def random_teleport_procedure(
    controller, config: DetectionConfig, num_steps: int, rng_gen
) -> list[tuple[Path, np.ndarray, list[dict]]]:
    samples = []
    
    for k in tqdm(range(num_steps), desc="Generating random teleports"):
        idx_x, idx_z = rng_gen.choice(controller.grid_reachable_positions)
        idx_yaw = rng_gen.randint(0, controller.yaw_bins)
        idx_pitch = rng_gen.randint(0, controller.pitch_bins)
        
        pose = DiscretizedAgentPose(
            idx_x=idx_x,
            idx_z=idx_z,
            idx_yaw=idx_yaw,
            idx_pitch=idx_pitch,
            yaw_bins=controller.yaw_bins,
            pitch_bins=controller.pitch_bins
        )
        
        teleport_agent_pose(controller, pose)
        img = controller.last_event.frame

        label = get_ground_truth_bbx(controller.last_event, min_pixel_area=config.min_pixel_area, class_names=config.class_names)
        samples.append((pose2fname(prefix=controller.scene_name, pose=pose), img, label))

    return samples


if __name__ == "__main__":
    import random 
    controller = setup_controller(
        scene_name="FloorPlan21",
        img_shape=(640, 640),
        grid_size=0.25,
        visibility_distance=1.0,
        rotate_step_degrees=30,
        render_instance_segmentation=True,
        render_depth_image = False,
        snap_to_grid = False,
        continuous = True,
        cloud_rendering = False,
        id = 0,
        quality = "Ultra",
    )  

    config = DetectionConfig(
        min_pixel_area=0,
        class_names=load_object_classes("all")
    )
    
    train_samples = random_teleport_procedure(controller, config, num_steps=100, rng_gen=random)
    eval_samples = random_teleport_procedure(controller, config, num_steps=100, rng_gen=random)
    save_dataset(Path("data"), Path("my_dataset"), class_names=config.class_names, splits={"train": train_samples, "eval": eval_samples})