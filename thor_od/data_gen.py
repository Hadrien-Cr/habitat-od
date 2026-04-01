import numpy as np
from tqdm import tqdm
from pathlib import Path
import math

from common.utils.data_utils import DiscretizedAgentPose, pose2fname
from common.utils.pose_utils import teleport_agent_pose
from common.utils.controller import setup_controller
from common.utils.object_utils import get_ground_truth

from thor_od.utils.dataset_utils import DatasetGenerationConfig, save_dataset, load_dataset
from thor_od.utils.sampling_utils import balanced_supsampling, coverage_subsampling, covisibility_subsampling


def random_teleport_collection(config: DatasetGenerationConfig) -> list[tuple[Path, np.ndarray, dict]]:
    samples = []

    rng_gen = np.random.default_rng(config.seed)

    for scene_name in tqdm(config.scenes, desc="Processing scenes"):
        scene_samples = []

        samples_per_scene = math.ceil(config.num_samples / len(config.scenes))

        controller = setup_controller(
            scene_name=scene_name,
            img_shape=(config.img_height, config.img_width),
            grid_size=config.grid_size,
            visibility_distance=config.visibility_distance,
            rotate_step_degrees=int(360 / config.yaw_bins),
            render_instance_segmentation=True,
            render_depth_image = False,
            snap_to_grid = False,
            continuous = False,
            cloud_rendering = True,
            id = 0,
            quality = "Ultra",
        )  

        for step in tqdm(range(config.downsampling_factor * samples_per_scene), desc="Generating random teleports"):
            idx_x, idx_z = rng_gen.choice(controller.grid_reachable_positions)
            idx_yaw = rng_gen.integers(0, controller.yaw_bins)
            idx_pitch = rng_gen.integers(0, controller.pitch_bins)
            
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
            label = get_ground_truth(controller.last_event, min_pixel_area=config.min_pixel_area)
            scene_samples.append((pose2fname(prefix=controller.scene_name, pose=pose), img, label))

        controller.stop()

        if config.downsampling == "random":
            scene_samples = rng_gen.choice(scene_samples, size=samples_per_scene, replace=False)
        elif config.downsampling == "covisibility":
            sampled_idx = covisibility_subsampling(scene_samples, num_samples = samples_per_scene, rng_gen=rng_gen)
            scene_samples = [scene_samples[i] for i in sampled_idx] 
        elif config.downsampling == "balanced":
            sampled_idx = balanced_supsampling(scene_samples, num_samples = samples_per_scene, rng_gen=rng_gen)
            scene_samples = [scene_samples[i] for i in sampled_idx] 
        elif config.downsampling == "coverage":
            sampled_idx = coverage_subsampling(scene_samples, num_samples = samples_per_scene, rng_gen=rng_gen)
            scene_samples = [scene_samples[i] for i in sampled_idx]
        else:
            scene_samples = scene_samples[:samples_per_scene]

        assert len(scene_samples) == samples_per_scene

        samples.extend(scene_samples)

    return samples[:config.num_samples]
