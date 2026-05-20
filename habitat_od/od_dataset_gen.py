import numpy as np
import os
from pathlib import Path
from omegaconf import OmegaConf
from PIL import Image
from collections import defaultdict
from tqdm import tqdm

import habitat # type: ignore
from habitat.config import read_write # type: ignore
from habitat.config.default import get_agent_config # type: ignore
from detectron2.utils.visualizer import ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog

from common.hssd_od_open_voc.hssd_env import HSSD_OpenVoc_Env
from common.utils.data_utils import agent_state2fname
from common.utils.dataset_utils import save_lvis_dataset, make_dataset_dict
from common.utils.plot_utils import plot_segmentation_gt
from common.utils.sampling_utils import area_bin_sampling
from common.interfaces import Labels


def od_dataset_gen(config, split_name, scenes_to_include=None) -> None:
    dataset_root = Path(config.DATA_GEN.dataset.data_root) / config.DATA_GEN.dataset.dataset_name

    if os.path.exists(dataset_root / split_name):
        overwrite = input(f"Dataset {dataset_root / split_name} already exists. Do you want to overwrite it? [y/n] ")
        if overwrite != "y":
            print("Exiting without overwriting.")
            return
        
        os.system(f"rm -rf {dataset_root / split_name}")

    os.makedirs(dataset_root / split_name / "tmp" / "images", exist_ok=True)

    rng_gen = np.random.default_rng(0)
    habitat_env = HSSD_OpenVoc_Env(config=config, vocab_name=config.DATA_GEN.vocab)
    class_mapping = habitat_env.get_class_mapping()
    classes = list(class_mapping.keys())

    meta = MetadataCatalog.get(split_name)
    meta.set(thing_classes=classes)

    per_class_candidates_samples = defaultdict(list)
    fnames = []
    per_scene_class_object_occurences = defaultdict(int)

    scenes_to_include = list(scenes_to_include) # type: ignore

    for i, scene in enumerate(scenes_to_include):
        habitat_env.change_scene(scene)
        per_scene_class_object_occurences = defaultdict(int)

        print("-----------------")
        print("Collection in Scene = ", scene, f"({i+1}/{len(scenes_to_include)})")

        habitat_obj_occupancy_grid = habitat_env.get_oracle_object_occupancy_grid(config.DATA_GEN.meters_per_grid_pixel)
        objid_to_class = habitat_env.get_objid_to_class()

        pbar = tqdm(objid_to_class.items(), desc="Processing objects")
    
        for object_id, class_name in pbar:
            if class_name == "unknown":
                continue

            if per_scene_class_object_occurences[class_name] >= 5:
                continue
            
            pbar.set_description(f"Processing objects - {class_name:20s}")
            candidate_agent_states = habitat_obj_occupancy_grid.get_all_viewpoints(object_id, viewpoint_spacing=config.DATA_GEN.viewpoint_spacing)
            rng_gen.shuffle(candidate_agent_states) # type: ignore
            candidate_agent_states = candidate_agent_states[0:config.DATA_GEN.per_class_num_samples // 4]

            if not candidate_agent_states:
                continue

            per_scene_class_object_occurences[class_name] += 1
            
            for agent_state in candidate_agent_states:
                obs, labels = habitat_env.get_obs_gt(agent_state)

                if not any([
                    inst["class_name"] == class_name and inst["mask_area"] >= config.DATA_GEN.min_pixel_area 
                    for inst in labels.instances
                ]):
                    continue
                
                fname = agent_state2fname(
                    "cls-" + class_name.replace(" ", "_") 
                    +  "-habitat_scene-" + scene 
                    + "-objid-" + str(object_id)
                    + "-agent_state-",
                    agent_state
                )

                if fname in fnames:
                    continue
                
                im = Image.fromarray(obs.rgb)
                im.save(dataset_root / split_name / "tmp" / "images" / f"{str(fname)}.jpg", format="JPEG")
                per_class_candidates_samples[class_name].append((str(fname), labels))
                fnames.append(fname)


    # Post processing: for each class, performs a downsampling to reach "per_class_num_samples"
    out : list[tuple[str, Labels]] = []
    os.makedirs(dataset_root / split_name / "images", exist_ok=True)

    pbar = tqdm(per_class_candidates_samples.items(), desc="Post-Processing classes")

    for class_name, candidate_samples in pbar:
        pbar.set_description(f"Post-Processing classes - {class_name:20s}")

        if not candidate_samples:
            continue

        rng_gen.shuffle(candidate_samples)
        selected_indices = area_bin_sampling(
            [labels for _, labels in candidate_samples],
            rng_gen,
            mask_filtering_fn=lambda m: (m["class_name"] == class_name),
            num_samples=config.DATA_GEN.per_class_num_samples,
        ) 
        assert len(selected_indices) <= config.DATA_GEN.per_class_num_samples

        selected_samples = [candidate_samples[i] for i in selected_indices]
        rejected_samples = [candidate_samples[i] for i in range(len(candidate_samples)) if i not in selected_indices]
        
        for (fname, labels) in rejected_samples:
            os.system(f"rm {dataset_root / split_name / 'tmp' / 'images' /  (fname + '.jpg')}")

        for (fname, labels) in selected_samples:
            assert os.path.exists(dataset_root / split_name / "tmp" / "images" / f"{str(fname)}.jpg")
            os.system(f"mv {dataset_root / split_name / 'tmp' / 'images' / (fname + '.jpg')} {dataset_root / split_name / 'images' / (fname + '.jpg')}")
            assert os.path.exists(dataset_root / split_name / "images" / f"{str(fname)}.jpg")

            # save a semantic visualization of the GT
            img = np.array(Image.open(dataset_root / split_name / 'images' / (fname + '.jpg')))
            dataset_dict = make_dataset_dict(img, labels, classes)       
            sem_im = plot_segmentation_gt(img, dataset_dict, meta)
            sem_im.save(dataset_root / split_name / 'images' / (fname + '_semantic_vis.jpg'), format="JPEG")

        if selected_samples:
            out.extend(selected_samples)
    
    os.system(f"rm -rf {dataset_root / split_name / 'tmp'}")

    save_lvis_dataset(
        dataset_root = dataset_root,
        dataset_name = split_name,
        list_samples = out,
        img_size=tuple(obs.rgb.shape[:2]), # type: ignore
        classes = meta.thing_classes
    )


if __name__ == "__main__":
    for config_path in [
        "habitat_od/config/data_gen_mpcat40.yaml", 
        "habitat_od/config/data_gen_nyu40.yaml", 
        "habitat_od/config/data_gen_scannet200.yaml", 
        "habitat_od/config/data_gen_hssd40.yaml",
        "habitat_od/config/data_gen_hssd500.yaml", 
    ]:
        config = habitat.get_config(config_path=config_path)

        for i in range(len(config.DATA_GEN.dataset.splits)):
            split_name = config.DATA_GEN.dataset.splits[i].split_name
            od_dataset_gen(
                config, 
                split_name=split_name, 
                scenes_to_include=config.DATA_GEN.dataset.splits[i].scenes_to_include
            )