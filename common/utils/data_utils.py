from common.env_utils.sense import Sense, get_sense_info, get_class_from_modality_code, SenseInfo
import numpy as np
import os

def load_data(path: str) -> Sense:
    sense_info = get_sense_info(path)
    return get_class_from_modality_code(sense_info.mod).load(path)

def save_obs(dataset_path: str, episode_id: int, observations: list, timestamp: int, modalities: list[str]) -> list[str]:
    paths = []

    for camera_id, camera_obs in enumerate(observations):
        for modality, data in camera_obs.items():
            if modality not in modalities:
                continue
            saved_path = _save_data(
                dataset_path,
                int(episode_id),
                modality,
                int(camera_id),
                int(timestamp),
                data,
            )
            paths.append(saved_path)
            
    return paths

def _remove_data(dataset_path: str, episode_id: int, camera_id: int, timestamp: int, modalities: list[str]) -> None:
    for modality in modalities:
        path = f"{dataset_path}/episode_{episode_id:06d}_modality_{modality}_step_{timestamp:05d}_id_{camera_id}.npy"
        if os.path.exists(path):
            os.remove(path)

def _save_data(dataset_path: str, episode_id: int, modality: str, camera_id: int, timestamp: int, data) -> str:
    path = f"{dataset_path}/episode_{episode_id:06d}_modality_{modality}_step_{timestamp:05d}_id_{camera_id}.npy"
    np.save(
        path,
        data,
    )
    return path
