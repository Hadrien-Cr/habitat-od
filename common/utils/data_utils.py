import os
import cv2
from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

from habitat_sim.agent.agent import AgentState

from common.utils.pose_utils import quaternion_from_rpy, rpy_from_quaternion


def load_img(path: Path) -> np.ndarray:
    # Check image extension and try to load accordingly
    img_path_jpg = (path.with_suffix(".jpg"))
    img_path_png = (path.with_suffix(".png"))

    if img_path_jpg.exists():
        img_path = img_path_jpg
    elif img_path_png.exists():
        img_path = img_path_png
    else:
        raise FileNotFoundError(f"No image file found for {path}  with .jpg or .png extension.")

    with open(img_path, "rb") as f:
        im = Image.open(f)
        im = np.array(im)

    assert isinstance(im, np.ndarray), f"Image loading failed for {path}"

    return im


def agent_state2fname(prefix: str, pose: AgentState) -> Path:
    """Get the filename corresponding to the given pose."""
    (x,y,z) = pose.position
    (_,_,yaw) = rpy_from_quaternion(pose.rotation)
    str_x, str_y, str_z, str_yaw = str(round(x,3)).replace(".", "p"), str(round(y,3)).replace(".", "p"), str(round(z,3)).replace(".", "p"), str(round(yaw,3)).replace(".", "p")
    fname = Path(f"{prefix}_x_{str_x}_y_{str_y}_z_{str_z}_yaw_{str_yaw}")
    return fname


def fname2agent_state(fname: Path) -> AgentState:
    fname_str = str(fname.stem)

    def extract_numerical_value(string: str, prefix: str):
        where_start = fname_str.find(prefix) + len(prefix)
        where_end = where_start + (
            fname_str[where_start:].find("_")
            if fname_str[where_start:].find("_") != -1
            else len(fname_str[where_start:])
        )
        return float(fname_str[where_start:where_end].replace("p", "."))
    
    new_state = AgentState()

    x = extract_numerical_value(fname_str, "_x_")
    y = extract_numerical_value(fname_str, "_y_")
    z = extract_numerical_value(fname_str, "_z_")
    yaw = int(extract_numerical_value(fname_str, "_yaw_"))

    new_state.position = np.array([x,y,z], dtype = np.float32)
    new_state.rotation = quaternion_from_rpy(0, 0,  yaw)
    return new_state


def delete_image(
    data_dir: Path,
    fname: Path,
):
    img_dir = data_dir / "images"
    path = img_dir / f"{str(fname)}.jpg"
    os.remove(path)

def move_image(
    data_dir: Path,
    fname: Path,
    target_dir: Path
):
    img_dir = data_dir / "images"
    path = img_dir / f"{str(fname)}.jpg"
    target_path = target_dir / "images" / f"{str(fname)}.jpg"

    assert path.exists(), f"Image to move not found at {path}"
    os.makedirs(target_dir / "images", exist_ok=True)
    os.rename(path, target_path)
    assert target_path.exists(), f"Failed to move image from {path} to {target_path}"


def enumerate_fnames(source_data_dir: Path) -> list[Path]:
    l = []
    if not os.path.exists(source_data_dir):
        return []
    for image_name in os.listdir(source_data_dir / "images"):
        l.append(Path(image_name))
    return sorted(l)  # type: ignore
