import os
import cv2
from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

from habitat_sim.agent.agent import AgentState

from common.utils.pose_utils import quaternion_from_rpy, rpy_from_quaternion


def rgb_to_hex(rgb):
    r,g,b = rgb
    return '#%02x%02x%02x' % (int(r), int(g), int(b))


def hex_to_rgb(hx):
    """hx is a string, begins with #. ASSUME len(hx)=7."""
    if len(hx) != 7:
        raise ValueError("Hex must be #------")
    hx = hx[1:]  # omit the '#'
    r = int('0x'+hx[:2], 16)
    g = int('0x'+hx[2:4], 16)
    b = int('0x'+hx[4:6], 16)
    return (r,g,b)


def make_colors(num, seed=1, ctype=1) -> list:
    """Return `num` number of unique colors in a list,
    where colors are [r,g,b] lists."""
    rng_gen = np.random.default_rng(seed)
    colors = []

    def random_unique_color(colors, ctype, rng_gen):
        """
        ctype=1: completely random
        ctype=2: red random
        ctype=3: blue random
        ctype=4: green random
        ctype=5: yellow random
        """
        if ctype == 1:
            color = "#%06x" % rng_gen.integers(0x444444, 0x999999)
            while color in colors:
                color = "#%06x" % rng_gen.integers(0x444444, 0x999999)
        elif ctype == 2:
            color = "#%02x0000" % rng_gen.integers(0xAA, 0xFF)
            while color in colors:
                color = "#%02x0000" % rng_gen.integers(0xAA, 0xFF)
        elif ctype == 4:  # green
            color = "#00%02x00" % rng_gen.integers(0xAA, 0xFF)
            while color in colors:
                color = "#00%02x00" % rng_gen.integers(0xAA, 0xFF)
        elif ctype == 3:  # blue
            color = "#0000%02x" % rng_gen.integers(0xAA, 0xFF)
            while color in colors:
                color = "#0000%02x" % rng_gen.integers(0xAA, 0xFF)
        elif ctype == 5:  # yellow
            h = rng_gen.integers(0xAA, 0xFF)
            color = "#%02x%02x00" % (h, h)
            while color in colors:
                h = rng_gen.integers(0xAA, 0xFF)
                color = "#%02x%02x00" % (h, h)
        else:
            raise ValueError("Unrecognized color type %s" % (str(ctype)))
        return color

    while len(colors) < num:
        colors.append(list(hex_to_rgb(random_unique_color(colors,ctype=ctype,rng_gen=rng_gen))))
    return colors


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
    str_x, str_y, str_z, str_yaw = str(round(x,2)).replace(".", "p"), str(round(y,2)).replace(".", "p"), str(round(z,2)).replace(".", "p"), str(round(yaw,2)).replace(".", "p")
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


def save_img(
    img: np.ndarray,
    data_dir: Path,
    fname: Path,
) -> Path:
    img_dir = data_dir / "images"
    os.makedirs(img_dir, exist_ok=True)
    path = img_dir / f"{str(fname)}.jpg"

    im = Image.fromarray(img)
    im.save(path, format="JPEG")
    return path


def enumerate_fnames(source_data_dir: Path) -> list[Path]:
    l = []
    if not os.path.exists(source_data_dir):
        return []
    for image_name in os.listdir(source_data_dir / "images"):
        l.append(Path(image_name))
    return sorted(l)  # type: ignore
