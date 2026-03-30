import os
import uuid
from pathlib import Path
import numpy as np
from PIL import Image

from .pose_utils import DiscretizedAgentPose


def xyxy_to_normalized_xywh(box: tuple[int, int, int, int], size: tuple[int,int], center=True) -> tuple[float, float, float, float]:
    img_width, img_height = size
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1
    x = x1
    y = y1
    w = box_width
    h = box_height
    if center:
        x = ((x1 + x2) / 2)
        y = ((y1 + y2) / 2)
    x /= img_width
    y /= img_height
    w /= img_width
    h /= img_width
    return x, y, w, h


def normalized_xywh_to_xyxy(xywh: tuple[float, float, float, float], size: tuple[int,int], center=True) -> tuple[int, int, int, int]:
    x, y, w, h = xywh
    img_width, img_height = size
    x *= img_width
    y *= img_height
    w *= img_width
    h *= img_height
    if center:
        x1 = int(round(x - w / 2))
        x2 = int(round(x + w / 2))
        y1 = int(round(y - h / 2))
        y2 = int(round(y + h / 2))
    else:
        x1 = int(round(x))
        y1 = int(round(y))
        x2 = int(round(x + w))
        y2 = int(round(y + h))
    return x1, y1, x2, y2


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


def load_label(
    path: Path, fname: Path, class_names: list[str], img_size: tuple[int,int]
) -> list[dict]:
    w, h = img_size

    label_boxes: list[dict] = []

    with open(path) as fa:
        for line in fa.readlines():
            annot = line.strip().split()
            assert len(annot) == 5

            class_id = int(annot[0])
            x,y,w,h = list(map(float, annot[1:]))
            xyxy = normalized_xywh_to_xyxy(
                (x,y,w,h), img_size, center=True
            )

            label_boxes.append(
                dict(
                    xmin=xyxy[0],
                    ymin=xyxy[1],
                    xmax=xyxy[2],
                    ymax=xyxy[3],
                    confidence=1.0,  # GT dummy confidence
                    bbx_confidence=1.0,
                    class_wise_confidence=np.array(
                        [i == class_id for i in range(len(class_names))], dtype=float
                    ),
                    class_id=class_id,
                    class_name=class_names[class_id],
                    bbx_features=np.array([], dtype=float),
                )
            )
    return label_boxes


def load_img(fname: Path) -> np.ndarray:
    # Check image extension and try to load accordingly
    img_path_jpg = (fname.with_suffix(".jpg"))
    img_path_png = (fname.with_suffix(".png"))

    if img_path_jpg.exists():
        img_path = img_path_jpg
    elif img_path_png.exists():
        img_path = img_path_png
    else:
        raise FileNotFoundError(f"No image file found for {fname}  with .jpg or .png extension.")

    with open(img_path, "rb") as f:
        im = Image.open(f)
        im = np.array(im)

    assert isinstance(im, np.ndarray), f"Image loading failed for {fname}"

    return im


def pose2fname(prefix: str, pose: DiscretizedAgentPose) -> Path:
    """Get the filename corresponding to the given pose."""
    fname = Path(f"{prefix}-x{pose.idx_x}-z{pose.idx_z}-y{pose.idx_yaw}-p{pose.idx_pitch}-by{pose.yaw_bins}-bp{pose.pitch_bins}")
    return fname


def fname2pose(fname: Path) -> DiscretizedAgentPose:
    fname_str = str(fname.stem)

    def extract_numerical_value(string: str, prefix: str):
        where_start = fname_str.find(prefix) + len(prefix)
        where_end = where_start + (
            fname_str[where_start:].find("-")
            if fname_str[where_start:].find("-") != -1
            else len(fname_str[where_start:])
        )
        return float(fname_str[where_start:where_end])

    idx_x = int(extract_numerical_value(fname_str, "-x"))
    idx_z = int(extract_numerical_value(fname_str, "-z"))
    idx_yaw = int(extract_numerical_value(fname_str, "-y"))
    idx_pitch = int(extract_numerical_value(fname_str, "-p"))
    yaw_bins = int(extract_numerical_value(fname_str, "-by"))
    pitch_bins = int(extract_numerical_value(fname_str, "-bp"))

    return DiscretizedAgentPose(idx_x, idx_z, idx_yaw, idx_pitch, yaw_bins, pitch_bins)

def save_label(
    gt_bounding_boxes: list[dict],
    data_dir: Path,
    fname: Path,
    img_shape: tuple[int, int, int],
) -> None:
    label_dir = data_dir / "labels"
    os.makedirs(label_dir, exist_ok=True)

    path = label_dir / (fname.stem + ".txt")

    if path.exists():
        return

    annotations = []

    for bbx in sorted(gt_bounding_boxes, key=lambda x: x["class_id"]):
        x_center, y_center, w, h = xyxy_to_normalized_xywh(
            (bbx["xmin"], bbx["ymin"], bbx["xmax"], bbx["ymax"]), img_shape[:2], center=True
        )
        annotations.append(f"{bbx['class_id']} {x_center} {y_center} {w} {h}")

    with open(path, "w") as f:
        f.write("\n".join(annotations) + "\n")

def save_img(
    img: np.ndarray,
    data_dir: Path,
    fname: Path,
) -> Path:
    img_dir = data_dir / "images"
    os.makedirs(img_dir, exist_ok=True)

    path = img_dir / f"{str(fname)}.jpg"

    if path.exists():
        return path

    saveimg(img, path)

    return path


def saveimg(img, path):
    im = Image.fromarray(img)
    im.save(path, format="PNG")


def enumerate_fnames(source_data_dir: Path) -> list[Path]:
    l = []
    for image_name in os.listdir(source_data_dir / "images"):
        l.append(Path(image_name))
    return sorted(l)  # type: ignore
