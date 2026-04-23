import os
import cv2
from pathlib import Path
import numpy as np
from PIL import Image

def xy_to_normalized_xy(xy: tuple[int, int], size: tuple[int,int]) -> tuple[float, float]:
    x, y = xy
    img_width, img_height = size
    x /= img_width
    y /= img_height
    return x, y

def normalized_xy_to_xy(normalized_xy: tuple[float, float], size: tuple[int,int]) -> tuple[int, int]:
    x, y = normalized_xy
    img_width, img_height = size
    x *= img_width
    y *= img_height
    return int(round(x)), int(round(y))

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


def load_bounding_boxes(
    path: Path, img_size: tuple[int,int]
) -> list:
    w, h = img_size

    bounding_boxes = []

    with open(path) as fa:
        for line in fa.readlines():
            annot = line.strip().split()
            assert len(annot) == 5

            class_id = int(annot[0])
            x,y,w,h = list(map(float, annot[1:]))
            xyxy = normalized_xywh_to_xyxy(
                (x,y,w,h), img_size, center=True
            )
            bounding_boxes.append((class_id, xyxy))
        
    return bounding_boxes

def load_segmentation_masks(
    path: Path, img_size: tuple[int,int]
) -> list[np.ndarray]:
    w, h = img_size

    binary_masks = []

    with open(path) as fa:
        for line in fa.readlines():
            annot = line.strip().split()

            binary_mask = np.zeros((h, w), dtype=np.uint8)
            class_id = int(annot[0])
            _mask_polygon = list(map(float, annot[1:]))
            
            mask_polygon = []
            
            for i in range(0, len(_mask_polygon), 2):
                x, y = _mask_polygon[i], _mask_polygon[i+1]
                x, y = normalized_xy_to_xy((x, y), img_size)
                mask_polygon.append(x)
                mask_polygon.append(y)

            cv2.fillPoly(binary_mask, pts=[np.array(mask_polygon, dtype=np.int32).reshape(-1, 1, 2)], color=(class_id, class_id, class_id))
            binary_masks.append((class_id, binary_mask))

    return binary_masks


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


def pose2fname(prefix: str, pose: tuple) -> Path:
    """Get the filename corresponding to the given pose."""
    (x,y,z), (r,p,yaw) = pose
    fname = Path(f"{prefix}_x_{x:.2f}_z_{z:.2f}_yaw_{yaw:.2f}")
    return fname



def save_ground_truth(
    ground_truth: list[tuple[str, dict]],
    data_dir: Path,
    fname: Path,
    img_shape: tuple[int, int, int],
    class_mapping: dict[str, int],
    save_bounding_boxes: bool,
    save_segmentations_masks: bool
) -> None:
    label_dir = data_dir / "labels"
    os.makedirs(label_dir, exist_ok=True)

    bounding_box_path = label_dir / (fname.stem + ".txt")
    segmentation_masks_path = label_dir / (fname.stem + ".txt")

    # if bounding_box_path.exists():
    #     return

    bounding_box_annotations = []
    segmentation_mask_annotations = []

    for class_name, object_detection_info in ground_truth:

        if class_name not in class_mapping:
            raise ValueError

        class_id = class_mapping[class_name]

        if save_bounding_boxes:
            xmin, ymin, xmax, ymax = object_detection_info["bounding_box"]
            x_center, y_center, w, h = xyxy_to_normalized_xywh(
                (xmin, ymin, xmax, ymax), img_shape[:2], center=True
            )
            bounding_box_annotations.append(f"{class_id} {x_center} {y_center} {w} {h}")

        if save_segmentations_masks:
            mask_polygon = object_detection_info["mask_polygon"]
            mask_polygon_str = ""
            for x, y in mask_polygon:
                normx, normy = xy_to_normalized_xy((x, y), img_shape[:2])
                mask_polygon_str += f" {normx} {normy}"
            segmentation_mask_annotations.append(f"{class_id}{mask_polygon_str}")

    if save_bounding_boxes:
        with open(bounding_box_path, "w") as f:
            f.write("\n".join(bounding_box_annotations) + "\n")

    if save_segmentations_masks:
        with open(segmentation_masks_path, "w") as f:
            f.write("\n".join(segmentation_mask_annotations) + "\n")

def save_img(
    img: np.ndarray,
    data_dir: Path,
    fname: Path,
) -> Path:
    img_dir = data_dir / "images"
    os.makedirs(img_dir, exist_ok=True)

    path = img_dir / f"{str(fname)}.jpg"

    # if path.exists():
    #     return path

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
