import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image, ImageDraw
import cv2
from detectron2.utils.visualizer import Visualizer as DetVisualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.data import Metadata

def plot_semantic_2d_map(
    bg_grid,
    sem_grid,
    colors: dict[int, tuple[int,int,int]],
    classes: list[str],
    scale=8
) -> Image.Image:
    # sem: (h, w, nc)
    num_sem_classes = sem_grid.shape[-1]

    colored = np.zeros((*sem_grid.shape[:2], 3), dtype=np.uint8)

    colored[bg_grid == 0] = [255, 255, 255]
    colored[bg_grid == 1] = [0, 0, 0]

    for c in range(num_sem_classes):
        object_mask = (sem_grid[:, :, c] == 1)
        colored[object_mask] = colors[c]

    # Convert to image and upscale x4
    img = Image.fromarray(colored)
    img = img.resize(
        (img.width * scale, img.height * scale),
        resample=Image.NEAREST  # type: ignore
    )

    draw = ImageDraw.Draw(img)

    for c in range(num_sem_classes):
        object_mask = (sem_grid[:, :, c] == 1)

        labeled_mask, num = ndimage.label(object_mask)

        for i in range(1, num + 1):
            region = labeled_mask == i
            coords = np.column_stack(np.where(region))

            if len(coords) < 10:
                continue

            y, x = coords.mean(axis=0)

            x *= scale
            y *= scale

            draw.text(
                (x, y),
                classes[c],
                fill=(255,255,255),
                font_size=20,
                anchor="mm"  # center text on region centroid
            )
    return img

def draw_line(
    start: tuple[int, int],
    end: tuple[int, int],
    mat: np.ndarray,
    steps: int = 25,
    w: int = 1,
) -> np.ndarray:
    for i in range(steps + 1):
        x = int(np.rint(start[0] + (end[0] - start[0]) * i / steps))
        y = int(np.rint(start[1] + (end[1] - start[1]) * i / steps))
        mat[x - w : x + w, y - w : y + w] = 1
    return mat

def get_contour_points(
    pos: tuple[float, float, float],
    origin: tuple[float, float],
    size: int = 20,
) -> np.ndarray:
    x, y, o = pos
    pt1 = (int(x) + origin[0], int(y) + origin[1])
    pt2 = (
        int(x + size / 1.5 * np.cos(o + np.pi * 4 / 3)) + origin[0],
        int(y + size / 1.5 * np.sin(o + np.pi * 4 / 3)) + origin[1],
    )
    pt3 = (int(x + size * np.cos(o)) + origin[0], int(y + size * np.sin(o)) + origin[1])
    pt4 = (
        int(x + size / 1.5 * np.cos(o - np.pi * 4 / 3)) + origin[0],
        int(y + size / 1.5 * np.sin(o - np.pi * 4 / 3)) + origin[1],
    )

    return np.array([pt1, pt2, pt3, pt4])

def plot_array(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr)
    

def plot_mask(mask) -> Image.Image:
    colored = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
    colored[mask == 1] = [255, 255, 255]
    return Image.fromarray(colored)

def plot_segmentation_pred(rgb: np.ndarray, instances, classes: list[str], colors: list[tuple[float, float, float]], scale: float = 1.0) -> Image.Image:
    from detectron2.data import Metadata
    metadata = Metadata()
    metadata.set(thing_classes=classes)
    metadata.set(thing_colors=colors)

    det_visualizer = DetVisualizer(
        rgb,
        metadata=metadata,
        scale=scale,
        instance_mode=ColorMode.SEGMENTATION
    )
    vis_img = det_visualizer.draw_instance_predictions(instances.to("cpu"), jittering=False)
    result = vis_img.get_image()
    im = Image.fromarray(result)
    return im


def plot_segmentation_gt(rgb: np.ndarray, gt_instances, classes: list[str], colors: list[tuple[float, float, float]], scale: float = 1.0) -> Image.Image:
    from detectron2.data import Metadata
    metadata = Metadata()
    metadata.set(thing_classes=classes)
    metadata.set(thing_colors=colors)

    det_visualizer = DetVisualizer(
        rgb,
        metadata=metadata,
        scale=scale,
        instance_mode=ColorMode.SEGMENTATION
    )

    for box, class_id in zip(
        gt_instances.gt_boxes.tensor.cpu().numpy(),
        gt_instances.gt_classes.cpu().numpy()
    ):
        x1, y1, x2, y2 = box
        height_ratio = (y2 - y1) / rgb.shape[0]
        font_size = (
            np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2) * 0.5 * det_visualizer._default_font_size
        )
        det_visualizer.draw_text(
            classes[class_id],
            font_size=font_size,
            position=(x1, y2 - 10 * height_ratio),
            horizontal_alignment="left",
            color="red"
        )

    vis_img = det_visualizer.overlay_instances(
        boxes=gt_instances.gt_boxes.tensor.cpu().numpy(),
        masks=None,
        labels=None,
        assigned_colors=[(1.0, 0.0, 0.0)] * len(gt_instances),
    )
    result = vis_img.get_image()
    im = Image.fromarray(result)
    return im


def plot_segmentation_gt_and_pred(rgb: np.ndarray, pred_instances, gt_instances, classes: list[str], colors: list[tuple[float, float, float]], scale: float = 0.5) -> Image.Image:
    if pred_instances is not None:
        im = plot_segmentation_pred(rgb, pred_instances, classes, colors, scale)
        rgb = np.array(im)

    if gt_instances is not None:
        im = plot_segmentation_gt(rgb, gt_instances, ["gt_" + c for c in classes], [(1.0, 0.0, 0.0) for _ in classes], scale=1.0)
        rgb = np.array(im)

    return im

def make_mosaic(
    list_fnames_images: list[tuple[str, np.ndarray]],
    target_size: int = 1_000_000,
    N_cols: int = 4
) -> Image.Image:
    n =  len(list_fnames_images)
    processed_images = []

    for i, (filename, img) in enumerate(list_fnames_images):
        # add text overlay with filename
        cv2.putText(
            img,
            filename,
            (5, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        processed_images.append(img)


    # 3. Create the nx4 mosaic
    rows = []
    for i in range((n + N_cols - 1) // N_cols):
        # Stack 4 images horizontally (cols)
        start_idx = i * N_cols
        end_idx = (i + 1) * N_cols
        row_of_images = processed_images[start_idx:end_idx]
        if len(row_of_images) == 0:
            continue
        elif len(row_of_images) == N_cols:
            rows.append(np.hstack(row_of_images))
        else:
            # If not enough images to fill the last row, pad with black images
            n_missing = N_cols - len(row_of_images)
            black_image = np.zeros_like(processed_images[0])
            row_of_images.extend([black_image] * n_missing)
            rows.append(np.hstack(row_of_images))

    final_mosaic = np.vstack(rows)

    downscale_factor = np.ceil(np.sqrt(target_size / (final_mosaic.shape[0] * final_mosaic.shape[1])))
    final_mosaic = cv2.resize(
        final_mosaic,
        (
            int(final_mosaic.shape[1] * downscale_factor),
            int(final_mosaic.shape[0] * downscale_factor),
        ),
    )
    return Image.fromarray(final_mosaic)