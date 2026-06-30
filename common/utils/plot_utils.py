from matplotlib.pyplot import text
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
    classes: dict[int, str],
    meters_per_grid_pixel: float = 0.01,
    scale=1.0
) -> Image.Image:
    # sem: (h, w, nc)
    num_sem_classes = sem_grid.shape[-1]

    colored = np.zeros((*sem_grid.shape[:2], 3), dtype=np.uint8)
    colored[bg_grid == 0] = [235, 235, 235]
    colored[bg_grid == 1] = [30, 30, 30]
    areas_m = np.zeros(num_sem_classes, dtype=np.float32)

    for c in range(num_sem_classes):
        object_mask = (sem_grid[:, :, c] == 1)
        areas_m[c] = np.sum(object_mask) * meters_per_grid_pixel ** 2

    for c in sorted(range(num_sem_classes), key=lambda x: areas_m[x], reverse=True):
        object_mask = (sem_grid[:, :, c] == 1)
        if not np.any(object_mask):
            continue
        
        if c not in colors:
            colored[object_mask] = [50 + 50*areas_m[c], 50 + 50*areas_m[c], 50 + 50*areas_m[c]]
        else:
            colored[object_mask] = colors[c]

    img = Image.fromarray(colored)
    img = img.resize(
        (img.width * scale, img.height * scale),
        resample=Image.NEAREST  # type: ignore
    )
    draw = ImageDraw.Draw(img)
    
    for c in sorted(range(num_sem_classes), key=lambda x: areas_m[x], reverse=True):
        object_mask = (sem_grid[:, :, c] == 1)
        
        if not np.any(object_mask):
            continue

        object_mask = (sem_grid[:, :, c] == 1)

        labeled_mask, num = ndimage.label(object_mask)

        for i in range(1, num + 1):
            region = labeled_mask == i
            coords = np.column_stack(np.where(region))

            if areas_m[c] < 0.05:
                continue

            y, x = coords.mean(axis=0)

            x *= scale
            y *= scale

            draw.text(
                (x, y),
                classes[c],
                fill=(255,255,255),
                font_size = 1.5 * np.clip(np.sqrt(areas_m[c]), 0.1, 2) * scale,
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

def plot_segmentation_pred(rgb: np.ndarray, pred_instances, classes: list[str], colors: list[tuple[float, float, float]], scale: float = 1.0) -> Image.Image:
    det_visualizer = DetVisualizer(
        rgb,
        scale=scale,
        instance_mode=ColorMode.SEGMENTATION,
        font_size_scale=0.8
    )

    for i, (box, class_id, score) in enumerate(zip(
        pred_instances.pred_boxes.tensor.cpu().numpy(),
        pred_instances.pred_classes.cpu().numpy(),
        pred_instances.scores.cpu().numpy()
    )):
        x1, y1, x2, y2 = box
        height_ratio = (y2 - y1) / rgb.shape[0]
        font_size = (
            np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2) * 0.5 * det_visualizer._default_font_size
        )
        det_visualizer.draw_text(
            f"{i}: pt={classes[class_id]} {round(100*score)}%",
            font_size=font_size,
            position=(x1, y1),
            horizontal_alignment="left",
            color="#{:02x}{:02x}{:02x}".format(*colors[class_id])
        )

    vis_img = det_visualizer.overlay_instances(
        boxes=pred_instances.pred_boxes.tensor.cpu().numpy(),
        masks=pred_instances.pred_masks.cpu().numpy() if pred_instances.pred_masks is not None else None,
        labels=None,
        assigned_colors=[(x/255, y/255, z/255) for class_id in pred_instances.pred_classes.cpu().numpy() for x, y, z in [colors[class_id]]],
    )
    result = vis_img.get_image()
    im = Image.fromarray(result)
    return im


def plot_segmentation_gt(rgb: np.ndarray, gt_instances, classes: list[str], colors: list[tuple[float, float, float]], scale: float = 1.0) -> Image.Image:
    det_visualizer = DetVisualizer(
        rgb,
        scale=scale,
        instance_mode=ColorMode.SEGMENTATION,
        font_size_scale=0.8
    )

    for i, (box, class_id) in enumerate(zip(
        gt_instances.gt_boxes.tensor.cpu().numpy(),
        gt_instances.gt_classes.cpu().numpy()
    )):
        x1, y1, x2, y2 = box
        height_ratio = (y2 - y1) / rgb.shape[0]
        font_size = (
            np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2) * 0.5 * det_visualizer._default_font_size
        )
        det_visualizer.draw_text(
            f"{i}: gt={classes[class_id]}",
            font_size=font_size,
            position=(x1, y2 - 1.5 * font_size),
            horizontal_alignment="left",
            color="#{:02x}{:02x}{:02x}".format(*colors[class_id])
        )

    vis_img = det_visualizer.overlay_instances(
        boxes=gt_instances.gt_boxes.tensor.cpu().numpy(),
        masks=None,
        labels=None,
        assigned_colors=[(x/255, y/255, z/255) for class_id in gt_instances.gt_classes.cpu().numpy() for x, y, z in [colors[class_id]]],
        alpha=0.5
    )
    result = vis_img.get_image()
    im = Image.fromarray(result)
    return im


def plot_segmentation(
    rgb: np.ndarray, 
    pred_instances, 
    gt_instances, 
    classes: list[str], 
    colors: list[tuple[float, float, float]], 
    scale: float = 0.5,
    title: str = ""
) -> Image.Image:
    
    if gt_instances is not None:
        im = plot_segmentation_gt(rgb, gt_instances, classes, [(255, 0, 0) for _ in classes], scale=scale)
        rgb = np.array(im)
        scale = 1.0 


    if pred_instances is not None:
        im = plot_segmentation_pred(rgb, pred_instances, classes, colors, scale)
        rgb = np.array(im)


    if title:
        fontscale = 0.5
        cv2.putText(
            rgb,
            title,
            (int((rgb.shape[1] / 2)), int(fontscale * 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontscale,
            (255, 255, 255),
            thickness=2,
        )
        im = Image.fromarray(rgb)
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