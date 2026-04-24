import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
from PIL import Image, ImageDraw

def draw_dotted_line(img,pt1,pt2,color,thickness=1,gap=20):
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts= []
    for i in  np.arange(0,dist,gap):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)

    for p in pts:
        cv2.circle(img,p,thickness,color,-1)


def draw_dotted_poly(img,pts,color,thickness=1):
    s=pts[0]
    e=pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s=e
        e=p
        draw_dotted_line(img,s,e,color,thickness)


def draw_dotted_rect(img,x1,y1,x2,y2,color,thickness=1):
    pts = [(x1,y1),(x2,y1),(x2,y2),(x1,y2)] 
    draw_dotted_poly(img,pts,color,thickness)


def plot_pred_bounding_box(
    img, xyxy, score, label, color, line_thickness=1, offset=0
) -> np.ndarray:
    tl = line_thickness or round(
        0.002 * (img.shape[0] + img.shape[1]) / 2
    )  # line/font thickness
    x1, y1, x2, y2 = map(int, map(round, xyxy))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=tl, lineType=cv2.LINE_AA)
    tf = 1
    t_size = cv2.getTextSize(
        label + f" {score:.2f}", 0, fontScale=tl / 4, thickness=tf
    )[0]
    cv2.rectangle(
        img,
        (x1, y1 - offset * t_size[1]),
        (x1 + t_size[0], y1 - (offset + 1) * t_size[1] - 3),
        color,
        -1,
        cv2.LINE_AA,
    )  # filled
    cv2.putText(
        img,
        label + f" {score:.2f}",
        (x1, y1 - offset * t_size[1] - 2),
        0,
        tl / 4,
        [255, 255, 255],
        thickness=tf,
        lineType=cv2.LINE_AA,
    )
    return img


def plot_label_bounding_box(img, xyxy, label, color, line_thickness=1, alpha=0.3) -> np.ndarray:
    tl = line_thickness or round(
        0.002 * (img.shape[0] + img.shape[1]) / 2
    )  # line/font thickness

    x1, y1, x2, y2 = map(int, map(round, xyxy))
    tf = 1
    t_size = cv2.getTextSize("GT: " + label, 0, fontScale=tl / 4, thickness=tf)[0]

    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    cv2.rectangle(
        img, (x1, y2), (x1 + t_size[0], y2 - t_size[1] + 3), color, -1, cv2.LINE_AA
    )  # filled
    cv2.putText(
        img,
        "GT: " + label ,
        (x1, y2 + 2),
        0,
        tl / 4,
        [255, 255, 255],
        thickness=tf,
        lineType=cv2.LINE_AA,
    )
    return img

def plot_pred_mask(img, mask, label, color, line_thickness=1, alpha=0.3, draw_dotted_bbx:  bool =  False) -> np.ndarray:
    tl = line_thickness or round(
        0.002 * (img.shape[0] + img.shape[1]) / 2
    )  # line/font thickness

    # Apply semi-transparent colored mask overlay
    overlay = img.copy()
    overlay[mask.astype(bool)] = color
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # Draw mask contour
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(img, contours, -1, color, thickness=tl, lineType=cv2.LINE_AA)

    # Draw label box + text at top-left of the bounding rect of the mask
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    tf = 1
    t_size = cv2.getTextSize(label, 0, fontScale=tl / 4, thickness=tf)[0]
    cv2.rectangle(
        img,
        (x, y),
        (x + t_size[0], y - t_size[1] - 3),
        color,
        -1,
        cv2.LINE_AA,
    )  # filled

    if draw_dotted_bbx: draw_dotted_rect(img, x, y, x + w, y + h, color, thickness=tl)
    
    cv2.putText(
        img,
        label,
        (x, y - 2),
        0,
        tl / 4,
        [255, 255, 255],
        thickness=tf,
        lineType=cv2.LINE_AA,
    )

    return img


from PIL import Image, ImageDraw, ImageFont
import numpy as np
from scipy import ndimage


def plot_semantic_2d_map(
    sem,
    colors: dict[int, str],
    class_mapping: dict[str, int],
    scale=8
) -> Image.Image:
    # sem: (w, h, 1 + nc)
    num_sem_classes = sem.shape[-1] - 1

    colored = np.zeros((*sem.shape[:2], 3), dtype=np.uint8)

    white_mask = (sem[:, :, 0] == 0)
    colored[white_mask] = [255, 255, 255]
    colored[~white_mask] = [0, 0, 0]

    for c in range(num_sem_classes):
        object_mask = (sem[:, :, c+1] == 1)
        colored[object_mask] = colors[c]

    # Convert to image and upscale x4
    img = Image.fromarray(colored)
    img = img.resize(
        (img.width * scale, img.height * scale),
        resample=Image.NEAREST  # keeps grid cells crisp
    )

    draw = ImageDraw.Draw(img)

    inv_class_mapping = {i: c for c, i in class_mapping.items()}

    for c in range(num_sem_classes):
        object_mask = (sem[:, :, c+1] == 1)

        labeled_mask, num = ndimage.label(object_mask)

        for i in range(1, num + 1):
            region = labeled_mask == i
            coords = np.column_stack(np.where(region))

            if len(coords) < 30:
                continue

            y, x = coords.mean(axis=0)

            x *= scale
            y *= scale

            draw.text(
                (x, y),
                inv_class_mapping[c],
                fill=(255,255,255),
                font_size=20,
                anchor="mm"  # center text on region centroid
            )
    return img

def plot_mask(mask) -> Image.Image:
    colored = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
    colored[mask] = [255, 255, 255]
    return Image.fromarray(colored)


def plot_label_mask(img, mask, label, color, line_thickness=1, alpha=0.3, draw_dotted_bbx: bool = False) -> np.ndarray:
    tl = line_thickness or round(
        0.002 * (img.shape[0] + img.shape[1]) / 2
    )  # line/font thickness

    # Apply semi-transparent colored mask overlay
    overlay = img.copy()
    overlay[mask.astype(bool)] = color
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # Draw mask contour
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(img, contours, -1, color, thickness=tl, lineType=cv2.LINE_AA)

    # Draw label box + text at bottom-left of the bounding rect of the mask
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    tf = 1
    t_size = cv2.getTextSize("GT: " + label, 0, fontScale=tl / 4, thickness=tf)[0]

    if draw_dotted_bbx: draw_dotted_rect(img, x, y, x + w, y + h, color, thickness=tl)
    
    cv2.rectangle(
        img,
        (x, y + h),
        (x + t_size[0], y + h - t_size[1] + 3),
        color,
        -1,
        cv2.LINE_AA,
    )  # filled
    cv2.putText(
        img,
        "GT: " + label,
        (x, y + h + 2),
        0,
        tl / 4,
        [255, 255, 255],
        thickness=tf,
        lineType=cv2.LINE_AA,
    )

    return img

def plot_segmentation(
    img,
    prediction: list[dict],
    labels: list[dict],
    colors,
    line_thickness=2,
):
    _img = img.copy()
    if isinstance(img, Image.Image):
        _img = np.array(img)

    # Draw prediction
    for pred in prediction:
        _img = plot_pred_mask(
            _img,
            pred["mask"],
            pred["class_name"],
            colors[pred["class_name"]],
            line_thickness=line_thickness,
        )

    # Draw labels
    for gt in labels:
        _img = plot_label_mask(
            _img,
            gt["mask"],
            gt["class_name"],
            colors[gt["class_name"]],
            line_thickness=line_thickness,
        )

    return _img


def plot_object_detection(
    img,
    prediction: list[dict],
    labels: list[dict],
    colors,
    line_thickness=2,
):

    _img = img.copy()
    if isinstance(img, Image.Image):
        _img = np.array(img)

    # Draw prediction
    for pred in prediction:
        xyxy = pred["bounding_box"]
        _img = plot_pred_bounding_box(
            _img,
            xyxy,
            pred["confidence"],
            pred["class_name"],
            colors[pred["class_name"]],
            line_thickness=line_thickness,
        )

    # Draw labels
    for gt in labels:
        xyxy = gt["bounding_box"]
        _img = plot_label_bounding_box(
            _img,
            xyxy,
            gt["class_name"],
            colors[gt["class_name"]],
            line_thickness=line_thickness,
        )

    return _img


def make_mosaic(
    list_fnames_images: list[tuple[str, np.ndarray]],
    target_height: int = 2000,
    sort: bool = False,
    N_cols: int = 4
) -> np.ndarray:
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

    # 2. Resize all images to the smallest dimension found for uniform tiles
    # This step is crucial for np.hstack/np.vstack to work.
    min_h = min(img.shape[0] for img in processed_images)
    min_w = min(img.shape[1] for img in processed_images)
    tile_size = (min_w, min_h)  # (width, height) for cv2.resize

    resized_images = [cv2.resize(img, tile_size) for img in processed_images]

    # 3. Create the nx4 mosaic
    rows = []
    for i in range((n + N_cols - 1) // N_cols):
        # Stack 4 images horizontally (cols)
        start_idx = i * N_cols
        end_idx = (i + 1) * N_cols
        row_of_images = resized_images[start_idx:end_idx]
        if len(row_of_images) == 0:
            continue
        elif len(row_of_images) == N_cols:
            rows.append(np.hstack(row_of_images))
        else:
            # If not enough images to fill the last row, pad with black images
            n_missing = N_cols - len(row_of_images)
            black_image = np.zeros_like(resized_images[0])
            row_of_images.extend([black_image] * n_missing)
            rows.append(np.hstack(row_of_images))

    final_mosaic = np.vstack(rows)

    downscale_factor = target_height / final_mosaic.shape[0]
    final_mosaic = cv2.resize(
        final_mosaic,
        (
            int(final_mosaic.shape[1] * downscale_factor),
            int(final_mosaic.shape[0] * downscale_factor),
        ),
    )
    return final_mosaic