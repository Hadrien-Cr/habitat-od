import numpy as np
import matplotlib.pyplot as plt
import cv2

import cv2
import numpy as np
from PIL import Image

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

def plot_pred_mask(img, mask, label, color, line_thickness=1, alpha=0.3) -> np.ndarray:
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

    draw_dotted_rect(img, x, y, x + w, y + h, color, thickness=tl)
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


def plot_label_mask(img, mask, label, color, line_thickness=1, alpha=0.3) -> np.ndarray:
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

    draw_dotted_rect(img, x, y, x + w, y + h, color, thickness=tl)
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
    prediction: list,
    labels: list,
    classes,
    colors,
    line_thickness=2,
):
    _img = img.copy()
    if isinstance(img, Image.Image):
        _img = np.array(img)

    # Draw prediction
    for class_id, mask, confidence in prediction:
        _img = plot_pred_mask(
            _img,
            mask,
            classes[class_id],
            colors[class_id],
            line_thickness=line_thickness,
        )

    # Draw labels
    for class_id, mask in labels:
        _img = plot_label_mask(
            _img,
            mask,
            classes[class_id],
            colors[class_id],
            line_thickness=line_thickness,
        )

    return _img


def plot_object_detection(
    img,
    prediction: list,
    labels: list,
    classes,
    colors,
    line_thickness=2,
):

    _img = img.copy()
    if isinstance(img, Image.Image):
        _img = np.array(img)

    # Draw prediction
    for class_id, (xmin, ymin, xmax, ymax), confidence in prediction:
        xyxy = (xmin, ymin, xmax, ymax)
        _img = plot_pred_bounding_box(
            _img,
            xyxy,
            confidence,
            classes[class_id],
            colors[class_id],
            line_thickness=line_thickness,
        )

    # Draw labels
    for class_id, (xmin, ymin, xmax, ymax) in labels:
        xyxy = (xmin, ymin, xmax, ymax)
        _img = plot_label_bounding_box(
            _img,
            xyxy,
            classes[class_id],
            colors[class_id],
            line_thickness=line_thickness,
        )

    return _img
