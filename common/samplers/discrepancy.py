"""Count-discrepancy scoring: compares predicted-instance counts inside the geometric
overlap of two temporally-adjacent collected frames. Ported from
habitat_embodied_al/third_party/embodied-active-learning-od's uutils/discrepancy_utils.py
(get_overlapping_region/get_overlapping_items/list_count_difference), which keys off
AI2Thor's 8-directional (45deg) grid pose -- ExplorationEnv's action space is only
move_forward/turn_left/turn_right (common/agents/random_agent.py), no strafe and no
diagonal step distinct from a straight one, so the "move_left"/diagonal branches from the
original don't apply and are dropped. habitat's turn_angle (12-way here, see
common/config/hssd-hab/default.yaml) doesn't tile a square lattice the way AI2Thor's 45deg
grid does, so DiscretizedPose's idx_x/idx_z are a real-valued odometry accumulator advanced
from the action actually taken (see advance_pose), not a position snapped from world
coordinates -- exact only for classifying the single-action transition between two poses
compared by count_discrepancy, and only within the same collection round (see Candidate.
round_idx), since the accumulator resets every round.
"""
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from detectron2.structures import Instances  # type: ignore


@dataclass(frozen=True)
class DiscretizedPose:
    idx_x: float
    idx_z: float
    idx_yaw: int


def _yaw_dx_dz(idx_yaw: int, num_yaw: int) -> tuple:
    angle = 2 * math.pi * idx_yaw / num_yaw
    return math.sin(angle), math.cos(angle)


def advance_pose(pose: DiscretizedPose, action: str, moved: bool, num_yaw: int) -> DiscretizedPose:
    if action == "move_forward" and moved:
        dx, dz = _yaw_dx_dz(pose.idx_yaw, num_yaw)
        return DiscretizedPose(pose.idx_x + dx, pose.idx_z + dz, pose.idx_yaw)
    if action == "turn_left":
        return DiscretizedPose(pose.idx_x, pose.idx_z, (pose.idx_yaw - 1) % num_yaw)
    if action == "turn_right":
        return DiscretizedPose(pose.idx_x, pose.idx_z, (pose.idx_yaw + 1) % num_yaw)
    return pose


def _is_move_ahead(prev: DiscretizedPose, curr: DiscretizedPose) -> bool:
    return prev.idx_yaw == curr.idx_yaw and (prev.idx_x, prev.idx_z) != (curr.idx_x, curr.idx_z)


def _is_rotate_left(prev: DiscretizedPose, curr: DiscretizedPose, num_yaw: int) -> bool:
    return (prev.idx_x, prev.idx_z) == (curr.idx_x, curr.idx_z) and (curr.idx_yaw - prev.idx_yaw) % num_yaw == num_yaw - 1


def _is_rotate_right(prev: DiscretizedPose, curr: DiscretizedPose, num_yaw: int) -> bool:
    return (prev.idx_x, prev.idx_z) == (curr.idx_x, curr.idx_z) and (curr.idx_yaw - prev.idx_yaw) % num_yaw == 1


def is_neighbor(pose: DiscretizedPose, other: DiscretizedPose, num_yaw: int) -> bool:
    """True if `other` is reachable from `pose` via a single move_forward/turn_left/turn_right.
    Unlike _is_move_ahead (only ever safe on consecutive-in-time poses, since a single action
    separates them by construction), this checks the step distance explicitly -- candidates
    pulled from a pool can be any number of collinear steps apart."""
    if pose.idx_yaw == other.idx_yaw:
        dx, dz = other.idx_x - pose.idx_x, other.idx_z - pose.idx_z
        return math.isclose(dx * dx + dz * dz, 1.0, abs_tol=1e-6)
    return _is_rotate_left(pose, other, num_yaw) or _is_rotate_right(pose, other, num_yaw)


Region = tuple  # ((xmin, xmax), (ymin, ymax))


def get_overlapping_region(
    width: int, height: int, grid_size: float, prev_pose: DiscretizedPose, curr_pose: DiscretizedPose, num_yaw: int,
) -> tuple:
    margin_forward = 0.25
    margin_rotate = 0.4

    if _is_move_ahead(prev_pose, curr_pose):
        margin_step = int(grid_size * width * margin_forward)
        overlap_prev = ((margin_step, width - margin_step), (margin_step, height - margin_step))
        overlap_curr = ((0, width), (0, height))
    elif _is_rotate_left(prev_pose, curr_pose, num_yaw):
        overlap_prev = ((0, int(width * (1 - margin_rotate))), (0, height))
        overlap_curr = ((int(width * margin_rotate), width), (0, height))
    elif _is_rotate_right(prev_pose, curr_pose, num_yaw):
        overlap_prev = ((int(width * margin_rotate), width), (0, height))
        overlap_curr = ((0, int(width * (1 - margin_rotate))), (0, height))
    else:  # no-op transition (blocked move_forward, or prev == curr)
        overlap_prev = ((0, width), (0, height))
        overlap_curr = ((0, width), (0, height))

    return overlap_prev, overlap_curr


def is_small(area: float, min_area: float) -> bool:
    return area < 1.25 * min_area


def is_close_to_boundary(box: tuple, region: Region, width: int, height: int, threshold: float = 0.1) -> bool:
    (xmin, xmax), (ymin, ymax) = region
    x0, y0, x1, y1 = box
    xcenter, ycenter = (x0 + x1) / 2, (y0 + y1) / 2

    distance_to_boundary = min(
        max(xcenter - xmin, 0), max(xmax - xcenter, 0), max(ycenter - ymin, 0), max(ymax - ycenter, 0),
    )
    return distance_to_boundary < threshold * max(width, height)


def _items_in_region(instances: Instances, region: Region) -> list:
    """Returns (class_id, box, area) for detections whose overlap with `region` covers
    more than a quarter of their own area -- same fraction the original uses."""
    (xmin, xmax), (ymin, ymax) = region
    items = []
    boxes = instances.pred_boxes.tensor.numpy() if len(instances) else np.zeros((0, 4))
    classes = instances.pred_classes.numpy() if len(instances) else np.zeros((0,), dtype=int)

    for box, class_id in zip(boxes, classes):
        x0, y0, x1, y1 = box
        area = (x1 - x0) * (y1 - y0)
        intersection = max(0, min(xmax, x1) - max(xmin, x0)) * max(0, min(ymax, y1) - max(ymin, y0))
        if intersection > 0.25 * area:
            items.append((int(class_id), tuple(box.tolist()), float(area)))
    return items


def first_match(list1: list, list2: list) -> dict:
    """Arbitrary index-to-index pairing between two same-class item lists (order doesn't
    matter for a pure count comparison), up to the shorter list's length."""
    n = min(len(list1), len(list2))
    return {i: i for i in range(n)}


def list_count_difference(list1: list, list2: list, can_be_unmatched, normalize: bool = False) -> float:
    """list1/list2: (class_id, box, area) tuples. A matched pair (same class, paired 1:1)
    counts as one shared union element and no difference. An unmatched item counts as one
    union element and one difference, unless it can_be_unmatched (too small / at the crop
    boundary), in which case it's dropped from both counts entirely."""
    matched1: set = set()
    matched2: set = set()
    n_pairs = 0
    for class_id in {c for c, _, _ in list1} | {c for c, _, _ in list2}:
        objs1 = [i for i, x in enumerate(list1) if x[0] == class_id]
        objs2 = [i for i, x in enumerate(list2) if x[0] == class_id]
        for i1, i2 in first_match([list1[i] for i in objs1], [list2[i] for i in objs2]).items():
            matched1.add(objs1[i1])
            matched2.add(objs2[i2])
            n_pairs += 1

    count_difference = 0.0
    n_union = n_pairs
    for i, item in enumerate(list1):
        if i in matched1 or can_be_unmatched(item, True):
            continue
        count_difference += 1
        n_union += 1
    for i, item in enumerate(list2):
        if i in matched2 or can_be_unmatched(item, False):
            continue
        count_difference += 1
        n_union += 1

    if normalize:
        return count_difference / n_union if n_union > 0 else 0.0
    return count_difference


def count_discrepancy(
    prev_instances: Instances, curr_instances: Instances,
    prev_pose: DiscretizedPose, curr_pose: DiscretizedPose,
    width: int, height: int, grid_size: float, num_yaw: int, min_area: float,
    normalize: bool = False,
) -> float:
    region_prev, region_curr = get_overlapping_region(width, height, grid_size, prev_pose, curr_pose, num_yaw)
    prev_items = _items_in_region(prev_instances, region_prev)
    curr_items = _items_in_region(curr_instances, region_curr)

    def can_be_unmatched(item: tuple, from_prev: bool) -> bool:
        _class_id, box, area = item
        region = region_prev if from_prev else region_curr
        return is_small(area, min_area) or is_close_to_boundary(box, region, width, height)

    return list_count_difference(prev_items, curr_items, can_be_unmatched, normalize=normalize)
