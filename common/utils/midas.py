"""Monocular depth estimation with MiDaS (torch.hub isl-org/MiDaS), ported from home-robot's
home_robot/perception/midas.py."""
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

_HUB_REPO = "isl-org/MiDaS"
# DPT_BEiT_L_512 (home-robot's choice) crashes on timm>=0.9's renamed Block.drop_path
_MODEL_TYPE = "DPT_Large"
_MIN_DISPARITY = 1e-6


class Midas:
    """MiDaS relative inverse depth, scaled and shifted onto a reference metric depth."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = torch.hub.load(_HUB_REPO, _MODEL_TYPE, trust_repo=True).to(device).eval()
        self.transform = torch.hub.load(_HUB_REPO, "transforms", trust_repo=True).dpt_transform

    def relative_disparity(self, rgb: np.ndarray) -> np.ndarray:
        """MiDaS' affine-invariant inverse depth of a uint8 RGB image, at the image's resolution."""
        with torch.no_grad():
            prediction = self.model(self.transform(rgb).to(self.device))
        prediction = F.interpolate(prediction.unsqueeze(1), size=rgb.shape[:2], mode="bicubic", align_corners=False)
        return prediction.squeeze().cpu().numpy()

    def depth_estimate(self, rgb: np.ndarray, depth: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Metric depth of rgb, with the mse and mean absolute error of its fit to depth's valid (> 0) pixels."""
        disparity = self.relative_disparity(rgb)
        valid = depth > 0
        scale, shift = fit_scale_shift(disparity[valid], 1.0 / depth[valid])
        adjusted = 1.0 / np.maximum(scale * disparity + shift, _MIN_DISPARITY)
        errors = adjusted[valid] - depth[valid]
        return adjusted, float((errors ** 2).mean()), float(np.abs(errors).mean())


def fit_scale_shift(prediction: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
    """Least-squares (scale, shift) minimizing |scale * prediction + shift - target|^2."""
    design = np.stack([prediction, np.ones_like(prediction)], axis=1)
    (scale, shift), *_ = np.linalg.lstsq(design, target, rcond=None)
    return float(scale), float(shift)
