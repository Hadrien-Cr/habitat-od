from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial.kdtree import KDTree
from transformers import CLIPModel, CLIPProcessor


def overlap_ratio_matrix(points_a: List[np.ndarray], points_b: List[np.ndarray], radius: float) -> np.ndarray:
    """(i, j) = fraction of points_a[i] lying within radius of some point of points_b[j] (concept-graphs' "overlap")."""
    trees_b = [KDTree(points) for points in points_b]
    return np.array([
        [(tree.query(points, k=1, distance_upper_bound=radius)[0] < radius).mean() for tree in trees_b]
        for points in points_a
    ]).reshape(len(points_a), len(points_b))


def visual_similarity_matrix(features_a: torch.Tensor, features_b: torch.Tensor) -> torch.Tensor:
    """(i, j) = cosine similarity between features_a[i] and features_b[j]."""
    return F.normalize(features_a, dim=-1) @ F.normalize(features_b, dim=-1).T


class ClipCropEncoder:
    """Hugging Face CLIP image features of padded bounding-box crops, as concept-graphs computes them."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cuda"):
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def encode_crops(self, rgb: np.ndarray, boxes: np.ndarray, padding: int = 20) -> torch.Tensor:
        """(N, D) normalized features of the crops of rgb around each xyxy box."""
        image = Image.fromarray(rgb)
        crops = [image.crop(_padded_box(box, padding, image.size)) for box in boxes]
        pixels = self.processor(images=crops, return_tensors="pt")["pixel_values"].to(self.device)
        return F.normalize(self.model.get_image_features(pixel_values=pixels), dim=-1).cpu()


def _padded_box(box: np.ndarray, padding: int, image_size: tuple) -> tuple:
    width, height = image_size
    x_min, y_min, x_max, y_max = box
    return max(x_min - padding, 0), max(y_min - padding, 0), min(x_max + padding, width), min(y_max + padding, height)
