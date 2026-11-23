"""SigLIP text and patch encoder (IMPRINT's static-baseline default backbone): unit embeddings in one joint space."""
from typing import List

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

SIGLIP_MODEL = "google/siglip-base-patch16-256"


class SiglipEncoder:
    def __init__(self, model_name: str = SIGLIP_MODEL, device: str = "cuda"):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device
        vision_config = self.model.config.vision_config
        self.dim = vision_config.hidden_size
        self.grid = vision_config.image_size // vision_config.patch_size

    @torch.no_grad()
    def patch_embeddings(self, images: List[np.ndarray]) -> torch.Tensor:
        """(N, grid * grid, D) unit embeddings of RGB uint8 images' patches, row-major, in the text embeddings' space:
        each patch token goes through the pooling head's value, output and MLP layers as if the head attended to it
        alone (MaskCLIP's dense-feature trick)."""
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        tokens = self.model.vision_model(**inputs).last_hidden_state
        head = self.model.vision_model.head
        values = torch.nn.functional.linear(tokens, head.attention.in_proj_weight[2 * self.dim:], head.attention.in_proj_bias[2 * self.dim:])
        attended = head.attention.out_proj(values)
        return torch.nn.functional.normalize(attended + head.mlp(head.layernorm(attended)), dim=-1).cpu()

    @torch.no_grad()
    def text_embeddings(self, texts: List[str]) -> torch.Tensor:
        """(N, D) unit embeddings of texts, padded to max_length as SigLIP was trained."""
        inputs = self.processor(text=texts, padding="max_length", truncation=True, return_tensors="pt").to(self.device)
        return torch.nn.functional.normalize(self.model.get_text_features(**inputs), dim=-1).cpu()
