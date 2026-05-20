import sys
import torch
from pathlib import Path
import numpy as np

DETIC_ROOT = str(Path(__file__).parent / "../../third_party/Detic/")

sys.path.insert(
    0, DETIC_ROOT
)
sys.path.insert(
    0, str(Path(DETIC_ROOT) / "third_party/CenterNet2/")
)

from third_party.Detic.detic.modeling.text.text_encoder import (  # noqa:E402
    build_text_encoder,
)

def get_clip_embeddings(vocabulary: list[str], prompt: str = "a ") -> torch.Tensor:
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()
    texts = [prompt + x for x in vocabulary]
    emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    return emb

def save_clip_embeddings(embeddings: torch.Tensor, path: Path):
    np.save(path, embeddings.cpu().numpy().astype(np.float32))

def load_clip_embeddings(path: Path) -> torch.Tensor:
    return torch.tensor(np.load(path)).to(torch.float32)
