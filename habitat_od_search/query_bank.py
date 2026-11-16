"""The search agent's queries: a query name joins at most one text part ("text_cat", the episode's main category; or
"text_full", its whole fine-grained object_category) and at most one bank part with "+" -- "gt_views" (Detic region
embeddings of simulator views of the episode's own goal objects, every other object removed) or "web" (of Bing images
of its fine-grained string), both saved by build_query_banks.py under QUERY_BANKS_DIR/<scene_id>/<category>/.
"""
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from common.vision.detic import Vocabulary
from habitat_od_search.hssd_rare_dataset import main_category
from habitat_od_search.search_agent import TargetQuery

QUERY_BANKS_DIR = Path("habitat_od_search") / "logs" / "query_banks"
TEXT_QUERIES = ("text_cat", "text_full")
BANK_FILES = {"gt_views": "gt_features.npz", "web": "scraped_features.npz"}


def query_parts(query: str) -> Tuple[Optional[str], Optional[str]]:
    """(text part, bank part) of query, either None when absent."""
    parts = query.split("+")
    texts = [part for part in parts if part in TEXT_QUERIES]
    banks = [part for part in parts if part in BANK_FILES]
    assert len(texts) <= 1 and len(banks) <= 1 and len(texts) + len(banks) == len(parts), f"invalid query {query!r}"
    return (texts[0] if texts else None, banks[0] if banks else None)


def bank_dir(scene_id: str, category: str) -> Path:
    return QUERY_BANKS_DIR / scene_id / category.replace("/", "_")


def load_bank(scene_id: str, category: str, bank: str) -> torch.Tensor:
    """(K, D) region embeddings of category's bank in scene_id."""
    path = bank_dir(scene_id, category) / BANK_FILES[bank]
    assert path.exists(), f"missing query bank {path}: run habitat_od_search/build_query_banks.py first"
    return torch.from_numpy(np.load(path)["features"])


def text_embedding(text: str) -> torch.Tensor:
    return Vocabulary(f"search_{text.replace('/', '_')}", (text,)).clip_embeddings()[:, 0]


def target_query(query: str, scene_id: str, category: str) -> TargetQuery:
    text, bank = query_parts(query)
    text_string = None if text is None else (main_category(category) if text == "text_cat" else category)
    return TargetQuery(
        None if text_string is None else text_embedding(text_string),
        None if bank is None else load_bank(scene_id, category, bank),
    )
