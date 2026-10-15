import os
import sys
from pathlib import Path
from typing import Sequence

from detectron2.config import CfgNode, get_cfg  # type: ignore
from detectron2.data import DatasetCatalog, MetadataCatalog  # type: ignore
from detectron2.data.datasets.builtin import _PREDEFINED_SPLITS_LVIS  # type: ignore
from detectron2.data.datasets.lvis import get_lvis_instances_meta, register_lvis_instances  # type: ignore

DETIC_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "Detic"
sys.path[:0] = [str(DETIC_ROOT), str(DETIC_ROOT / "third_party" / "CenterNet2")]

from centernet.config import add_centernet_config  # type: ignore  # noqa: E402
from detic.config import add_detic_config  # type: ignore  # noqa: E402


def _detic_path(path: str) -> str:
    """path resolved against DETIC_ROOT when it names a file there, else as-is (URLs, 'rand', absolute paths)."""
    return str(DETIC_ROOT / path) if path and (DETIC_ROOT / path).exists() else path


def detic_cfg(config_file: str, opts: Sequence[str] = ()) -> CfgNode:
    """Detic config_file merged with opts, its weights and metadata paths resolved from DETIC_ROOT."""
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(list(opts))
    cfg.MODEL.WEIGHTS = _detic_path(cfg.MODEL.WEIGHTS)
    cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = _detic_path(cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH)
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = _detic_path(cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH)
    return cfg


def register_lvis_v1(names: Sequence[str]) -> None:
    """Re-registers detectron2's builtin LVIS v1 splits against DETIC_ROOT/datasets (see setup_detic_data.sh)."""
    datasets_root = str(DETIC_ROOT / "datasets")
    for name in names:
        DatasetCatalog.remove(name)
        MetadataCatalog.remove(name)
        image_root, json_file = _PREDEFINED_SPLITS_LVIS["lvis_v1"][name]
        register_lvis_instances(name, get_lvis_instances_meta(name), os.path.join(datasets_root, json_file), os.path.join(datasets_root, image_root))
