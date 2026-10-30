#!/usr/bin/env bash
# Lays out the data Detic's BoxSup-C2_L_CLIP_SwinB_896b32_4x training reads
# under Detic's own datasets/ and models/ dirs, the paths its configs and metadata already point at:
#   datasets/coco/{train2017,val2017}            COCO 2017 images (LVIS v1 reuses them)
#   datasets/lvis/lvis_v1_{train,val}.json       LVIS v1 annotations
#   models/swin_base_patch4_window7_224_22k.pkl  Swin-B ImageNet-22k backbone, converted to detectron2's format
# Run from repo root: common/vision/setup_detic_data.sh [--coco-images PATH]

set -euo pipefail

usage() {
    cat >&2 <<EOF
Usage: $0 [--detic-root PATH] [--coco-images PATH]

  --detic-root   Detic clone (default: third_party/Detic)
  --coco-images  existing dir holding COCO 2017's train2017/ and val2017/, symlinked instead of
                 downloading them (~19GB) from images.cocodataset.org
EOF
    exit 1
}

detic_root="third_party/Detic"
coco_images=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --detic-root) detic_root="$2"; shift 2 ;;
        --coco-images) coco_images="$2"; shift 2 ;;
        *) usage ;;
    esac
done

coco_dir="$detic_root/datasets/coco"
lvis_dir="$detic_root/datasets/lvis"
models_dir="$detic_root/models"
mkdir -p "$coco_dir" "$lvis_dir" "$models_dir"

for split in train2017 val2017; do
    [[ -e "$coco_dir/$split" ]] && continue
    if [[ -n "$coco_images" ]]; then
        ln -s "$(realpath "$coco_images/$split")" "$coco_dir/$split"
    else
        wget -c "http://images.cocodataset.org/zips/$split.zip" -P "$coco_dir"
        unzip -q "$coco_dir/$split.zip" -d "$coco_dir" && rm "$coco_dir/$split.zip"
    fi
done

for split in train val; do
    [[ -e "$lvis_dir/lvis_v1_$split.json" ]] && continue
    wget -c "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_$split.json.zip" -P "$lvis_dir"
    unzip -q "$lvis_dir/lvis_v1_$split.json.zip" -d "$lvis_dir" && rm "$lvis_dir/lvis_v1_$split.json.zip"
done

swin="$models_dir/swin_base_patch4_window7_224_22k"
if [[ ! -e "$swin.pkl" ]]; then
    wget -c "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth" -P "$models_dir"
    python "$detic_root/tools/convert-thirdparty-pretrained-model-to-d2.py" --path "$swin.pth" && rm "$swin.pth"
fi
