"""
Standalone viewer for common/utils/concept_graph.py::ConceptGraphMemory.

Reuses test_semantic_point_cloud.py's FrontierAgent-driven write/read harness. Each fused
frame's gt_instances (GT masks and depth) get the Detic region embedding of their box, are projected with project_detections_to_objects, merged into the map via
ConceptGraphMemory.__add__, and .resolve() runs concept-graphs' filter/denoise, then
ConceptGraphMemory.keep_foreground keeps the objects detected in enough of the frames showing them and
ConceptGraphMemory.classify classifies each object from its raytraced masks in every explored frame.
Then saves the same retrieval_mosaic.png as test_semantic_point_cloud.py.

Has no pytest test functions - run it directly.
"""

from functools import partial

import torch
from detectron2.structures import Boxes, Instances

from common.env_utils.object_annotations import resolve_classes
from common.utils.concept_graph import ConceptGraphMemory, FrameDetections, project_detections_to_objects
from common.utils.image_memory import Frame, ImageMemory
from common.utils.projection_utils import DepthCamera, ResolvedObjects
from common.vision.detic import Vocabulary
from detector import Detector
from test_semantic_point_cloud import _AGENT_HEIGHT, _ENV_NAME, _VOCAB_NAME, HFOV_DEG, MAX_DEPTH, MIN_DEPTH, run_viewer, save_retrieval_mosaic, viewer_arg_parser

_MIN_MESH_VISIBILITY_FRACTION = 0.2


@torch.no_grad()
def frame_objects(detector: Detector, frame_id: int, frame: Frame, gt_instances: Instances) -> FrameDetections:
    features, _, transform = detector._backbone(frame.rgb)
    boxes = Boxes(torch.as_tensor(transform.apply_box(gt_instances.gt_boxes.tensor.numpy()), dtype=torch.float32, device=detector._device))
    gt_instances.region_features = detector._region_embeddings(features, boxes).cpu()
    gt_instances.scores = torch.ones(len(gt_instances))
    floor_height = frame.pose[1, 3] - _AGENT_HEIGHT
    return project_detections_to_objects(frame_id, frame, gt_instances, DepthCamera(HFOV_DEG, MIN_DEPTH, MAX_DEPTH), floor_height)


def resolve_and_classify(detector: Detector, fused: ConceptGraphMemory, memory: ImageMemory) -> ResolvedObjects:
    fused.image_memory = memory
    camera = DepthCamera(HFOV_DEG, MIN_DEPTH, MAX_DEPTH)
    fused.resolve()
    fused.keep_foreground(camera, _MIN_MESH_VISIBILITY_FRACTION)
    fused.classify(detector, camera, _MIN_MESH_VISIBILITY_FRACTION, {})
    return fused.resolved_objects


if __name__ == "__main__":
    parser = viewer_arg_parser(__doc__, "ConceptGraphMemory")
    parser.add_argument("--config-file", required=True, help="Detic config whose region embeddings are fused")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sam-checkpoint", default="data/sam/sam_vit_b_01ec64.pth")
    args = parser.parse_args()
    vocabulary = Vocabulary(_VOCAB_NAME, tuple(resolve_classes(_ENV_NAME, _VOCAB_NAME)))
    detector = Detector(args.config_file, args.checkpoint, vocabulary, args.sam_checkpoint)
    viewer, fused, memory = run_viewer(
        args, "test_concept_graph", ConceptGraphMemory(), partial(frame_objects, detector), partial(resolve_and_classify, detector),
    )
    save_retrieval_mosaic(viewer, fused, memory, args.retrieval_objects)
