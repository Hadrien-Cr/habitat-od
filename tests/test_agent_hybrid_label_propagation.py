"""
Reruns test_agent_hybrid.py's HybridFrontierAgent exploration, steered by its own ObjectTracker, of procthor_scene.py's ProcTHOR-Train-0 with
the same Detector, but fuses its detections, labelled with Detic's class probabilities of their region embeddings over
the PROCTHOR100 classes, into a 3DLabelProp LabelPropagationPointCloud (common/utils/label_propagation_point_cloud.py), resolved by
per-voxel max-logit votes and 26-connected components. Into tests/testdump/test_agent_hybrid_label_propagation/ (emptied when
the test starts) go the same exploration outputs and, of the resolved point cloud, the same maps, evaluation and
retrieval mosaics as test_agent_hybrid.py's.

Needs the same data, GPU and checkpoints as test_agent_hybrid.py.
"""

from pathlib import Path

from common.utils.label_propagation_point_cloud import LabelPropagationPointCloud, predicted_point_cloud
from procthor_scene import DEPTH_CAMERA
from test_agent_hybrid import build_detector, explore_and_fuse, pytestmark, retrieve_and_save_mosaics, save_resolved_maps, timed  # noqa: F401 - pytestmark skips this module too

_TESTDUMP_DIR = Path(__file__).parent / "testdump" / "test_agent_hybrid_label_propagation"


def test_label_propagation_point_cloud_map():
    detector = build_detector()
    run, point_cloud = explore_and_fuse(
        detector, _TESTDUMP_DIR, LabelPropagationPointCloud(),
        lambda detections, frame, instances, floor: predicted_point_cloud(frame, instances, DEPTH_CAMERA, floor, detector),
    )
    try:
        with timed("Resolving"):
            point_cloud.resolve()
        save_resolved_maps(run, point_cloud)
        retrieve_and_save_mosaics(run, point_cloud)
    finally:
        run.sim.close()


if __name__ == "__main__":
    test_label_propagation_point_cloud_map()
