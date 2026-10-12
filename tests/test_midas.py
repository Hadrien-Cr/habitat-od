"""
Standalone test for common/utils/midas.py::Midas.

Renders rgb + depth from random navigable poses of habitat-test-scenes' apartment_1 on a raw
habitat_sim.Simulator, estimates each frame's depth with MiDaS fitted to the simulator's depth, and
checks the fitted depth's absolute relative error, averaged over views since a view facing a
textureless wall at close range can fail on its own. A mosaic of rgb / simulator depth / estimated
depth / absolute error per frame is saved to tests/testdump/test_midas/ for visual inspection.

Requires data/scene_datasets/habitat-test-scenes/apartment_1.glb and a CUDA device - skipped otherwise.
Downloads the MiDaS DPT_Large weights through torch.hub on first run.
"""

import os
import shutil

import cv2
import habitat_sim
import numpy as np
import pytest
import torch
from habitat_sim.utils.common import quat_from_angle_axis

from common.utils.midas import Midas, fit_scale_shift
from common.utils.plot_utils import make_mosaic

_SCENE = "data/scene_datasets/habitat-test-scenes/apartment_1.glb"
_RESOLUTION = (480, 640)
_SENSOR_HEIGHT_M = 0.88
_NUM_FRAMES = 8
_MAX_ABS_REL = 0.15
_MAX_ERROR_VIS_M = 1.0
_TESTDUMP_DIR = os.path.join(os.path.dirname(__file__), "testdump", "test_midas")

pytestmark = [
    pytest.mark.skipif(not os.path.exists(_SCENE), reason=f"requires {_SCENE}"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device"),
]


def _sensor_spec(uuid: str, sensor_type: habitat_sim.SensorType) -> habitat_sim.CameraSensorSpec:
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = uuid
    spec.sensor_type = sensor_type
    spec.resolution = list(_RESOLUTION)
    spec.position = [0.0, _SENSOR_HEIGHT_M, 0.0]
    return spec


def _build_sim() -> habitat_sim.Simulator:
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = _SCENE
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [
        _sensor_spec("rgb", habitat_sim.SensorType.COLOR),
        _sensor_spec("depth", habitat_sim.SensorType.DEPTH),
    ]
    return habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))


def _random_views(sim: habitat_sim.Simulator, rng: np.random.Generator) -> list:
    """(rgb, depth) of _NUM_FRAMES seeded random navigable positions and headings."""
    sim.pathfinder.seed(0)
    views = []
    for _ in range(_NUM_FRAMES):
        state = habitat_sim.AgentState()
        state.position = sim.pathfinder.get_random_navigable_point()
        state.rotation = quat_from_angle_axis(rng.uniform(0, 2 * np.pi), np.array([0.0, 1.0, 0.0]))
        sim.get_agent(0).set_state(state)
        observations = sim.get_sensor_observations()
        views.append((observations["rgb"][:, :, :3], observations["depth"]))
    return views


@pytest.fixture(scope="module")
def estimated_views():
    """(rgb, simulator depth, estimated depth, mean absolute error) of every random view."""
    sim = _build_sim()
    views = _random_views(sim, np.random.default_rng(0))
    sim.close()
    midas = Midas()
    return [(rgb, depth, *midas.depth_estimate(rgb, depth)[::2]) for rgb, depth in views]


def _depth_image(depth: np.ndarray, max_depth: float) -> np.ndarray:
    colored = cv2.applyColorMap((np.clip(depth / max_depth, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def _save_mosaic(estimated_views: list) -> None:
    shutil.rmtree(_TESTDUMP_DIR, ignore_errors=True)
    os.makedirs(_TESTDUMP_DIR, exist_ok=True)
    tiles = []
    for idx, (rgb, depth, estimated, mean_error) in enumerate(estimated_views):
        max_depth = depth.max()
        tiles += [
            (f"view{idx}", rgb.copy()),
            ("simulator depth", _depth_image(depth, max_depth)),
            ("estimated depth", _depth_image(estimated, max_depth)),
            (f"abs error, mean {mean_error:.2f} m", _depth_image(np.abs(estimated - depth), _MAX_ERROR_VIS_M)),
        ]
    make_mosaic(tiles, N_cols=4).save(os.path.join(_TESTDUMP_DIR, "depth_mosaic.png"))


def test_fit_scale_shift_recovers_affine_map():
    prediction = np.random.default_rng(0).uniform(0.1, 5.0, 1000)
    assert np.allclose(fit_scale_shift(prediction, 2.5 * prediction - 0.3), (2.5, -0.3))


def test_estimated_depth_matches_simulator_depth(estimated_views):
    _save_mosaic(estimated_views)
    abs_rels = [np.mean(np.abs(estimated[depth > 0] - depth[depth > 0]) / depth[depth > 0]) for _, depth, estimated, _ in estimated_views]
    assert np.mean(abs_rels) < _MAX_ABS_REL, f"per-view absolute relative errors {np.round(abs_rels, 3)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
