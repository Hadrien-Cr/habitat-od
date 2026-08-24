"""
Cross-env sanity check for the real ObjectDetectorGTSensor (common/env_utils/
object_detector_sensors.py) + its object_annotations.py backing (common/env_utils/
object_annotations.py) - unlike test_hssd.py/test_gibson.py/test_mp3d.py/test_procthor.py,
which each reimplement a standalone, simplified version of the baking/decoding logic inline
against raw habitat_sim, this test goes through the actual sensor for all 4 supported
env_names (HSSD-HAB, MP3D, Gibson-Semantic, ProcTHOR-hab) - same ObjectDetectorGTSensor
collect_dataset.py drives, just constructed directly against a raw habitat_sim.Simulator here
instead of through the full ExplorationEnv/Hydra config path.

For each env with data available (independently skipped otherwise - this test still runs
against whichever subset of the 4 datasets you have locally, same convention as the sibling
files), instantiates a real ObjectDetectorGTSensor against one scene and samples random
navigable positions through sensor.decompose_frame() until _N_NONEMPTY_FRAMES_PER_ENV frames
with at least one surviving detection are found (same "non_empty" notion
habitat_embodied_al/collection.py::visualize_mosaic uses for its own mosaics) or
_MAX_ATTEMPTS_PER_ENV is hit. Detections are rendered with plot_segmentation_gt
(common/utils/plot_utils.py) - the same box+"i: gt=<class>" label style visualize_mosaic uses
to build a collected run's own train_mosaic.png/val_mosaic.png - and tiled into that env's own
mosaic via make_mosaic, dumped to tests/testdump/test_object_detector/<env_name>_mosaic.png
(one file per env, not one combined mosaic) for visual inspection.

Colors/classes come from sensor.get_classes() + vocab_constants.make_colors() directly rather
than MetadataCatalog, since ProcTHOR-hab's vocabulary is fully data-driven (see
object_annotations.py::_load_procthor_classes) and was never registered into MetadataCatalog -
this keeps all 4 envs on one uniform code path instead of special-casing that one.

Requires HABITAT_DATA pointing at real data for at least one of HSSD-HAB/MP3D/Gibson-Semantic/
ProcTHOR-hab (see INSTALL.MD) - skipped entirely otherwise.
"""

import math
import os

import habitat_sim
import magnum as mn
import numpy as np
import pytest
from detectron2.structures import Instances

from habitat.config.default_structured_configs import ObjectDetectorGTSensorConfig
from common.env_utils.object_detector_sensors import ObjectDetectorGTSensor
from common.env_utils.vocab_constants import make_colors
from common.utils.plot_utils import plot_segmentation_gt, make_mosaic

HABITAT_DATA = os.environ.get("HABITAT_DATA")

_RESOLUTION = [640, 640]  # matches train_mosaic.png's tile size
_HFOV_DEG = 90.0
_AGENT_HEIGHT = 0.88  # matches common/config/hssd-hab/default.yaml's agent height
_N_NONEMPTY_FRAMES_PER_ENV = 32
_MAX_ATTEMPTS_PER_ENV = 400  # generous cap so a scene with few visible/labeled objects can't hang forever
_TESTDUMP_DIR = os.path.join(os.path.dirname(__file__), "testdump", "test_object_detector")
os.system(f"rm -rf {_TESTDUMP_DIR}")

# One scene per env - the same ones test_hssd.py/test_mp3d.py/test_gibson.py/test_procthor.py
# already use as their own smoke-test scene, so this needs no data beyond what those require.
_ENV_SPECS = {
    "HSSD-HAB": dict(
        vocab_name="HSSD80",
        dataset_config=f"{HABITAT_DATA}/scene_datasets/hssd-hab/hssd-hab.scene_dataset_config.json" if HABITAT_DATA else None,
        scene_id="102816756",
    ),
    "MP3D": dict(
        vocab_name="MPCAT40",
        dataset_config=f"{HABITAT_DATA}/scene_datasets/mp3d/mp3d.scene_dataset_config.json" if HABITAT_DATA else None,
        scene_id="2azQ1b91cZZ",
    ),
    "Gibson-Semantic": dict(
        vocab_name="COCO80",
        dataset_config=f"{HABITAT_DATA}/scene_datasets/gibson_semantic/gibson_semantic.scene_dataset_config.json" if HABITAT_DATA else None,
        scene_id="Collierville",
    ),
    "ProcTHOR-hab": dict(
        vocab_name="unused",  # ProcTHOR-hab's vocabulary is fully data-driven - see object_annotations.py
        dataset_config=f"{HABITAT_DATA}/scene_datasets/ai2thor-hab/ai2thor-hab/ai2thor-hab.scene_dataset_config.json" if HABITAT_DATA else None,
        scene_id="ProcTHOR-Train-9632",
    ),
}


def _build_sim(dataset_config: str, scene_id: str) -> habitat_sim.Simulator:
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_dataset_config_file = dataset_config
    backend_cfg.scene_id = scene_id
    # MP3D/Gibson-Semantic have no rigid-object-managed furniture to begin with (their
    # semantic annotation comes from sim.semantic_scene.objects, loaded independently of
    # physics) - enabling it uniformly for all 4 envs is harmless for them and required for
    # HSSD-HAB/ProcTHOR-hab, which do place furniture as physics-managed rigid objects.
    backend_cfg.enable_physics = True

    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "rgb"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = _RESOLUTION
    rgb_spec.hfov = mn.Deg(_HFOV_DEG)
    rgb_spec.position = [0.0, _AGENT_HEIGHT, 0.0]

    semantic_spec = habitat_sim.CameraSensorSpec()
    semantic_spec.uuid = "semantic"
    semantic_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_spec.resolution = _RESOLUTION
    semantic_spec.hfov = rgb_spec.hfov
    semantic_spec.position = rgb_spec.position

    depth_spec = habitat_sim.CameraSensorSpec()
    depth_spec.uuid = "depth"
    depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_spec.resolution = _RESOLUTION
    depth_spec.hfov = rgb_spec.hfov
    depth_spec.position = rgb_spec.position

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_spec, semantic_spec, depth_spec]

    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))
    if not sim.pathfinder.is_loaded:  # HSSD-HAB/ProcTHOR-hab don't ship a precomputed .navmesh
        nav_settings = habitat_sim.NavMeshSettings()
        nav_settings.set_defaults()
        sim.recompute_navmesh(sim.pathfinder, nav_settings)
    return sim


def _random_agent_state(sim, rng) -> habitat_sim.AgentState:
    state = habitat_sim.AgentState()
    state.position = sim.pathfinder.get_random_navigable_point()
    yaw = rng.uniform(0, 2 * math.pi)
    state.rotation = np.quaternion(math.cos(yaw / 2), 0.0, math.sin(yaw / 2), 0.0)
    return state


def _as_gt_instances(instances: Instances) -> Instances:
    """Same filtering/renaming common/env_utils/sense.py::BBSense.get_bbs_as_gt() does for
    on-disk sense files - reimplemented here since that class expects a stored sense file's
    SenseInfo/path, not a live decompose_frame() result."""
    keep = [
        i for i, info in enumerate(instances.infos)
        if not info["filtered_low_area"] and not info["filtered_low_visibility"]
    ]
    kept = instances[keep]
    target = Instances(kept.image_size)
    target.gt_boxes = kept.pred_boxes
    target.gt_classes = kept.pred_classes
    if kept.has("pred_masks"):
        target.gt_masks = kept.pred_masks
    return target


def test_generates_cross_env_detection_mosaics():
    rng = np.random.default_rng(0)
    any_env_ran = False

    for env_name, spec in _ENV_SPECS.items():
        if not spec["dataset_config"] or not os.path.exists(spec["dataset_config"]):
            print(f"{env_name}: skipped, no data at {spec['dataset_config']}")
            continue

        sim = _build_sim(spec["dataset_config"], spec["scene_id"])
        try:
            if not sim.pathfinder.is_loaded:
                print(f"{env_name}: skipped, no navmesh for scene {spec['scene_id']}")
                continue

            config = ObjectDetectorGTSensorConfig(
                env_name=env_name, vocab_name=spec["vocab_name"],
                area_thr=250, filter_low_visibility=True, min_visibility_fraction=0.15,
                filter_out_classes=[],
            )
            sensor = ObjectDetectorGTSensor(sim, config)
            sensor.setup_semantic_labels()

            classes = sensor.get_classes()
            colors = make_colors(len(classes), seed=0, ctype=0)

            agent = sim.get_agent(0)
            tiles = []
            attempts = 0
            while len(tiles) < _N_NONEMPTY_FRAMES_PER_ENV and attempts < _MAX_ATTEMPTS_PER_ENV:
                attempts += 1
                agent.set_state(_random_agent_state(sim, rng))

                obs = sim.get_sensor_observations()
                instances = sensor.decompose_frame(obs["semantic"], agent.get_state(), obs["depth"])["instances"]
                gt_instances = _as_gt_instances(instances)
                if len(gt_instances) == 0:
                    continue

                im = plot_segmentation_gt(obs["rgb"][:, :, :3].copy(), gt_instances, classes, colors)
                tiles.append((f"{env_name}_{len(tiles):02d}", np.array(im)))

            print(f"{env_name}: {len(tiles)} non-empty frames found in {attempts} attempts")

            if not tiles:
                continue
            any_env_ran = True

            os.makedirs(_TESTDUMP_DIR, exist_ok=True)
            out_path = os.path.join(_TESTDUMP_DIR, f"{env_name.lower()}_mosaic.png")
            make_mosaic(tiles, N_cols=4).save(out_path)
            print(f"{env_name}: {len(tiles)} frames dumped to {out_path}")
        finally:
            sim.close()

    if not any_env_ran:
        pytest.skip(
            "requires HABITAT_DATA pointing at real data for at least one of "
            "HSSD-HAB/MP3D/Gibson-Semantic/ProcTHOR-hab (see INSTALL.MD)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
