"""
End-to-end check that ObjectDetectorGTSensor (common/env_utils/object_detector_sensors.py)
correctly classifies and separates HSSD-HAB's decomposed dining-set composites,
using the real scene they're placed in (scenes/102344022.scene_instance.json)
and the real sensor pipeline (setup_semantic_labels + decompose_frame) - not a
synthetic stand-in. See tests/test_decomposed.py for the underlying
ObjectSemanticsHSSD classification check this builds on.

Each test finds a dining set's real object instances in the scene, renders a
view of it, and checks decompose_frame's detections: exactly one "table" and
one "chair" per chair instance, each with its own separate box/mask (not
merged - see ObjectDetectorGTSensor.decompose_frame's obj_id*1000+class_id
semantic buffer encoding). A bounding-box snapshot is saved to
tests/testdump/test_composite_objects/ for visual sanity-checking.
"""
import os

import habitat_sim
import magnum as mn
import numpy as np
import pytest
from PIL import Image, ImageDraw

from habitat.config.default_structured_configs import ObjectDetectorGTSensorConfig
from common.env_utils.object_detector_sensors import ObjectDetectorGTSensor, get_all_objects, object_shortname_from_handle

HABITAT_DATA = os.environ.get("HABITAT_DATA")
_DATASET_CONFIG = (
    f"{HABITAT_DATA}/scene_datasets/hssd-hab/hssd-hab.scene_dataset_config.json"
    if HABITAT_DATA else None
)
_SCENE_PATH = (
    f"{HABITAT_DATA}/scene_datasets/hssd-hab/scenes/102344022.scene_instance.json"
    if HABITAT_DATA else None
)

pytestmark = pytest.mark.skipif(
    not _DATASET_CONFIG or not os.path.exists(_DATASET_CONFIG),
    reason="requires HABITAT_DATA pointing at a real hssd-hab dataset",
)

_TESTDUMP_DIR = os.path.join(os.path.dirname(__file__), "testdump", "test_composite_objects")
os.system(f"rm -rf {_TESTDUMP_DIR}")
_ROLE_COLORS = {"table": (255, 165, 0), "chair": (0, 220, 0)}

# (set name, table part template id, chair part template id, camera position
# looking at the set) - the same two dining sets as tests/test_decomposed.py,
# both placed for real in scenes/102344022.scene_instance.json.
DINING_SETS = [
    ("pavilion", "eeaf34edd2065a3fa2af3fc021cd343ca029f696_part_1", "eeaf34edd2065a3fa2af3fc021cd343ca029f696_part_3", [-5.84, 0.9, -3.2]),
    ("lakeland", "ba28803b05660ca87ad0650276988f02dce1081e_part_1", "ba28803b05660ca87ad0650276988f02dce1081e_part_4", [-15.7, 0.9, 2.65]),
]


@pytest.fixture(scope="module")
def sensor():
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_dataset_config_file = _DATASET_CONFIG
    backend_cfg.scene_id = _SCENE_PATH
    backend_cfg.enable_physics = True

    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "rgb"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = [480, 480]
    rgb_spec.hfov = mn.Deg(90)

    semantic_spec = habitat_sim.CameraSensorSpec()
    semantic_spec.uuid = "semantic"
    semantic_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_spec.resolution = [480, 480]
    semantic_spec.hfov = mn.Deg(90)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_spec, semantic_spec]

    ns = habitat_sim.NavMeshSettings()
    ns.set_defaults()
    backend_cfg.navmesh_settings = ns

    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))

    config = ObjectDetectorGTSensorConfig(
        env_name="HSSD-HAB/HSSD80", area_thr=0.0, filter_low_visibility=False,
        min_visibility_fraction=0.0, filter_out_classes=[],
    )
    s = ObjectDetectorGTSensor(sim, config)
    s.setup_semantic_labels()
    yield s
    sim.close()


def _object_ids_for_template(sim, template_id: str) -> list[int]:
    """Every real object instance in the scene whose template is exactly
    `template_id` (e.g. all 6 repeated chair instances of one dining set)."""
    return [
        obj.object_id for obj in get_all_objects(sim)
        if object_shortname_from_handle(obj.handle) == template_id
    ]


@pytest.mark.parametrize("set_name,table_part,chair_part,camera_pos", DINING_SETS)
def test_dining_set_classified_and_separated_by_real_sensor(sensor, set_name, table_part, chair_part, camera_pos):
    sim = sensor._sim
    classes = sensor.get_classes()

    table_ids = set(_object_ids_for_template(sim, table_part))
    chair_ids = set(_object_ids_for_template(sim, chair_part))
    assert table_ids, f"{set_name}: no real instance of table template {table_part} found in the scene"
    assert chair_ids, f"{set_name}: no real instance of chair template {chair_part} found in the scene"

    state = habitat_sim.AgentState()
    state.position = np.array(camera_pos, dtype=np.float32)
    sim.get_agent(0).set_state(state)

    obs = sim.get_sensor_observations()
    result = sensor.decompose_frame(obs["semantic"], sim.get_agent(0).get_state())
    instances = result["instances"]

    detections = [
        {"box": box, "class_name": classes[class_id], "object_id": info["object_id"]}
        for box, class_id, info in zip(
            instances.pred_boxes.tensor.numpy(), instances.pred_classes.numpy(), instances.infos,
        )
    ]

    image = Image.fromarray(obs["rgb"][:, :, :3].copy())
    draw = ImageDraw.Draw(image)
    for d in detections:
        color = _ROLE_COLORS.get(d["class_name"], (200, 0, 0))
        x1, y1, x2, y2 = d["box"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1, max(0, y1 - 12)), f"{d['class_name']} obj{d['object_id']}", fill=color)
    os.makedirs(_TESTDUMP_DIR, exist_ok=True)
    image.save(os.path.join(_TESTDUMP_DIR, f"{set_name}.png"))

    detections_by_id = {d["object_id"]: d for d in detections}

    found_table_ids = table_ids & detections_by_id.keys()
    found_chair_ids = chair_ids & detections_by_id.keys()
    assert found_table_ids, f"{set_name}: table instance {table_ids} not detected in frame"
    assert found_chair_ids, f"{set_name}: no chair instance {chair_ids} detected in frame"

    for obj_id in found_table_ids:
        assert detections_by_id[obj_id]["class_name"] == "table", (
            f"{set_name}: table instance obj{obj_id} classified as {detections_by_id[obj_id]['class_name']!r}"
        )
    for obj_id in found_chair_ids:
        assert detections_by_id[obj_id]["class_name"] == "chair", (
            f"{set_name}: chair instance obj{obj_id} classified as {detections_by_id[obj_id]['class_name']!r}"
        )

    # every detected chair must get its own separate box, not merged with siblings
    assert len({tuple(detections_by_id[i]["box"]) for i in found_chair_ids}) == len(found_chair_ids), (
        f"{set_name}: some chair instances share an identical box (not separated): {found_chair_ids}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
