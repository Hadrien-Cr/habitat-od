"""
Tests for mesh_visibility_fraction (common/env_utils/visibility_utils.py), the check
ObjectDetectorGTSensor.decompose_frame uses to filter out barely-visible detections before
they reach the vocab/area filters. It compares the actual number of visible mask pixels
against how many pixels the object's largest AABB face would occupy if fully visible and
facing the camera head-on at the mask's mean depth - so occlusion, cropping, and
foreshortening (an edge-on view) all pull the ratio down without needing to project the
object's real mesh.

Everything is placed high above apartment_1's own geometry, so only the objects placed here
are reachable by the semantic/depth buffers under test.
"""

import os

import magnum as mn
import numpy as np
import pytest
from PIL import Image, ImageDraw

import habitat_sim
from habitat_sim.utils.common import colorize_ids

from common.env_utils.visibility_utils import compute_obj_dimensions, mesh_visibility_fraction

_Y = 50.0
_DONUT_POS = mn.Vector3(0.0, _Y, -1.5)
_BORDER_DONUT_POS = mn.Vector3(1.5, _Y, -1.5)  # crops the donut roughly in half (empirical)
_OCCLUDER_POS = mn.Vector3(0.05, _Y, -1.4)  # cube covers roughly half the donut (empirical)
_CUBE_PARKED_POS = mn.Vector3(20.0, _Y, 20.0)
_PANEL_PARKED_POS = mn.Vector3(-20.0, _Y, 20.0)
_PANEL_POS = mn.Vector3(0.0, _Y, -1.5)
_PANEL_HALF_EXTENT = 0.3      # x/y half-size
_PANEL_HALF_THICKNESS = 0.02  # z half-size
_PANEL_TILT_DEG = 85.0  # about Y - near edge-on without fully vanishing (empirical)
_SENSOR_UUID = "semantic"
_RGB_UUID = "rgb"
_DEPTH_UUID = "depth"
_TESTDUMP_DIR = os.path.join(os.path.dirname(__file__), "testdump", "test_object_visibility")
os.system(f"rm -rf {_TESTDUMP_DIR}")


@pytest.fixture(scope="module")
def sim():
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = "data/scene_datasets/habitat-test-scenes/apartment_1.glb"
    backend_cfg.enable_physics = True

    semantic_spec = habitat_sim.CameraSensorSpec()
    semantic_spec.uuid = _SENSOR_UUID
    semantic_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_spec.resolution = [256, 256]
    semantic_spec.hfov = mn.Deg(90)
    semantic_spec.position = [0.0, 0.0, 0.0]

    # RGB sensor is only used to dump human-readable debug snapshots to tests/testdump/.
    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = _RGB_UUID
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = semantic_spec.resolution
    rgb_spec.hfov = semantic_spec.hfov
    rgb_spec.position = semantic_spec.position

    depth_spec = habitat_sim.CameraSensorSpec()
    depth_spec.uuid = _DEPTH_UUID
    depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_spec.resolution = semantic_spec.resolution
    depth_spec.hfov = semantic_spec.hfov
    depth_spec.position = semantic_spec.position

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [semantic_spec, rgb_spec, depth_spec]

    s = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))
    agent = s.get_agent(0)
    state = habitat_sim.AgentState()
    state.position = np.array([0.0, _Y, 0.0])
    agent.set_state(state)
    yield s
    s.close()


@pytest.fixture(scope="module")
def donut(sim):
    obj_templates_mgr = sim.get_object_template_manager()
    rom = sim.get_rigid_object_manager()
    donut_template_id = obj_templates_mgr.load_configs("data/test_assets/objects/donut")[0]
    obj = rom.add_object_by_template_id(donut_template_id)
    obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC  # no physics drift
    obj.translation = _DONUT_POS
    obj.semantic_id = 1
    return obj


@pytest.fixture(scope="module")
def cube(sim):
    obj_templates_mgr = sim.get_object_template_manager()
    rom = sim.get_rigid_object_manager()
    box_template = habitat_sim.attributes.ObjectAttributes()
    box_template.render_asset_handle = "data/test_assets/objects/transform_box.glb"
    box_template.scale = np.array([0.065, 0.065, 0.065])  # ~comparable to half the donut's ring
    template_id = obj_templates_mgr.register_template(box_template, "occlusion_test_cube")
    obj = rom.add_object_by_template_id(template_id)
    obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
    obj.translation = _CUBE_PARKED_POS
    obj.semantic_id = 2
    return obj


@pytest.fixture(scope="module")
def thin_panel(sim):
    """A box squashed thin along local Z, unrotated so Z lines up with the camera's -Z axis:
    near/far faces are ~0.02m apart along the view but ~0.3m across it."""
    obj_templates_mgr = sim.get_object_template_manager()
    rom = sim.get_rigid_object_manager()
    panel_template = habitat_sim.attributes.ObjectAttributes()
    panel_template.render_asset_handle = "data/test_assets/objects/transform_box.glb"
    panel_template.scale = np.array([_PANEL_HALF_EXTENT, _PANEL_HALF_EXTENT, _PANEL_HALF_THICKNESS])
    template_id = obj_templates_mgr.register_template(panel_template, "thin_panel_test")
    obj = rom.add_object_by_template_id(template_id)
    obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
    obj.translation = _PANEL_PARKED_POS
    obj.semantic_id = 3
    return obj


@pytest.fixture(autouse=True)
def _reset_scene(sim, donut, cube, thin_panel):
    donut.translation = _DONUT_POS
    cube.translation = _CUBE_PARKED_POS
    thin_panel.translation = _PANEL_PARKED_POS
    thin_panel.rotation = mn.Quaternion()
    yield


def _frame(sim):
    """Renders the current scene state and returns (semantic_obs, depth_obs, agent_state),
    matching what ObjectDetectorGTSensor.decompose_frame is called with."""
    obs = sim.get_sensor_observations()
    return obs[_SENSOR_UUID], obs[_DEPTH_UUID], sim.get_agent(0).get_state()


def _save_scene_snapshot(sim, name, fraction):
    """Saves an RGB | semantic debug PNG to tests/testdump/ annotated with the computed
    fraction, so it can be eyeballed against what the sensors actually saw."""
    os.makedirs(_TESTDUMP_DIR, exist_ok=True)
    obs = sim.get_sensor_observations()
    rgb = obs[_RGB_UUID][:, :, :3]
    semantic_img = colorize_ids(obs[_SENSOR_UUID])

    combined = Image.fromarray(np.concatenate([rgb, semantic_img], axis=1))
    draw = ImageDraw.Draw(combined)
    draw.text((4, 4), f"mesh_visibility_fraction={fraction:.3f}", fill=(255, 255, 0))

    safe_name = name.replace("/", "_")
    combined.save(os.path.join(_TESTDUMP_DIR, f"{safe_name}.png"))


def test_donut_fully_visible(sim, donut, request):
    semantic_obs, depth_obs, agent_state = _frame(sim)
    mask = semantic_obs == donut.semantic_id

    fraction = mesh_visibility_fraction(compute_obj_dimensions(donut), mask, agent_state, depth_obs)
    _save_scene_snapshot(sim, request.node.name, fraction)

    # A torus's AABB is mostly empty space (the hole, the gaps around the ring), so even a
    # fully unoccluded, uncropped donut scores well under 1.0 - what matters is it's still
    # well above the cropped/occluded cases below (~0.07-0.12).
    assert fraction > 0.15, f"expected the fully-visible donut to score highest, got {fraction:.3f}"


def test_donut_half_in_border(sim, donut, request):
    donut.translation = _BORDER_DONUT_POS
    semantic_obs, depth_obs, agent_state = _frame(sim)
    mask = semantic_obs == donut.semantic_id

    fraction = mesh_visibility_fraction(compute_obj_dimensions(donut), mask, agent_state, depth_obs)
    _save_scene_snapshot(sim, request.node.name, fraction)

    assert 0.08 <= fraction <= 0.16, f"expected roughly half the donut to be cropped, got {fraction:.3f}"


def test_donut_half_occluded_by_cube(sim, donut, cube, request):
    cube.translation = _OCCLUDER_POS
    semantic_obs, depth_obs, agent_state = _frame(sim)
    mask = semantic_obs == donut.semantic_id

    fraction = mesh_visibility_fraction(compute_obj_dimensions(donut), mask, agent_state, depth_obs)
    _save_scene_snapshot(sim, request.node.name, fraction)

    assert 0.05 <= fraction <= 0.12, f"expected roughly half the donut to be occluded, got {fraction:.3f}"


def test_thin_panel_tilted_edge_on_is_foreshortened(sim, thin_panel, request):
    """Nothing occludes the panel and its AABB (so its "fully visible" baseline pixel count)
    doesn't change with rotation, but rotated near edge-on it casts a thin sliver of a mask
    instead of its full face - mesh_visibility_fraction should reflect that shrunken
    silhouette relative to the same panel facing the camera head-on."""
    dims = compute_obj_dimensions(thin_panel)

    thin_panel.translation = _PANEL_POS
    thin_panel.rotation = mn.Quaternion()
    semantic_obs, depth_obs, agent_state = _frame(sim)
    mask = semantic_obs == thin_panel.semantic_id
    flat_fraction = mesh_visibility_fraction(dims, mask, agent_state, depth_obs)
    _save_scene_snapshot(sim, f"{request.node.name}_flat", flat_fraction)

    thin_panel.rotation = mn.Quaternion.rotation(mn.Deg(_PANEL_TILT_DEG), mn.Vector3.y_axis())
    semantic_obs, depth_obs, agent_state = _frame(sim)
    mask = semantic_obs == thin_panel.semantic_id
    tilted_fraction = mesh_visibility_fraction(dims, mask, agent_state, depth_obs)
    _save_scene_snapshot(sim, f"{request.node.name}_tilted", tilted_fraction)

    assert tilted_fraction < 0.2, (
        f"expected the edge-on tilted panel to score low despite having no occluder, got {tilted_fraction:.3f}"
    )
    assert tilted_fraction < flat_fraction, (
        f"tilted panel ({tilted_fraction:.3f}) should score lower than the same panel facing "
        f"the camera head-on ({flat_fraction:.3f})"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
