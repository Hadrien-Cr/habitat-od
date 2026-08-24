"""
Renders one isometric rgb|(rgb+semantic overlay) snapshot (side by side) of an entire
scene, for one HSSD-HAB scene, one Gibson-Semantic scene, one MP3D scene, and two
ProcTHOR-hab scenes - raw habitat_sim.Simulator, no habitat-lab Env/task. Each scene is
independently skipped if its data isn't present, so this runs against whichever of the four
datasets you have locally.

All use the same outside diagonal-elevated camera for a "dollhouse cutaway" view. HSSD,
Gibson, and MP3D are real-world scans with no exterior roof, so that camera sees straight
through where the roof would be - MP3D's raw assets are authored Z-up (its
mp3d.scene_dataset_config.json declares "up": [0, 0, 1]), but habitat_sim converts that at
load time, so world coordinates are already Y-up like the others and it needs no special
handling here. ProcTHOR scenes are procedurally generated *with* a real roof/ceiling mesh
(confirmed by rendering it directly and getting a single flat-colored image), so for
ProcTHOR the ceiling is stripped first: each stage .glb has one "Ceiling#room|N|c" node per
room in its gltf scene graph (found via trimesh), distinct from the floor/wall nodes, so
they can be removed and the stage re-exported. A minimal scene_dataset_config.json is then
built pointing at that stripped stage plus the *original* scene_instance.json (so furniture
placement is untouched) and the original furniture object templates - written under
_TESTDUMP_DIR, never modifying anything under HABITAT_DATA.

Doesn't touch sim.semantic_scene.objects for HSSD/ProcTHOR (neither populates it in this
pipeline - only the rendered per-pixel semantic buffer is used here), so none of the
per-object obb/aabb axis issues from test_gibson.py are relevant. MP3D and Gibson do
populate it, but this file never reads it either, for consistency.
"""

import glob
import json
import math
import os

import habitat_sim
import magnum as mn
import numpy as np
import pytest
import quaternion
import trimesh
from PIL import Image

from habitat_sim.utils.common import colorize_ids

HABITAT_DATA = os.environ.get("HABITAT_DATA")
_HSSD_CONFIG = f"{HABITAT_DATA}/scene_datasets/hssd-hab/hssd-hab.scene_dataset_config.json" if HABITAT_DATA else None
_GIBSON_CONFIG = f"{HABITAT_DATA}/scene_datasets/gibson_semantic/gibson_semantic.scene_dataset_config.json" if HABITAT_DATA else None
_PROCTHOR_CONFIG = f"{HABITAT_DATA}/scene_datasets/ai2thor-hab/ai2thor-hab/ai2thor-hab.scene_dataset_config.json" if HABITAT_DATA else None
_MP3D_CONFIG = f"{HABITAT_DATA}/scene_datasets/mp3d/mp3d.scene_dataset_config.json" if HABITAT_DATA else None

_HSSD_SCENE = "102816756"
_GIBSON_SCENE = "Corozal"
_MP3D_SCENE = "17DRP5sb8fy"
_PROCTHOR_SCENE = "ProcTHOR-Train-9632"
_PROCTHOR_SCENE_2 = "ProcTHOR-Train-6340"  # multi-room house, exercises the ceiling-stripping across several rooms at once

_RESOLUTION = [2048, 2048]
_HFOV_DEG = 90.0
_WHITE_WASH_ALPHA = 0.25  # how much a white layer lightens rgb before the semantic color goes on top
_OVERLAY_ALPHA = 1.0  # semantic color's alpha over the white-washed rgb
_ISO_DIST_MULT = 0.7  # camera distance from scene center, as a multiple of the scene's diagonal extent
_VOID_GRAY = 100  # replaces the empty black void outside the rendered scene, in both panels
_TESTDUMP_DIR = os.path.join(os.path.dirname(__file__), "testdump", "test_isometric_scene")
_PROCTHOR_NOCEIL_DIR = os.path.join(_TESTDUMP_DIR, "procthor_noceil")
os.system(f"rm -rf {_TESTDUMP_DIR}")


def _load_sim(dataset_config, scene_id):
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_dataset_config_file = dataset_config
    backend_cfg.scene_id = scene_id
    backend_cfg.enable_physics = False

    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "rgb"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = _RESOLUTION
    rgb_spec.hfov = mn.Deg(_HFOV_DEG)

    semantic_spec = habitat_sim.CameraSensorSpec()
    semantic_spec.uuid = "semantic"
    semantic_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_spec.resolution = _RESOLUTION
    semantic_spec.hfov = mn.Deg(_HFOV_DEG)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_spec, semantic_spec]

    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))
    if not sim.pathfinder.is_loaded:  # HSSD scenes don't ship a precomputed .navmesh
        nav_settings = habitat_sim.NavMeshSettings()
        nav_settings.set_defaults()
        sim.recompute_navmesh(sim.pathfinder, nav_settings)
    return sim


_CROP_MARGIN_FRAC = 0.1


def _crop_to_content(rgb: np.ndarray, margin_frac: float = _CROP_MARGIN_FRAC):
    """Bounding box of rgb's non-black pixels, padded by margin_frac of its own size on
    each side - used to cut the mostly-empty black canvas down to just the rendered scene."""
    mask = np.any(rgb != 0, axis=2)
    rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
    r0, r1 = rows[0], rows[-1] + 1
    c0, c1 = cols[0], cols[-1] + 1
    margin_r, margin_c = int((r1 - r0) * margin_frac), int((c1 - c0) * margin_frac)
    return (
        max(0, r0 - margin_r), min(rgb.shape[0], r1 + margin_r),
        max(0, c0 - margin_c), min(rgb.shape[1], c1 + margin_c),
    )


def _look_at_quat(forward, world_up=np.array([0.0, 1.0, 0.0])):
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:  # forward ~parallel to world_up
        right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    return quaternion.from_rotation_matrix(np.array([right, up, -forward]).T)


def _render_rgb_and_overlay(sim, cam_pos, target) -> Image.Image:
    state = habitat_sim.AgentState()
    state.position = cam_pos
    state.rotation = _look_at_quat(target - cam_pos)
    sim.get_agent(0).set_state(state)

    obs = sim.get_sensor_observations()
    rgb = obs["rgb"][:, :, :3]
    semantic = obs["semantic"]
    color = colorize_ids(semantic)
    h, w = semantic.shape

    rgb_layer = Image.fromarray(np.dstack([rgb, np.full((h, w), 255, dtype=np.uint8)]))
    non_black = np.any(rgb != 0, axis=2)  # mask of actual rendered scene vs the empty black void
    wash_alpha = np.where(non_black, int(255 * _WHITE_WASH_ALPHA), 0).astype(np.uint8)
    white_rgb = np.full((h, w, 3), 255, dtype=np.uint8)
    white_layer = Image.fromarray(np.dstack([white_rgb, wash_alpha]))
    washed = Image.alpha_composite(rgb_layer, white_layer)  # rgb, lightened by a translucent white mask (void stays black)

    color_layer = Image.fromarray(np.dstack([color, np.full((h, w), int(255 * _OVERLAY_ALPHA), dtype=np.uint8)]))
    composited = np.array(Image.alpha_composite(washed, color_layer).convert("RGB"))
    washed_rgb = np.array(washed.convert("RGB"))

    labeled = semantic != 0
    overlay = np.where(labeled[:, :, None], composited, washed_rgb)  # background stays washed rgb, no color tint

    r0, r1, c0, c1 = _crop_to_content(rgb)  # based on the black void, before it's recolored below

    void_color = np.array([_VOID_GRAY, _VOID_GRAY, _VOID_GRAY], dtype=np.uint8)
    rgb_display = np.where(non_black[:, :, None], rgb, void_color)
    overlay_display = np.where(non_black[:, :, None], overlay, void_color)

    combined = np.concatenate([rgb_display[r0:r1, c0:c1], overlay_display[r0:r1, c0:c1]], axis=1)
    return Image.fromarray(combined)


def _isometric_rgb_and_overlay(sim) -> Image.Image:
    """Outside diagonal-elevated camera - for open-top scan meshes (HSSD, Gibson)."""
    lower, upper = (np.array([b.x, b.y, b.z]) for b in sim.pathfinder.get_bounds())
    upper = np.array([upper[0], (lower[1] + upper[1]) / 2, upper[2]])
    center, extent = (lower + upper) / 2, upper - lower
    direction = np.array([1.0, 1.0, 1.0]) / math.sqrt(3)  # classic isometric viewing direction
    cam_pos = center + direction * np.linalg.norm(extent) * _ISO_DIST_MULT
    return _render_rgb_and_overlay(sim, cam_pos, center)


def _find_one(pattern: str) -> str:
    matches = glob.glob(pattern, recursive=True)
    assert len(matches) == 1, f"expected exactly one match for {pattern}, got {matches}"
    return matches[0]


def _strip_ceiling_dataset_config(procthor_config: str, scene_id: str, out_dir: str):
    """Builds a minimal scene_dataset_config.json in out_dir, identical to the original
    ProcTHOR scene except its stage .glb has every "Ceiling#room|*|c" node (and its child
    geometry) removed, so the outside dollhouse camera can see inside like HSSD/Gibson
    instead of just hitting the roof. Returns (dataset_config_path, new_scene_id)."""
    ai2_root = os.path.dirname(procthor_config)  # .../ai2thor-hab/ai2thor-hab

    stage_glb = _find_one(f"{ai2_root}/assets/stages/ProcTHOR/**/{scene_id}.glb")
    scene_instance_path = _find_one(f"{ai2_root}/configs/scenes/ProcTHOR/**/{scene_id}.scene_instance.json")

    stage = trimesh.load(stage_glb, process=False)
    ceiling_nodes = [n for n in stage.graph.nodes if n.startswith("Ceiling")]
    to_remove = [n for cn in ceiling_nodes for n in [cn, *stage.graph.transforms.children.get(cn, [])]]
    for n in to_remove:
        if n in stage.graph.nodes:
            stage.graph.transforms.remove_node(n)

    os.makedirs(out_dir, exist_ok=True)
    new_scene_id = f"{scene_id}-noceil"
    stage.export(f"{out_dir}/{new_scene_id}.glb")

    stage_cfg = {
        "render_asset": f"{new_scene_id}.glb",
        "up": [0, 1, 0],
        "front": [-1, 0, 0],
        "requires_lighting": True,
        "margin": 0.01,
        "friction_coefficient": 0.8,
        "restitution_coefficient": 0.0,
    }
    with open(f"{out_dir}/{new_scene_id}.stage_config.json", "w") as f:
        json.dump(stage_cfg, f)

    scene_instance = json.load(open(scene_instance_path))
    scene_instance["stage_instance"]["template_name"] = new_scene_id  # object_instances (furniture) untouched
    with open(f"{out_dir}/{new_scene_id}.scene_instance.json", "w") as f:
        json.dump(scene_instance, f)

    dataset_cfg = {
        "stages": {"paths": {".json": [out_dir]}},
        "scene_instances": {"paths": {".json": [out_dir]}},
        "objects": {"paths": {".json": [f"{ai2_root}/configs/objects"]}},
    }
    dataset_cfg_path = f"{out_dir}/custom.scene_dataset_config.json"
    with open(dataset_cfg_path, "w") as f:
        json.dump(dataset_cfg, f)

    return dataset_cfg_path, new_scene_id


@pytest.mark.skipif(
    not _HSSD_CONFIG or not os.path.exists(_HSSD_CONFIG),
    reason="requires HABITAT_DATA pointing at a real hssd-hab dataset (see INSTALL.MD)",
)
def test_hssd_isometric_scene():
    sim = _load_sim(_HSSD_CONFIG, _HSSD_SCENE)
    img = _isometric_rgb_and_overlay(sim)
    sim.close()

    os.makedirs(_TESTDUMP_DIR, exist_ok=True)
    img.save(os.path.join(_TESTDUMP_DIR, f"hssd_{_HSSD_SCENE}_isometric.pdf"))


@pytest.mark.skipif(
    not _GIBSON_CONFIG or not os.path.exists(_GIBSON_CONFIG),
    reason="requires HABITAT_DATA pointing at a real gibson_semantic dataset (see INSTALL.MD)",
)
def test_gibson_isometric_scene():
    sim = _load_sim(_GIBSON_CONFIG, _GIBSON_SCENE)
    img = _isometric_rgb_and_overlay(sim)
    sim.close()

    os.makedirs(_TESTDUMP_DIR, exist_ok=True)
    img.save(os.path.join(_TESTDUMP_DIR, f"gibson_{_GIBSON_SCENE}_isometric.pdf"))


@pytest.mark.skipif(
    not _MP3D_CONFIG or not os.path.exists(_MP3D_CONFIG),
    reason="requires HABITAT_DATA pointing at a real mp3d dataset (see INSTALL.MD)",
)
def test_mp3d_isometric_scene():
    sim = _load_sim(_MP3D_CONFIG, _MP3D_SCENE)
    img = _isometric_rgb_and_overlay(sim)
    sim.close()

    os.makedirs(_TESTDUMP_DIR, exist_ok=True)
    img.save(os.path.join(_TESTDUMP_DIR, f"mp3d_{_MP3D_SCENE}_isometric.pdf"))


@pytest.mark.skipif(
    not _PROCTHOR_CONFIG or not os.path.exists(_PROCTHOR_CONFIG),
    reason="requires HABITAT_DATA pointing at a real ai2thor-hab/ProcTHOR dataset (see INSTALL.MD)",
)
def test_procthor_isometric_scene():
    dataset_config, scene_id = _strip_ceiling_dataset_config(_PROCTHOR_CONFIG, _PROCTHOR_SCENE, _PROCTHOR_NOCEIL_DIR)
    sim = _load_sim(dataset_config, scene_id)
    img = _isometric_rgb_and_overlay(sim)
    sim.close()

    os.makedirs(_TESTDUMP_DIR, exist_ok=True)
    img.save(os.path.join(_TESTDUMP_DIR, f"procthor_{_PROCTHOR_SCENE}_isometric.pdf"))


@pytest.mark.skipif(
    not _PROCTHOR_CONFIG or not os.path.exists(_PROCTHOR_CONFIG),
    reason="requires HABITAT_DATA pointing at a real ai2thor-hab/ProcTHOR dataset (see INSTALL.MD)",
)
def test_procthor_isometric_scene_multiroom():
    dataset_config, scene_id = _strip_ceiling_dataset_config(_PROCTHOR_CONFIG, _PROCTHOR_SCENE_2, _PROCTHOR_NOCEIL_DIR)
    sim = _load_sim(dataset_config, scene_id)
    img = _isometric_rgb_and_overlay(sim)
    sim.close()

    os.makedirs(_TESTDUMP_DIR, exist_ok=True)
    img.save(os.path.join(_TESTDUMP_DIR, f"procthor_{_PROCTHOR_SCENE_2}_isometric.pdf"))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
