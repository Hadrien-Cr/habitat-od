"""
Regression coverage for github.com/3dlg-hcvc/hssd/issues/2 ("Previewing HSSD dataset
scenes in Habitat-2.0"): a user previewing HSSD scenes got flat, washed-out renders; the
HSSD/habitat-lab maintainers' fix was to enable PBR + image-based lighting (IBL), which -
in the habitat-sim version this repo pins (0.3.3, see third_party/habitat-sim) - is
controlled by habitat_sim._ext.habitat_sim_bindings.PbrShaderAttributes, defaulting to
enable_ibl=True (third_party/habitat-sim/src/esp/metadata/attributes/PbrShaderAttributes.cpp:11-36).

That default is loaded from a config habitat-sim resolves *relative to the process's
current working directory* - "./data/default.pbr_config.json"
(third_party/habitat-sim/src/esp/CMakeLists.txt:45,
src/esp/gfx/configure.h.cmake:5-6) - falling back, if absent, to bundled images that "must
be found in /data/pbr" (PbrShaderAttributes.cpp comment above ibl_blut_filename /
ibl_envmap_filename). Every entry point in this repo runs with CWD resolving `data/` to
$HABITAT_DATA (see CLAUDE.md), which is populated from the HSSD-HAB download and does not
ship either of those - unlike habitat-sim's own checkout, which bundles both
(third_party/habitat-sim/data/default.pbr_config.json,
third_party/habitat-sim/data/pbr/{bluts,env_maps}/*). Separately,
MetadataMediator::getPbrShaderAttributesManager() exists in C++
(third_party/habitat-sim/src/esp/metadata/MetadataMediator.h:197-200) but is never bound to
Python (third_party/habitat-sim/src/esp/bindings/MetadataMediatorBindings.cpp only exposes
ao/asset/lighting/object/physics/stage_template_manager) - so this repo has no way to
inspect or override IBL settings at runtime even if it wanted to.

test_ibl_resources_missing_from_project_data_root statically documents that gap.
test_scene_render_changes_with_ibl_resources_available renders the exact scene from the
issue (102343992) to empirically confirm the resource gap has a real effect on HSSD-hab's
rendered appearance.
"""
import os
from pathlib import Path

import habitat_sim
import magnum as mn
import numpy as np
import pytest
from PIL import Image

HABITAT_DATA = os.environ.get("HABITAT_DATA")
_DATASET_CONFIG = (
    f"{HABITAT_DATA}/scene_datasets/hssd-hab/hssd-hab.scene_dataset_config.json"
    if HABITAT_DATA else None
)
_HABITAT_SIM_CHECKOUT = Path(__file__).resolve().parent.parent / "third_party" / "habitat-sim"

# Scene ID from the issue's own repro (`test_scene = "data/hssd/hssd-scenes/scenes/102343992.glb"`).
_ISSUE_SCENE_ID = "scenes/102343992.scene_instance.json"
_TESTDUMP_DIR = os.path.join(os.path.dirname(__file__), "testdump", "test_image_based_lighting")
os.system(f"rm -rf {_TESTDUMP_DIR}")


def test_ibl_resources_missing_from_project_data_root():
    """habitat-sim resolves its default IBL config/resources as ./data/default.pbr_config.json
    and ./data/pbr/{bluts,env_maps}/* relative to CWD. This repo's data/ (-> $HABITAT_DATA)
    ships neither, so the PBR shader's environment map/BRDF LUT silently fail to load and HSSD
    scenes render without proper image-based lighting - the rendering-quality complaint in
    github.com/3dlg-hcvc/hssd/issues/2. Fails until those resources are vendored into
    $HABITAT_DATA (e.g. copied from third_party/habitat-sim/data/{default.pbr_config.json,pbr/})
    or a dataset-level .pbr_config.json is added pointing at valid, resolvable files."""
    data_root = Path("data")
    if not data_root.exists():
        pytest.skip("no data/ (HABITAT_DATA) configured in this environment")

    default_pbr_config = data_root / "default.pbr_config.json"
    fallback_manifest = data_root / "pbr" / "PbrImages.conf"
    fallback_brdf_lut = data_root / "pbr" / "bluts" / "brdflut_ldr_512x512.png"
    fallback_env_map = data_root / "pbr" / "env_maps" / "lythwood_room_1k.hdr"

    have_dataset_config = default_pbr_config.exists()
    have_fallback = fallback_manifest.exists() and fallback_brdf_lut.exists() and fallback_env_map.exists()

    assert have_dataset_config or have_fallback, (
        f"neither {default_pbr_config} nor the fallback {fallback_manifest}/"
        f"{fallback_brdf_lut}/{fallback_env_map} exist, so habitat-sim's IBL environment map "
        "cannot resolve when this repo's entry points run (see module docstring for why "
        "-> github.com/3dlg-hcvc/hssd/issues/2)"
    )


@pytest.mark.skipif(
    not _DATASET_CONFIG or not os.path.exists(_DATASET_CONFIG),
    reason="requires HABITAT_DATA pointing at a real hssd-hab dataset",
)
@pytest.mark.skipif(
    not _HABITAT_SIM_CHECKOUT.exists(),
    reason="requires the vendored third_party/habitat-sim checkout (ships fallback IBL resources)",
)
def test_scene_render_changes_with_ibl_resources_available():
    """Renders the exact scene from the issue (102343992) from this repo's own CWD (missing
    IBL resources, per test_ibl_resources_missing_from_project_data_root above) and again
    with CWD switched to third_party/habitat-sim (which bundles them), same camera pose both
    times. If IBL resource availability has no effect the two would be identical; empirically
    they are not (~64% of pixels differ by >10/255 on the scene's own furnished courtyard),
    confirming this repo's missing resources are not a no-op."""

    def render_from(cwd: Path) -> np.ndarray:
        prev_cwd = os.getcwd()
        try:
            os.chdir(cwd)
            backend_cfg = habitat_sim.SimulatorConfiguration()
            backend_cfg.scene_id = _ISSUE_SCENE_ID
            backend_cfg.scene_dataset_config_file = _DATASET_CONFIG
            backend_cfg.enable_physics = False

            rgb_spec = habitat_sim.CameraSensorSpec()
            rgb_spec.uuid = "rgb"
            rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
            rgb_spec.resolution = [480, 640]
            rgb_spec.hfov = mn.Deg(90)
            rgb_spec.position = [0.0, 1.2, 0.0]

            agent_cfg = habitat_sim.agent.AgentConfiguration()
            agent_cfg.sensor_specifications = [rgb_spec]

            sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))
            agent = sim.initialize_agent(0)
            state = habitat_sim.AgentState()
            state.position = np.array([0.0, 0.1, 0.0])
            agent.set_state(state)
            rgb = sim.get_sensor_observations()["rgb"][:, :, :3].copy()
            sim.close()
            return rgb
        finally:
            os.chdir(prev_cwd)

    without_resources = render_from(Path(__file__).resolve().parent.parent)
    with_resources = render_from(_HABITAT_SIM_CHECKOUT)

    os.makedirs(_TESTDUMP_DIR, exist_ok=True)
    diff = np.abs(without_resources.astype(int) - with_resources.astype(int))
    combined = Image.fromarray(np.concatenate([without_resources, with_resources], axis=1))
    combined.save(os.path.join(_TESTDUMP_DIR, "without_vs_with_ibl_resources.png"))
    Image.fromarray(diff.astype("uint8")).save(os.path.join(_TESTDUMP_DIR, "diff.png"))

    changed_fraction = (diff.max(axis=-1) > 10).mean()
    assert changed_fraction > 0.1, (
        f"only {changed_fraction:.1%} of pixels changed by >10/255 between rendering with vs. "
        "without IBL's env-map/BRDF-LUT resources available on CWD - expected the environment "
        "map to visibly affect this scene's shading (see testdump/test_image_based_lighting/)"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
