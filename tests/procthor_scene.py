"""
ProcTHOR-hab's ProcTHOR-Train-0, the first collection scene of habitat_embodied_sl/config/collection_procthor.yaml, on a
raw habitat_sim.Simulator shared by the tests: rgb, semantic and depth cameras, lit and sky-painted as ExplorationEnv
does, its navmesh recomputed (ProcTHOR-hab ships none) and its objects labelled with setup_semantic_labels' object_id *
1000 + class_id semantic ids. The recomputed navmesh also covers the roof; every agent run starts from the same
start_state on the floor island.

Needs ProcTHOR-hab under HABITAT_DATA/scene_datasets/ai2thor-hab.
"""

import os
from typing import Tuple

import habitat_sim
import magnum as mn
import numpy as np

from common.env_utils.env_base import SKY_RGB, navmesh_floor_offset
from common.env_utils.lighting import AI2THOR_BRIGHT_LIGHTS
from common.env_utils.object_annotations import PROCTHOR_ROOT, ObjectAnnotation, resolve_classes, setup_semantic_labels
from common.hybrid_exploration.exploration_map import MapConfig
from common.utils.pose_utils import quaternion_from_rpy
from common.utils.projection_utils import DepthCamera

SCENE_ID = "ProcTHOR-Train-0"
ENV_NAME = "ProcTHOR-hab"
VOCAB_NAME = "PROCTHOR100"
CLASSES = resolve_classes(ENV_NAME, VOCAB_NAME)
DATASET_CONFIG = os.path.join(PROCTHOR_ROOT, "ai2thor-hab.scene_dataset_config.json")
HFOV_DEG = 90.0
CAMERA_HEIGHT = 1.0
DEPTH_CAMERA = DepthCamera(hfov=HFOV_DEG, min_depth=0.1, max_depth=16.0)
MAP_CONFIG = MapConfig(fov_deg=HFOV_DEG, resolution_m=0.05, map_size_m=48.0, visibility_dist_m=5.0)
TURN_ANGLE_DEG = 45.0
START_POSITION = np.array([-6.435, 0.15, 2.566])
START_YAW_RAD = np.deg2rad(5 * TURN_ANGLE_DEG)  # snapped to the turn step, diagonal to the world axes
_RESOLUTION = [640, 640]


def _camera_spec(uuid: str, sensor_type: habitat_sim.SensorType) -> habitat_sim.CameraSensorSpec:
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = uuid
    spec.sensor_type = sensor_type
    spec.resolution = _RESOLUTION
    spec.hfov = mn.Deg(HFOV_DEG)
    spec.position = [0.0, CAMERA_HEIGHT, 0.0]
    spec.clear_color = [channel / 255.0 for channel in SKY_RGB]
    return spec


def _agent_config() -> habitat_sim.agent.AgentConfiguration:
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [
        _camera_spec("rgb", habitat_sim.SensorType.COLOR),
        _camera_spec("semantic", habitat_sim.SensorType.SEMANTIC),
        _camera_spec("depth", habitat_sim.SensorType.DEPTH),
    ]
    for action in ("turn_left", "turn_right"):
        agent_cfg.action_space[action].actuation.amount = TURN_ANGLE_DEG
    forward = agent_cfg.action_space["move_forward"].actuation.amount
    for action in ("move_left", "move_right", "move_backward"):
        agent_cfg.action_space[action] = habitat_sim.agent.ActionSpec(action, habitat_sim.agent.ActuationSpec(amount=forward))
    return agent_cfg


def procthor_sim(scene_id: str = SCENE_ID) -> Tuple[habitat_sim.Simulator, ObjectAnnotation]:
    """(Simulator of scene_id, its objects' annotation); setup_semantic_labels runs once, as a second call would read the
    ids it baked back as native ones and unlabel every object."""
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_dataset_config_file = DATASET_CONFIG
    backend_cfg.scene_id = scene_id
    backend_cfg.enable_physics = True
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [_agent_config()]))
    sim.set_light_setup(AI2THOR_BRIGHT_LIGHTS)
    nav_settings = habitat_sim.NavMeshSettings()
    nav_settings.set_defaults()
    nav_settings.include_static_objects = True
    assert sim.recompute_navmesh(sim.pathfinder, nav_settings)
    return sim, setup_semantic_labels(sim, ENV_NAME, VOCAB_NAME)


def floor_island(pathfinder: habitat_sim.PathFinder) -> int:
    """The lowest navmesh island, the floor under the roof the recomputed navmesh also covers."""
    return min(range(pathfinder.num_islands), key=lambda island: pathfinder.get_random_navigable_point(island_index=island)[1])


def start_state() -> habitat_sim.AgentState:
    """START_POSITION, on the floor island, facing START_YAW_RAD."""
    state = habitat_sim.AgentState()
    state.position = START_POSITION
    state.rotation = quaternion_from_rpy(0, 0, START_YAW_RAD)
    return state


def floor_height(sim: habitat_sim.Simulator, position: np.ndarray) -> float:
    return float(position[1]) - navmesh_floor_offset(sim, position)
