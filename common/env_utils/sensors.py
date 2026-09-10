import copy
import itertools
import logging
import math
from typing import Any
import habitat_sim
from omegaconf import DictConfig

import cv2
import habitat # type: ignore
import numpy as np
from gym import spaces
from habitat.core.registry import registry # type: ignore[import]
from habitat.core.simulator import AgentState, Sensor, SensorTypes, Simulator # type: ignore[import]
from habitat.utils.visualizations import fog_of_war, maps # type: ignore[import]
from PIL import Image, ImageDraw

from habitat_sim.utils import common as sim_utils
import habitat.sims.habitat_simulator.sim_utilities as sutils # type: ignore

from common.utils.pose_utils import get_yaw

log = logging.getLogger(__name__)


def get_obj_from_id(sim: habitat_sim.Simulator,obj_id: int,):
    rom = sim.get_rigid_object_manager()
    if rom.get_library_has_id(obj_id):
        return rom.get_object_by_id(obj_id)
    return None

def object_shortname_from_handle( object_handle: str) -> str:
    return object_handle.split("/")[-1].split(".")[0].split("_:")[0].split("_")[0]

@registry.register_sensor(name="position_sensor")
class AgentPositionSensor(habitat.Sensor):
    def __init__(self, sim, config, **kwargs: Any):
        super().__init__(config=config)
        self._sim = sim

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "position"

    # Defines the type of the sensor
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return habitat.SensorTypes.POSITION

    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(3,),
            dtype=np.float32,
        )

    # This is called whenver reset is called or an action is taken
    def get_observation(self, *args: Any, **kwargs: Any):
        return {
            'position': self._sim.get_agent_state().position,
            'orientation': self._sim.get_agent_state().rotation,
        }

@registry.register_sensor(name="colored_tdmap_sensor")
class ColoredTDMapSensor(habitat.Sensor):
    r"""Colored top-down navigability map (light gray = navigable, dark = not) with the
    agent's current position + heading marked (blue dot + line) -- same rendering
    tests/test_path_planning.py's GIF uses, minus the path trail/goal marker. The base map is
    cached per scene (keyed on sim.habitat_config.scene, same check ShortestPathFollower's own
    _build_follower uses) since it's identical every step within one scene."""

    _GROUND_FLOOR_MPP = 0.025  # 10 more resolution than the std walk
    _OUTPUT_MAP_SIZE = 512

    def __init__(self, sim, config, **kwargs: Any):
        super().__init__(config=config)
        self._sim = sim
        self._cached_scene = None
        self._base_map: Image.Image = None
        self._tdmap_shape = None

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "colored_tdmap"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0, high=255, shape=(self._OUTPUT_MAP_SIZE, self._OUTPUT_MAP_SIZE, 3), dtype=np.uint8)

    def _get_base_map(self):
        scene = self._sim.habitat_config.scene
        if self._cached_scene != scene:
            pathfinder = self._sim.pathfinder
            lower_bound, _ = pathfinder.get_bounds()
            tdmap = pathfinder.get_topdown_view(meters_per_pixel=self._GROUND_FLOOR_MPP, height=lower_bound[1])
            colored = np.full((*tdmap.shape, 3), 40, dtype=np.uint8)
            colored[tdmap == 1] = (210, 210, 210)
            self._base_map = Image.fromarray(colored).resize((self._OUTPUT_MAP_SIZE, self._OUTPUT_MAP_SIZE), Image.NEAREST).convert("RGB")
            self._tdmap_shape = tdmap.shape
            self._cached_scene = scene
        return self._base_map, self._tdmap_shape

    def get_observation(self, *args: Any, **kwargs: Any):
        base_map, tdmap_shape = self._get_base_map()
        pathfinder = self._sim.pathfinder
        state = self._sim.get_agent_state()
        sx, sy = self._OUTPUT_MAP_SIZE / tdmap_shape[1], self._OUTPUT_MAP_SIZE / tdmap_shape[0]

        def px(position):
            row, col = maps.to_grid(position[2], position[0], tdmap_shape, pathfinder=pathfinder)
            return col * sx, row * sy

        yaw = get_yaw(state.rotation.w, state.rotation.x, state.rotation.y, state.rotation.z)
        ax, ay = px(state.position)
        tx, ty = px(state.position + np.array([-0.25 * np.sin(yaw), 0, -0.25 * np.cos(yaw)]))

        img = base_map.copy()
        draw = ImageDraw.Draw(img)
        draw.line([ax, ay, tx, ty], fill=(30, 120, 220), width=3)
        draw.ellipse([ax - 3, ay - 3, ax + 3, ay + 3], fill=(30, 120, 220))
        return np.array(img)


@registry.register_sensor(name="agent_collision_sensor")
class AgentCollisionSensor(habitat.Sensor):
    r"""Estimates agent collision (when moving forward)
    Args:
        sim: reference to the simulator for calculating task observations.
        config: contains the MAP_SCALE, MAP_SIZE, HEIGHT_THRESH fields to
                decide grid-size, extents of the projection, and the thresholds
                for determining obstacles and explored space.
    """

    def __init__(self, sim, config, **kwargs):
        super().__init__(config=config)
        self._sim = sim
        self.prev_position = self._sim.get_agent_state().position
        self.prev_rotation = self._sim.get_agent_state().rotation

    def _get_uuid(self, *args, **kwargs):
        return "agent_collision_sensor"

    def _get_sensor_type(self, *args, **kwargs):
        return bool

    def _get_observation_space(self, *args, **kwargs):
        from gym import spaces

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(1,),
            dtype=np.int32,
        )

    def get_observation(self, observations, *args, episode, **kwargs):
        self.curr_position = self._sim.get_agent_state().position
        self.curr_rotation = self._sim.get_agent_state().rotation

        collision = False
        # print ("curr position:",self.curr_position)
        if np.linalg.norm(self.curr_position - self.prev_position) < 0.15 and np.allclose(self.curr_rotation, self.prev_rotation):
            collision = True
            # print("collision!")
        
        self.prev_position = copy.deepcopy(self.curr_position)
        self.prev_rotation = copy.deepcopy(self.curr_rotation)

        return collision