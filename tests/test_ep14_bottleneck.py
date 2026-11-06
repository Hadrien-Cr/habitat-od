"""Confirms that the spot where habitat_od_search's eval_200_gt episode 14 (scene 102344094, category
"table_lamp, TABLE LAMP-PORTER") got a NO_FEASIBLE_PLAN stop is a genuine navmesh bottleneck -- too narrow
for the configured agent radius -- rather than an artifact of DiscretePlanner's own partial, depth-built 2D
obstacle map. The navmesh (habitat_cfg.habitat.simulator.navmesh_include_static_objects=True) is baked by
habitat-sim for the configured agent radius independently of the planner's belief map, so it's ground truth
here: clearance at the stuck position is 0.000 m against a 0.18 m agent radius, i.e. the agent was already
standing at the absolute edge of where its own footprint can exist -- no obstacle-dilation setting in
DiscretePlanner could have made this cell safely traversible, since the real geometry leaves no margin at all.

The stuck world position (-4.8906, 0.1729, -2.8880) was read off the episode's own world_trajectory at the
step where it stopped making progress (habitat_od_search/logs/eval_200_gt_ep14_debug/images/
000_ep14_102344094_last_snapshot.png shows the agent wedged in the neck connecting two rooms).
"""
import os

import numpy as np
import pytest

from common.env_utils.env_base import build_env
from common.env_utils.object_annotations import get_all_objects

HABITAT_DATA = os.environ.get("HABITAT_DATA")
_HSSD_CONFIG = f"{HABITAT_DATA}/scene_datasets/hssd-hab/hssd-hab.scene_dataset_config.json" if HABITAT_DATA else None
_SCENE_ID = "102344094"
_STUCK_WORLD_POSITION = np.array([-4.8905797, 0.17293952, -2.888038])
_OBJECT_PARAMS = {
    "env_name": "HSSD-HAB", "vocab_name": "HSSD400", "min_area_fraction": 2e-4,
    "filter_low_visibility": True, "min_mesh_visibility_fraction": 0.2, "filter_classes": [],
}


@pytest.mark.skipif(
    not _HSSD_CONFIG or not os.path.exists(_HSSD_CONFIG),
    reason="requires HABITAT_DATA pointing at a real hssd-hab dataset (see INSTALL.MD)",
)
def test_stuck_position_is_a_genuine_navmesh_bottleneck() -> None:
    """Navmesh clearance at the stuck position is below the configured agent radius -- confirmed a real
    bottleneck, not a DiscretePlanner belief-map artifact. Also reports the nearest placed object."""
    env, habitat_cfg = build_env(_SCENE_ID, seed=0, max_steps=1, object_params=_OBJECT_PARAMS)
    try:
        pathfinder = env.get_pathfinder()
        agent_radius = habitat_cfg.habitat.simulator.agents.main_agent.radius

        assert pathfinder.is_navigable(_STUCK_WORLD_POSITION), "stuck position is not on the navmesh at all"
        clearance = pathfinder.distance_to_closest_obstacle(_STUCK_WORLD_POSITION, max_search_radius=2.0)

        objects = get_all_objects(env.get_sim())
        nearest = min(objects, key=lambda obj: np.linalg.norm(np.array(obj.translation) - _STUCK_WORLD_POSITION))
        nearest_dist = float(np.linalg.norm(np.array(nearest.translation) - _STUCK_WORLD_POSITION))

        print(f"navmesh clearance: {clearance:.3f} m, agent radius: {agent_radius:.3f} m")
        print(f"nearest object: {nearest.handle} at {nearest_dist:.3f} m (object origin, not footprint)")

        assert clearance < agent_radius, (
            f"navmesh clearance {clearance:.3f} m is no longer below the agent radius {agent_radius:.3f} m at "
            "the stuck position -- the scene or navmesh config changed since this was last confirmed a bottleneck"
        )
    finally:
        env.close()
