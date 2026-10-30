"""Loads IMPRINT's HSSD-rare ObjectNav episodes (third_party/IMPRINT/dataset/hssd_rare_eps/<scene_id>.json.gz,
habitat-lab's standard ObjectNavDatasetV1 shape) for habitat_od_search's evaluator. object_category on each episode
and goal is a fine-grained HSSD instance string (e.g. "table, Metal 25\" 2 Tier Accent Table"), rarity-tiered by its
main category (the text before the first comma) against third_party/IMPRINT/dataset/main_cat_split.json."""
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal  # type: ignore

IMPRINT_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "IMPRINT"
HSSD_RARE_EPISODES_DIR = IMPRINT_ROOT / "dataset" / "hssd_rare_eps"
MAIN_CAT_SPLIT_PATH = IMPRINT_ROOT / "dataset" / "main_cat_split.json"


@dataclass(frozen=True)
class ViewPoint:
    position: List[float]
    rotation: List[float]


@dataclass(frozen=True)
class SceneGoal:
    object_id: int
    position: List[float]
    object_category: str
    view_points: List[ViewPoint]


@dataclass(frozen=True)
class SceneEpisodes:
    scene_id: str
    episodes: List[NavigationEpisode]
    goals_by_category: Dict[str, List[SceneGoal]]


def load_scene_episodes(scene_id: str) -> SceneEpisodes:
    """Reads <scene_id>.json.gz, building habitat-lab NavigationEpisodes with their real
    start_position/start_rotation (so ExplorationEnv.reset(episode=...) places the agent exactly where
    HSSD-rare intends) and the goals_by_category entries each episode's object_category indexes into."""
    path = HSSD_RARE_EPISODES_DIR / f"{scene_id}.json.gz"
    with gzip.open(path) as f:
        data = json.load(f)

    episodes = [
        NavigationEpisode(
            episode_id=str(episode["episode_id"]),
            scene_id=scene_id,
            start_position=episode["start_position"],
            start_rotation=episode["start_rotation"],
            goals=[NavigationGoal(position=goal["position"]) for goal in data["goals_by_category"][f"{scene_id}_{episode['object_category']}"]],
            info={"object_category": episode["object_category"], **episode["info"]},
        )
        for episode in data["episodes"]
    ]
    goals_by_category = {
        category: [
            SceneGoal(
                object_id=goal["object_id"],
                position=goal["position"],
                object_category=goal["object_category"],
                view_points=[ViewPoint(vp["agent_state"]["position"], vp["agent_state"]["rotation"]) for vp in goal["view_points"]],
            )
            for goal in goals
        ]
        for category, goals in data["goals_by_category"].items()
    }
    return SceneEpisodes(scene_id, episodes, goals_by_category)


def available_scenes() -> List[str]:
    """Every scene_id with a saved hssd_rare_eps/<scene_id>.json.gz."""
    return sorted(path.name.removesuffix(".json.gz") for path in HSSD_RARE_EPISODES_DIR.glob("*.json.gz"))


def main_category(object_category: str) -> str:
    """object_category's coarse rarity-tier category: the text before its first comma."""
    return object_category.split(",", 1)[0].strip()


def rarity_tiers() -> Dict[str, str]:
    """{main category: rarity tier ("easy"/"medium"/"hard")} of every valid main category in main_cat_split.json."""
    valid = json.loads(MAIN_CAT_SPLIT_PATH.read_text())["valid"]
    return {category: tier for tier, categories in valid.items() for category in categories}
