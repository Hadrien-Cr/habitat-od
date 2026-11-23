"""Vocabulary of a static test set (nyu_depth_v2.py, sunrgbd.py): its raw class names grouped by canonical spelling
(lowercased, "_" and "-" read as spaces, a trailing plural "s" dropped), each group named by its most frequent
spelling and kept with at least MIN_OCCURRENCES objects, so classes with a handful of test objects don't dominate
the mean AP."""
from collections import Counter, defaultdict
from typing import Iterable

MIN_OCCURRENCES = 100


def spelling(name: str) -> str:
    """name lowercased, "_" and "-" read as spaces."""
    return " ".join(name.lower().replace("_", " ").replace("-", " ").split())


def canonical(name: str) -> str:
    """spelling(name) without a trailing plural "s" (but "ss", e.g. glass)."""
    spelled = spelling(name)
    return spelled[:-1] if spelled.endswith("s") and not spelled.endswith("ss") else spelled


def frequent_classes(occurrences: Iterable[str], excluded: tuple) -> dict:
    """{raw name: class} of occurrences' raw names (one per object) whose group has at least MIN_OCCURRENCES objects
    and whose class is not excluded, a group's class its most frequent spelling."""
    raw_counts = Counter(occurrences)
    groups = defaultdict(Counter)
    for name, count in raw_counts.items():
        groups[canonical(name)][spelling(name)] += count
    named = {key: group.most_common(1)[0][0] for key, group in groups.items() if sum(group.values()) >= MIN_OCCURRENCES}
    classes = {key: name for key, name in named.items() if name not in excluded}
    return {name: classes[canonical(name)] for name in raw_counts if canonical(name) in classes}
