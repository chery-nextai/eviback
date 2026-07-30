"""Reference implementation of EviBack's post-normalization lambda."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np


def validate_group_scales(group_ids: Sequence[Any], scales: Sequence[float]) -> np.ndarray:
    scale_array = np.asarray(scales, dtype=np.float64)
    if scale_array.shape != (len(group_ids),):
        raise ValueError("scales must have one value per response")
    if not np.all(np.isfinite(scale_array)) or np.any(scale_array <= 0):
        raise ValueError("scales must contain positive finite values")
    by_group: dict[Any, float] = {}
    for group_id, scale in zip(group_ids, scale_array):
        if group_id in by_group and not math.isclose(by_group[group_id], float(scale)):
            raise ValueError(
                "post-normalization scale must be constant within each group: "
                f"{group_id!r} has {by_group[group_id]} and {float(scale)}"
            )
        by_group[group_id] = float(scale)
    return scale_array


def compute_group_normalized_advantages(
    scores: Sequence[float],
    group_ids: Sequence[Any],
    *,
    scales: Sequence[float] | None = None,
    normalize_by_std: bool = True,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Normalize per GRPO group, then apply the fallback scale."""

    score_array = np.asarray(scores, dtype=np.float64)
    if score_array.shape != (len(group_ids),):
        raise ValueError("scores and group_ids must have the same length")
    if not np.all(np.isfinite(score_array)):
        raise ValueError("scores must be finite")
    scale_array = (
        np.ones_like(score_array)
        if scales is None
        else validate_group_scales(group_ids, scales)
    )
    indices: dict[Any, list[int]] = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        indices[group_id].append(index)
    advantages = np.zeros_like(score_array)
    for group_indices in indices.values():
        values = score_array[group_indices]
        if len(values) == 1:
            normalized = values.copy()
        else:
            centered = values - values.mean()
            std = values.std(ddof=1)
            normalized = centered / (std + epsilon) if normalize_by_std else centered
        advantages[group_indices] = normalized
    # Lambda is intentionally applied after normalization.
    return advantages * scale_array


def advantage_audit(
    scores: Sequence[float], group_ids: Sequence[Any], scales: Sequence[float]
) -> dict[str, Any]:
    pre = compute_group_normalized_advantages(scores, group_ids)
    post = compute_group_normalized_advantages(scores, group_ids, scales=scales)
    return {
        "advantage_pre_group_scale": pre.tolist(),
        "advantage_post_group_scale": post.tolist(),
        "scale": list(map(float, scales)),
        "pre_mean": float(pre.mean()) if len(pre) else 0.0,
        "post_mean": float(post.mean()) if len(post) else 0.0,
        "pre_abs_mean": float(np.abs(pre).mean()) if len(pre) else 0.0,
        "post_abs_mean": float(np.abs(post).mean()) if len(post) else 0.0,
    }