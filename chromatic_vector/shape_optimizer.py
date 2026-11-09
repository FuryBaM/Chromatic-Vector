"""Conservative raster-region union, before extracting shared SVG boundaries.

Merges only edge-adjacent low-contrast regions when at least one is small.
The SVG graph is rebuilt after merging, so there are no independently rounded
unions, dangling holes or different geometry on opposite sides of a boundary.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .palette import Segmentation
from .shape_graph import _neighborhood


@dataclass
class MergeStats:
    before: int
    after: int
    joined: int
    color_mae_delta: float


def merge_similar_shapes(seg: Segmentation, rgb: np.ndarray,
                         tolerance: float = 10.0, max_small_area: int = 48) -> tuple[Segmentation, MergeStats]:
    if tolerance < 0 or max_small_area < 0:
        raise ValueError('tolerance and max_small_area must be nonnegative')
    n = len(seg.sizes)
    if tolerance == 0 or max_small_area == 0 or n < 2:
        return seg, MergeStats(n, n, 0, 0.0)
    boundaries = _neighborhood(seg.labels)
    colors = seg.region_colors.astype(np.float64)
    raw = seg.labels
    roots = np.arange(n)
    sizes = seg.sizes.astype(np.int64).copy()
    mean = colors.copy()

    def root(i):
        while roots[i] != i:
            roots[i] = roots[roots[i]]
            i = roots[i]
        return i

    candidates = []
    for (a, b), common in boundaries.items():
        distance = float(np.linalg.norm(colors[a]-colors[b]))
        if distance <= tolerance and min(int(sizes[a]), int(sizes[b])) <= max_small_area:
            candidates.append((distance, -common, a, b))
    candidates.sort()
    for _, _, a, b in candidates:
        a, b = root(a), root(b)
        if a == b or min(sizes[a], sizes[b]) > max_small_area:
            continue
        if np.linalg.norm(mean[a]-mean[b]) > tolerance:
            continue
        # Deterministic union by size then region ID; keep weighted mean.
        if sizes[a] < sizes[b] or (sizes[a] == sizes[b] and a > b):
            a, b = b, a
        old_a, old_b = sizes[a], sizes[b]
        mean[a] = (mean[a]*old_a + mean[b]*old_b)/(old_a+old_b)
        sizes[a] += sizes[b]
        roots[b] = a
    parents = np.array([root(i) for i in range(n)], dtype=np.int32)
    _, reindex = np.unique(parents, return_inverse=True)
    new_labels = reindex[raw]
    new_n = int(reindex.max()) + 1
    new_sizes = np.bincount(new_labels.ravel(), minlength=new_n).astype(np.int64)
    # Compute the actual mean RGB values from original pixels, not averaged rounded colors.
    color_sums = np.stack([np.bincount(new_labels.ravel(), weights=rgb[..., c].ravel(), minlength=new_n)
                           for c in range(3)], axis=-1)
    new_colors = np.rint(color_sums/new_sizes[:,None]).astype(np.uint8)
    old_mae = np.abs(seg.region_colors[raw].astype(np.float32)-rgb.astype(np.float32)).mean()
    new_mae = np.abs(new_colors[new_labels].astype(np.float32)-rgb.astype(np.float32)).mean()
    # palette_ids records the original palette pixel assignment. It is intentionally
    # unchanged because region IDs and palette IDs represent different concepts.
    new_seg = Segmentation(new_labels, seg.palette_ids, new_colors, new_sizes)
    return new_seg, MergeStats(n, new_n, n-new_n, float(new_mae-old_mae))
