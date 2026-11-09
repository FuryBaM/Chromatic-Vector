"""Non-destructive, user-editable correction overlays for raster-to-vector region maps.

An RGBA PNG is used as an override layer: transparent pixels are unchanged;
opaque pixels become new fixed-fill color regions. RGB colors are exact entries
(no antialiased brush colors), with up to 256 unique overlay swatches.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image
from .palette import connected_regions, Segmentation


def apply_color_overlay(seg: Segmentation, rgb: np.ndarray, filename: str | Path) -> tuple[Segmentation, dict]:
    overlay = Image.open(filename).convert("RGBA")
    height, width = rgb.shape[:2]
    if overlay.size != (width, height):
        raise ValueError(f"Correction overlay size {overlay.size} differs from processed image {(width, height)}. "
                         "Use --max-side 0 to preserve input dimensions.")
    data = np.asarray(overlay)
    active = data[..., 3] >= 128
    if not np.any(active):
        return seg, {"corrected_pixels": 0, "correction_colors": 0}
    unique, inverse = np.unique(data[..., :3][active].reshape(-1, 3), axis=0, return_inverse=True)
    if len(unique) > 256:
        raise ValueError("Correction PNG has over 256 RGB colors. Paint with a hard-edged pencil, "
                         "not an antialiased brush.")
    palette = seg.palette_ids.copy()
    base = int(palette.max()) + 1
    palette[active] = base + inverse.astype(np.int32)
    new_seg = connected_regions(palette, rgb)
    for index, color in enumerate(unique):
        ids = np.unique(new_seg.labels[palette == base + index])
        new_seg.region_colors[ids] = color
    return new_seg, {"corrected_pixels": int(active.sum()), "correction_colors": len(unique)}
