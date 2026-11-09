"""Color quantization and region labeling. OpenCV + NumPy reference stage."""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class Segmentation:
    labels: np.ndarray  # H x W int32, unique connected-region IDs
    palette_ids: np.ndarray  # H x W int32, indices into palette
    region_colors: np.ndarray  # region-count x 3 uint8
    sizes: np.ndarray  # region-count int64


def poster_contrast(rgb):
    """Separate ink-dark shadows from lit planes with a hue-preserving tone curve."""
    light = np.dot(rgb, (.2126, .7152, .0722))
    target = np.interp(light, (0, 11, 32, 65, 100, 145, 180, 225, 255),
                       (0, 11, 23, 65, 111, 160, 194, 231, 255))
    shift = round(np.clip(.6 * (target - light), -min(rgb), 255 - max(rgb)))
    return tuple(channel + shift for channel in rgb)


def _oklab(rgb: np.ndarray) -> np.ndarray:
    """sRGB uint8 -> OKLab float32 (D65)."""
    v = rgb.astype(np.float32) / 255.0
    v = np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
    r, g, b = v[..., 0], v[..., 1], v[..., 2]
    l = np.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m = np.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s = np.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return np.stack((
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    ), axis=-1)


def quantize(rgb: np.ndarray, colors: int = 18, seed: int = 23) -> np.ndarray:
    """Cluster perceptual colors; assign every pixel to its nearest centroid."""
    if colors < 2 or colors > 256:
        raise ValueError("colors must be between 2 and 256")
    lab = _oklab(rgb).reshape(-1, 3)
    n = lab.shape[0]
    rng = np.random.default_rng(seed)
    sample = lab[rng.choice(n, size=min(n, 30000), replace=False)]
    sample = np.ascontiguousarray(sample * np.array((1.0, 1.5, 1.5), dtype=np.float32))
    cv2.setRNGSeed(seed)
    _, _, centers = cv2.kmeans(sample, colors, None,
                              (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 80, 0.005),
                              3, cv2.KMEANS_PP_CENTERS)
    result = np.empty(n, dtype=np.int32)
    scale = np.array((1.0, 1.5, 1.5), dtype=np.float32)
    for start in range(0, n, 30000):
        block = lab[start:start+30000] * scale
        squared = np.sum((block[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
        result[start:start+len(block)] = squared.argmin(axis=1)
    return result.reshape(rgb.shape[:2])


def connected_regions(palette_ids: np.ndarray, rgb: np.ndarray) -> Segmentation:
    """Make disconnected patches independent regions (even with the same palette color)."""
    labels = np.full(palette_ids.shape, -1, dtype=np.int32)
    current = 0
    for index in range(int(palette_ids.max())+1):
        mask = (palette_ids == index).astype(np.uint8)
        if not mask.any():
            continue
        count, components = cv2.connectedComponents(mask, connectivity=4)
        region = components > 0
        labels[region] = components[region] + current - 1
        current += count - 1
    if np.any(labels < 0):
        raise AssertionError("All pixels must be assigned to a region")
    flat = labels.reshape(-1)
    sizes = np.bincount(flat, minlength=current).astype(np.int64)
    color_sums = np.stack([
        np.bincount(flat, weights=rgb[..., channel].reshape(-1), minlength=current)
        for channel in range(3)
    ], axis=-1)
    mean_colors = np.rint(color_sums / sizes[:, None]).astype(np.uint8)
    return Segmentation(labels, palette_ids, mean_colors, sizes)


def merge_small_regions(palette_ids: np.ndarray, rgb: np.ndarray, min_area: int) -> np.ndarray:
    """Merge tiny components toward neighbors with long shared boundaries and similar color.

    Two rounds, then a final component labeling. Iterates over small regions, not image pixels.
    """
    if min_area <= 1:
        return palette_ids
    pixels = palette_ids.copy()
    for _ in range(2):
        seg = connected_regions(pixels, rgb)
        small = np.flatnonzero(seg.sizes < min_area)
        if len(small) == 0:
            break
        r = seg.labels
        a = np.concatenate((r[:, :-1].ravel(), r[:-1, :].ravel()))
        b = np.concatenate((r[:, 1:].ravel(), r[1:, :].ravel()))
        diff = a != b
        a, b = a[diff], b[diff]
        # Ordered adjacency allows counting shared border lengths once for each region.
        key = np.concatenate((a.astype(np.int64) * len(seg.sizes) + b,
                              b.astype(np.int64) * len(seg.sizes) + a))
        uniq, counts = np.unique(key, return_counts=True)
        source = uniq // len(seg.sizes)
        neighbor = uniq % len(seg.sizes)
        new_palette = np.empty(len(seg.sizes), dtype=np.int32)
        # Connected regions have a constant palette index.
        new_palette[r.ravel()] = pixels.ravel()
        for rid in small:
            left = np.searchsorted(source, rid, side="left")
            right = np.searchsorted(source, rid, side="right")
            indices = np.arange(left, right)
            indices = indices[seg.sizes[neighbor[indices]] >= seg.sizes[rid]]
            if not len(indices):
                continue
            candidates = neighbor[indices]
            color_dist = np.linalg.norm(
                seg.region_colors[candidates].astype(float) - seg.region_colors[rid].astype(float),
                axis=1)
            score = counts[indices] / (1.0 + color_dist * 0.025)
            new_palette[rid] = new_palette[candidates[np.argmax(score)]]
        pixels = new_palette[r]
    return pixels


def edge_aware_refine(rgb: np.ndarray, ids: np.ndarray, passes: int = 2,
                      strength: float = 0.00045, sigma: float = 0.045) -> np.ndarray:
    """Refine cluster IDs using an image-edge-gated Potts term.

    In smooth areas this repairs isolated quantization pixels. At strong input
    gradients, adjacent cluster boundaries remain free to pass. This is a
    deterministic, synchronous local optimization (not an AI model).
    """
    if passes <= 0:
        return ids
    lab = _oklab(rgb)
    n = int(ids.max()) + 1
    counts = np.bincount(ids.ravel(), minlength=n)
    sums = np.stack([np.bincount(ids.ravel(), weights=lab[..., j].ravel(), minlength=n)
                     for j in range(3)], axis=-1)
    centroids = (sums / np.maximum(1, counts)[:, None]).astype(np.float32)
    weights = np.array([1.0, 1.5, 1.5], dtype=np.float32)
    lab *= weights
    centroids *= weights
    h, w = ids.shape
    yy = np.arange(h)[:, None]
    xx = np.arange(w)[None, :]
    # Border pixels have fewer neighbors, not fictitious duplicated neighbors.
    neighbors = []
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        yn, xn = np.clip(yy + dy, 0, h - 1), np.clip(xx + dx, 0, w - 1)
        valid = ((yy + dy >= 0) & (yy + dy < h) &
                 (xx + dx >= 0) & (xx + dx < w))
        distance = np.sum((lab - lab[yn, xn]) ** 2, axis=-1)
        edge_weight = np.exp(-distance / (2 * sigma * sigma)).astype(np.float32) * valid
        neighbors.append((yn, xn, edge_weight))
    current = ids.copy()
    for _ in range(passes):
        candidates = [current] + [current[yn, xn] for yn, xn, _ in neighbors]
        best = current.copy()
        best_cost = np.full((h, w), np.inf, dtype=np.float32)
        for candidate in candidates:
            diff = lab - centroids[candidate]
            cost = np.sum(diff * diff, axis=-1)
            for yn, xn, edge_weight in neighbors:
                cost += strength * edge_weight * (candidate != current[yn, xn])
            better = cost < best_cost - 1.0e-10
            best[better] = candidate[better]
            best_cost[better] = cost[better]
        current = best
    return current


def segment(rgb: np.ndarray, colors: int = 18, min_area: int = 6, seed: int = 23,
            refine: int = 0) -> Segmentation:
    ids = quantize(rgb, colors, seed)
    if refine:
        ids = edge_aware_refine(rgb, ids, passes=refine)
    ids = merge_small_regions(ids, rgb, min_area)
    return connected_regions(ids, rgb)
