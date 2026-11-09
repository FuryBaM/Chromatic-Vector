"""Shape-first document model inferred from a connected region-ID raster.

This is a *topological* (not semantic/AI) hierarchy: a parent's region is the
closest raster region toward the canvas boundary. Regions remain disjoint in SVG.
"""
from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass, field
import numpy as np

from .contours import signed_area

LAYER_ORDER = ('background', 'silhouettes', 'secondary', 'shading', 'details')


@dataclass
class Shape:
    region_id: int
    fill_rgb: tuple[int, int, int]
    pixel_area: int
    bbox: tuple[int, int, int, int]
    outer: list[np.ndarray]
    holes: list[np.ndarray]
    parent_id: int | None
    nesting_depth: int
    layer: str
    adjacent: tuple[int, ...]
    contrast: float
    primitive: str = 'path'
    children: list[int] = field(default_factory=list)


@dataclass
class ShapeDocument:
    width: int
    height: int
    shapes: dict[int, Shape]
    layers: dict[str, list[int]]
    adjacency_lengths: dict[tuple[int, int], int]
    background_id: int | None

    @property
    def shape_count(self) -> int:
        return len(self.shapes)

    def layer_stats(self) -> dict[str, int]:
        return {layer: len(self.layers[layer]) for layer in LAYER_ORDER}


def _neighborhood(labels: np.ndarray) -> dict[tuple[int, int], int]:
    """Length of each shared raster border; vectorized over pixels."""
    n = int(labels.max()) + 1
    left = np.concatenate((labels[:, :-1].ravel(), labels[:-1, :].ravel()))
    right = np.concatenate((labels[:, 1:].ravel(), labels[1:, :].ravel()))
    different = left != right
    a = np.minimum(left[different], right[different]).astype(np.int64)
    b = np.maximum(left[different], right[different]).astype(np.int64)
    if not len(a):
        return {}
    encoded, counts = np.unique(a * n + b, return_counts=True)
    return {(int(k // n), int(k % n)): int(v) for k, v in zip(encoded, counts)}


def _is_full_rectangle(outer: list[np.ndarray], holes: list[np.ndarray]) -> bool:
    """Recognize exact *raster* axis-aligned rectangles without approximation."""
    if len(outer) != 1 or holes:
        return False
    ring = outer[0]
    low, high = ring.min(axis=0), ring.max(axis=0)
    rect_area = float(np.prod(high-low))
    if rect_area <= 0 or abs(signed_area(ring)-rect_area) > 1e-5:
        return False
    # A full-area enclosed polygon contained in its own bbox must be that bbox.
    return True


def build_document(labels: np.ndarray, colors: np.ndarray,
                   rings: dict[int, list[np.ndarray]]) -> ShapeDocument:
    if labels.ndim != 2 or labels.size == 0:
        raise ValueError('Nonempty 2D label map expected')
    h, w = labels.shape
    n = len(colors)
    sizes = np.bincount(labels.ravel(), minlength=n)
    neighbors = {i: {} for i in range(n) if sizes[i]}
    adjacent_lengths = _neighborhood(labels)
    for (a, b), length in adjacent_lengths.items():
        neighbors[a][b] = length
        neighbors[b][a] = length

    # Distance in the region-adjacency graph from the canvas border gives a
    # deterministic raster nesting hierarchy, including islands in holes.
    border = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    distances = {int(i): 0 for i in border}
    q = deque(sorted(distances))
    while q:
        rid = q.popleft()
        for other in sorted(neighbors[rid]):
            if other not in distances:
                distances[other] = distances[rid] + 1
                q.append(other)
    # Region-ID arrays may be synthetic and disconnected; still assign deterministically.
    for rid in neighbors:
        distances.setdefault(rid, 0)

    corners = [int(labels[0, 0]), int(labels[0, -1]), int(labels[-1, 0]), int(labels[-1, -1])]
    candidate = max(set(corners), key=lambda x: (corners.count(x), int(sizes[x]), -x))
    background_id = candidate if corners.count(candidate) >= 3 else None

    xx = np.arange(w, dtype=np.int32)[None, :]
    yy = np.arange(h, dtype=np.int32)[:, None]
    # Bounds via vectorized np.minimum/maximum.at, no per-region masks.
    x0 = np.full(n, w, dtype=np.int32); x1 = np.zeros(n, dtype=np.int32)
    y0 = np.full(n, h, dtype=np.int32); y1 = np.zeros(n, dtype=np.int32)
    np.minimum.at(x0, labels.ravel(), np.broadcast_to(xx, (h, w)).ravel())
    np.maximum.at(x1, labels.ravel(), np.broadcast_to(xx, (h, w)).ravel()+1)
    np.minimum.at(y0, labels.ravel(), np.broadcast_to(yy, (h, w)).ravel())
    np.maximum.at(y1, labels.ravel(), np.broadcast_to(yy, (h, w)).ravel()+1)
    # The background may be split into several near-identical color regions:
    # a large region touching 3+ canvas sides is a better candidate than an
    # isolated corner color. Avoid using largest-area alone (a central object
    # could be the largest region without being the background).
    candidates = [int(i) for i in border if sizes[i] >= h*w*0.10 and
                  (int(x0[i]==0)+int(y0[i]==0)+int(x1[i]==w)+int(y1[i]==h)) >= 3]
    if candidates:
        dominant = max(candidates, key=lambda i: (int(sizes[i]), -i))
        if background_id is None or sizes[dominant] > sizes[background_id]:
            background_id = dominant

    layers = {name: [] for name in LAYER_ORDER}
    shapes: dict[int, Shape] = {}
    pixel_total = w*h
    for rid in sorted(neighbors):
        parent_options = [other for other in neighbors[rid] if distances[other] < distances[rid]]
        parent = (max(parent_options, key=lambda other: (neighbors[rid][other], -other))
                  if parent_options else None)
        weights = neighbors[rid]
        contrast = (sum(np.linalg.norm(colors[rid].astype(float)-colors[j].astype(float))*length
                        for j, length in weights.items()) / max(1, sum(weights.values())))
        area = int(sizes[rid])
        if rid == background_id or (
            background_id is not None and rid in border and
            area >= max(128, int(pixel_total*0.0005)) and
            np.linalg.norm(colors[rid].astype(float)-colors[background_id].astype(float)) < 9.0
        ):
            layer = 'background'
        elif area >= max(64, int(pixel_total * 0.004)):
            layer = 'silhouettes'
        elif area >= max(24, int(pixel_total * 0.00020)):
            layer = 'secondary'
        elif contrast < 26.0:
            layer = 'shading'
        else:
            layer = 'details'
        loops = rings.get(rid, [])
        outer = [ring for ring in loops if signed_area(ring) > 0]
        holes = [ring for ring in loops if signed_area(ring) < 0]
        shape = Shape(rid, tuple(map(int, colors[rid])), area,
                      (int(x0[rid]), int(y0[rid]), int(x1[rid]), int(y1[rid])),
                      outer, holes, parent, distances[rid], layer,
                      tuple(sorted(weights)), float(contrast),
                      'rectangle-candidate' if _is_full_rectangle(outer, holes) else 'path')
        shapes[rid] = shape
        layers[layer].append(rid)
    for rid, shape in shapes.items():
        if shape.parent_id is not None:
            shapes[shape.parent_id].children.append(rid)
    return ShapeDocument(w, h, shapes, layers, adjacent_lengths, background_id)


def layer_geometry_styles(doc: ShapeDocument, simplify: float, smooth: float):
    """Per-region requested geometry. Shared-boundary engine picks stricter pair."""
    scalars = {
        'background': (1.25, 1.0),
        'silhouettes': (1.05, 1.0),
        'secondary': (0.85, 0.9),
        'shading': (1.15, 0.95),
        'details': (0.5, 0.6),
    }
    return {rid: ((min(0.1, simplify), 0.0) if shape.primitive == 'rectangle-candidate' else
                  (max(0.0, simplify*scalars[shape.layer][0]),
                   min(1.0, max(0.0, smooth*scalars[shape.layer][1]))))
            for rid, shape in doc.shapes.items()}
