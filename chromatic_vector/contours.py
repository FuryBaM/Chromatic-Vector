"""One-pass region-boundary extraction and topologically closed ring tracing.

This is the *standalone Python reference implementation*. The game native C++
implementation is a possible accelerator, NOT bundled or falsely claimed as ported.
"""
from collections import defaultdict
import numpy as np


def extract_edges(labels: np.ndarray) -> np.ndarray:
    """Return Nx5 [region, x0, y0, x1, y1] oriented with interior on the right.

    Every neighboring pixel pair is inspected once; both region sides of a shared
    boundary are emitted in opposite directions to preserve shared topology.
    Coordinates lie on the integer lattice between raster pixels.
    """
    h, w = labels.shape
    parts = []

    def add(region, x0, y0, x1, y1):
        if len(region):
            parts.append(np.column_stack((region, x0, y0, x1, y1)).astype(np.int32))

    y, x = np.nonzero(labels[:, :-1] != labels[:, 1:])
    left, right = labels[y, x], labels[y, x + 1]
    add(left, x+1, y, x+1, y+1)  # left component's right boundary
    add(right, x+1, y+1, x+1, y)

    y, x = np.nonzero(labels[:-1, :] != labels[1:, :])
    top, bottom = labels[y, x], labels[y + 1, x]
    add(top, x+1, y+1, x, y+1)
    add(bottom, x, y+1, x+1, y+1)

    # Outer canvas edges.
    yy = np.arange(h, dtype=np.int32)
    xx = np.arange(w, dtype=np.int32)
    add(labels[:, 0], np.zeros(h, dtype=np.int32), yy+1,
        np.zeros(h, dtype=np.int32), yy)
    add(labels[:, -1], np.full(h, w, dtype=np.int32), yy,
        np.full(h, w, dtype=np.int32), yy+1)
    add(labels[0, :], xx, np.zeros(w, dtype=np.int32), xx+1, np.zeros(w, dtype=np.int32))
    add(labels[-1, :], xx+1, np.full(w, h, dtype=np.int32),
        xx, np.full(w, h, dtype=np.int32))
    return np.concatenate(parts, axis=0) if parts else np.zeros((0, 5), dtype=np.int32)


def trace_rings(edges: np.ndarray) -> dict[int, list[np.ndarray]]:
    """Trace boundary loops, preferring right turns at diagonal contacts.

    Outer boundaries are clockwise in image coordinates; holes are anticlockwise.
    """
    starts = defaultdict(list)
    for i, (rid, x0, y0, x1, y1) in enumerate(edges):
        starts[(int(rid), int(x0), int(y0))].append(i)
    used = np.zeros(len(edges), dtype=bool)
    output = defaultdict(list)

    def direction(e):
        x0, y0, x1, y1 = map(int, e[1:])
        if x1 > x0: return 0  # east
        if y1 > y0: return 1  # south
        if x1 < x0: return 2  # west
        return 3               # north

    preference = {1: 0, 0: 1, 3: 2, 2: 3}
    for start in range(len(edges)):
        if used[start]:
            continue
        current = start
        rid, x0, y0 = map(int, edges[start, :3])
        origin = (x0, y0)
        ring = []
        while True:
            if used[current]:
                raise ValueError("Non-manifold contour: edge visited twice")
            edge = edges[current]
            used[current] = True
            _, ax, ay, bx, by = map(int, edge)
            ring.append((ax, ay))
            if (bx, by) == origin:
                break
            candidates = [i for i in starts.get((rid, bx, by), []) if not used[i]]
            if not candidates:
                raise ValueError(f"Open contour at {(bx, by)} in region {rid}")
            d0 = direction(edge)
            current = min(candidates,
                          key=lambda i: preference[(direction(edges[i]) - d0) % 4])
        if len(ring) >= 3:
            output[rid].append(np.array(ring, dtype=np.float64))
    return dict(output)


def signed_area(ring: np.ndarray) -> float:
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
