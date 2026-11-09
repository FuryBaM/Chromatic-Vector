"""Write editable, raster-free SVG paths for color regions."""
from html import escape
from pathlib import Path
import cv2
import numpy as np


def _fmt(x: float) -> str:
    return f"{float(x):.3f}".rstrip("0").rstrip(".")


def _simplify(ring: np.ndarray, epsilon: float) -> np.ndarray:
    if epsilon <= 0 or len(ring) < 5:
        return ring
    # cv2.approxPolyDP closed polylines preserve high-curvature corners reasonably well.
    out = cv2.approxPolyDP(ring.astype(np.float32), epsilon, True).reshape(-1, 2)
    return out.astype(float) if len(out) >= 3 else ring


def _bezier_path(points: np.ndarray, tension: float) -> str:
    """Experimental per-loop Bezier smoothing. Does NOT enforce shared-edge parity."""
    n = len(points)
    if n < 3:
        return ""
    parts = [f"M {_fmt(points[0, 0])} {_fmt(points[0, 1])}"]
    for i in range(n):
        prev, a, b, nxt = (points[(i-1) % n], points[i], points[(i+1) % n], points[(i+2) % n])
        t1 = (b - prev) * tension / 6.0
        t2 = (nxt - a) * tension / 6.0
        # Preserve hard corners (turn over ~70 degrees).
        v1, v2 = a-prev, b-a
        if np.linalg.norm(v1)*np.linalg.norm(v2) > 1e-8:
            if np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)) < 0.34:
                t1 = np.zeros(2)
        v1, v2 = b-a, nxt-b
        if np.linalg.norm(v1)*np.linalg.norm(v2) > 1e-8:
            if np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)) < 0.34:
                t2 = np.zeros(2)
        c1, c2 = a+t1, b-t2
        parts.append(f"C {_fmt(c1[0])} {_fmt(c1[1])} {_fmt(c2[0])} {_fmt(c2[1])} {_fmt(b[0])} {_fmt(b[1])}")
    parts.append("Z")
    return " ".join(parts)


def _ring_path(points: np.ndarray, smooth: float) -> str:
    if smooth > 0:
        return _bezier_path(points, smooth)
    out = [f"M {_fmt(points[0,0])} {_fmt(points[0,1])}"]
    out += [f"L {_fmt(x)} {_fmt(y)}" for x, y in points[1:]]
    return " ".join(out) + " Z"


def write_svg(path: str | Path, width: int, height: int, rings: dict[int, list[np.ndarray]],
              colors: np.ndarray, skip_regions: set[int] | None = None,
              simplify: float = 0.0, smooth: float = 0.0,
              shared_paths: dict[int, list[str]] | None = None) -> None:
    skip_regions = skip_regions or set()
    output = [
        '<?xml version="1.0" encoding="utf-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        '<title>Chromatic Vector — region contours</title>',
    ]
    for rid in sorted(rings):
        if rid in skip_regions:
            continue
        color = colors[rid]
        hexcolor = "#" + "".join(f"{int(v):02x}" for v in color)
        subpaths = (shared_paths[rid] if shared_paths is not None
                    else [_ring_path(_simplify(ring, simplify), smooth) for ring in rings[rid]])
        if not subpaths:
            continue
        d = " ".join(subpaths)
        output.append(f'  <path id="region-{rid}" fill="{hexcolor}" fill-rule="evenodd" d="{escape(d, quote=True)}"/>')
    output.append('</svg>')
    Path(path).write_text("\n".join(output), encoding="utf-8")
