"""Hierarchical SVG shape and layer export with exact geometric primitives.

Primitive substitution is strictly guarded: only a non-curved path with the
same raster area can become an SVG rectangle or polygon. Everything else is a
compound filled path built from the shared canonical boundary geometry.
"""
from __future__ import annotations
from collections import Counter
from html import escape
from pathlib import Path
import re
import numpy as np
from .shape_graph import LAYER_ORDER, ShapeDocument

_FMT = lambda v: f'{float(v):.3f}'.rstrip('0').rstrip('.') or '0'
_LINEAR = re.compile(r'([ML])\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)')


def _as_linear_polygon(d: str, expected_area: int, max_vertices=18):
    if 'C ' in d or 'Q ' in d or d.count('M ') != 1:
        return None
    parsed = list(_LINEAR.finditer(d))
    if not parsed or len(parsed) > max_vertices + 1:
        return None
    points = np.array([(float(m.group(2)), float(m.group(3))) for m in parsed], dtype=float)
    if len(points) < 3:  # simple polygon requires >=3 corners
        return None
    if not np.allclose(points[0], points[-1], atol=1e-5):
        # SVG closes implicitly with Z, so the last point need not equal start.
        if len(points) < 3:
            return None
    elif np.allclose(points[0], points[-1], atol=1e-5):
        points = points[:-1]
    x, y = points[:,0], points[:,1]
    area = 0.5*abs(float(np.dot(x, np.roll(y,-1))-np.dot(y, np.roll(x,-1))))
    if abs(area-expected_area) > 1e-3:
        return None
    return points


def _rectangle_of(points: np.ndarray, bbox: tuple[int,int,int,int]) -> bool:
    x0, y0, x1, y1 = bbox
    for a, b in zip(points, np.roll(points,-1,axis=0)):
        if not ((abs(a[0]-b[0])<1e-6 and (abs(a[0]-x0)<1e-6 or abs(a[0]-x1)<1e-6))
                or (abs(a[1]-b[1])<1e-6 and (abs(a[1]-y0)<1e-6 or abs(a[1]-y1)<1e-6))):
            return False
    return len(points) >= 4 and abs((x1-x0)*(y1-y0)-
                                     abs(float(np.dot(points[:,0],np.roll(points[:,1],-1))-
                                               np.dot(points[:,1],np.roll(points[:,0],-1)))*0.5)) < 1e-4


def write_layered_svg(path: str | Path, doc: ShapeDocument,
                      region_paths: dict[int, list[str]],
                      skip: set[int] | None = None) -> dict[str, int]:
    skip = skip or set()
    svg = [
        '<?xml version="1.0" encoding="utf-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" viewBox="0 0 {doc.width} {doc.height}" width="{doc.width}" height="{doc.height}">',
        '<title>Chromatic Vector v0.5 — optimized layered editable shapes</title>',
    ]
    counts = Counter()
    for layer in LAYER_ORDER:
        svg.append(f'  <g id="layer-{layer}" inkscape:groupmode="layer" inkscape:label="{layer}">')
        for rid in doc.layers[layer]:
            if rid in skip:
                continue
            shape = doc.shapes[rid]
            paths = region_paths.get(rid, [])
            if not paths:
                continue
            color = '#' + ''.join(f'{channel:02x}' for channel in shape.fill_rgb)
            parent = str(shape.parent_id) if shape.parent_id is not None else 'none'
            attrs = (f'id="shape-{rid}" fill="{color}" data-region-id="{rid}" '
                     f'data-parent-id="{parent}" data-nesting-depth="{shape.nesting_depth}" '
                     f'data-layer="{layer}" data-pixel-area="{shape.pixel_area}"')
            poly = (_as_linear_polygon(paths[0], shape.pixel_area) if
                    len(paths) == 1 and not shape.holes and len(shape.outer) == 1 else None)
            if poly is not None and _rectangle_of(poly, shape.bbox):
                x0,y0,x1,y1 = shape.bbox
                svg.append(f'    <rect {attrs} x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}"/>')
                shape.primitive = 'rectangle'
            elif poly is not None:
                coords = ' '.join(f'{_FMT(x)},{_FMT(y)}' for x, y in poly)
                svg.append(f'    <polygon {attrs} points="{coords}"/>')
                shape.primitive = 'polygon'
            else:
                path_d = ' '.join(paths)
                svg.append(f'    <path {attrs} fill-rule="evenodd" d="{escape(path_d, quote=True)}"/>')
                shape.primitive = 'compound-path' if shape.holes or len(shape.outer)>1 else 'bezier-path'
            counts[shape.primitive] += 1
        svg.append('  </g>')
    svg.append('</svg>')
    Path(path).write_text('\n'.join(svg), encoding='utf-8')
    return dict(counts)
