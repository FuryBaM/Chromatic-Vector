"""Replace covered concave runs of poster paths with straight edges."""
from dataclasses import replace
import re

import numpy as np
from shapely.geometry import Polygon

from .curve_fitting import sample_curves
from .hierarchy import PATH_PREFIX
from .metrics import render_svg


COMMANDS = re.compile(r'([MLCZ])\s*([^MLCZ]*)')


def straighten_recesses(path, scale):
    """Keep outward vertices and bridge short inward turns of one closed ring."""
    commands = [(letter, text.strip()) for letter, text in COMMANDS.findall(path)]
    if not commands or commands[0][0] != 'M' or commands[-1][0] != 'Z' or 'A' in path:
        return path, 0
    if any(letter not in ('L', 'C') for letter, _ in commands[1:-1]):
        return path, 0
    start = np.fromstring(commands[0][1], sep=' ')
    original_segments = len(commands)-2
    previous = start
    pieces = []
    for letter, text in commands[1:-1]:
        points = np.fromstring(text, sep=' ').reshape(-1, 2)
        endpoint = points[-1]
        if letter == 'L':
            pieces.append(np.vstack((previous, endpoint)))
        else:
            pieces.append(sample_curves([(previous, *points)], spacing=.6))
        previous = endpoint
    if len(pieces) < 3:
        return path, 0
    if not np.array_equal(previous, start):
        pieces.append(np.vstack((previous, start)))
        commands.insert(-1, ('L', commands[0][1]))
    original = Polygon(np.vstack(pieces))
    if not original.is_valid or not original.exterior.is_ccw:
        return path, 0

    simplified = [commands[0]]
    spans = []
    filled_recesses = []
    index = 0
    while index < len(pieces):
        end = min(len(pieces), index + 24)
        origin = pieces[index][0]
        while end > index + 1:
            endpoint = pieces[end-1][-1]
            chord = endpoint - origin
            length = np.linalg.norm(chord)
            if 0 < length <= 32*scale:
                offsets = np.vstack(pieces[index:end]) - origin
                depth = (chord[0]*offsets[:, 1] - chord[1]*offsets[:, 0]) / length
                if depth.min() >= -1e-6 and depth.max() <= 6*scale:
                    recess = Polygon(offsets + origin)
                    if recess.area < 1e-6:
                        break
                    # A straight lead-in can lie on the closing chord of the pocket.
                    if not recess.is_valid:
                        recess = recess.buffer(0)
                    if (not recess.is_empty and original.intersection(recess).area < 1e-6
                            and all(recess.intersection(other).area < 1e-6 for other in filled_recesses)):
                        filled_recesses.append(recess)
                        break
            end -= 1
        if end == index + 1:
            simplified.append(commands[end])
            spans.append(pieces[index])
        else:
            endpoint = pieces[end-1][-1]
            simplified.append(('L', ' '.join(commands[end][1].split()[-2:])))
            spans.append(np.vstack((origin, endpoint)))
        index = end

    outline = Polygon(np.vstack(spans))
    if simplified[-1][0] == 'L' and np.array_equal(np.fromstring(simplified[-1][1], sep=' '), start):
        simplified.pop()
    removed = original_segments - (len(simplified)-1)
    if removed <= 0 or not outline.is_valid or not outline.buffer(1e-6).covers(original):
        return path, 0
    return ' '.join(f'{letter} {text}' for letter, text in simplified) + ' Z', removed


def _within_render_budget(before, after, max_channel_delta):
    difference = np.abs(after[..., :3].astype(np.int16) - before[..., :3])
    return (np.array_equal(after[..., 3], before[..., 3])
            and difference.max() <= max_channel_delta and difference.mean() <= .04)


def stack_poster(document, labels, scale, max_error, working_baseline):
    """Place simplified opaque planes under their covers, with a rendered guard."""
    baseline = render_svg(document.svg(display=True).encode()) if document.display_size else working_baseline
    working_render = working_baseline
    original_shapes = sorted(document.shapes, key=lambda shape: shape.draw_order)
    lower_planes = []
    remaining = original_shapes.copy()
    saved = 0
    for shape in original_shapes:
        if (shape.kind != 'base' or shape.layer != 'base_planes' or shape.level != 'primary'
                or shape.fill[3] != 255 or shape.locked or shape.clipped_by_uid
                or len(shape.source_region_ids) != 1 or not shape.geometry.startswith(PATH_PREFIX)):
            continue
        rings = ['M '+ring for ring in shape.geometry[len(PATH_PREFIX):-1].split('M ') if ring]
        simplified = [straighten_recesses(ring, scale) for ring in rings]
        removed = sum(count for _, count in simplified)
        if not removed:
            continue
        changed = replace(shape, geometry=PATH_PREFIX+' '.join(path for path, _ in simplified)+'"')
        x0, y0, x1, y1 = shape.bbox
        padding = int(np.ceil(max_error)) + 2
        bounds = (max(0, int(x0)-padding), max(0, int(y0)-padding),
                  min(document.width, int(x1)+padding), min(document.height, int(y1)+padding))

        def mask(part):
            isolated = replace(document, shapes=[part], geometry_metrics={})
            return render_svg(isolated.svg(bounds=bounds).encode())[..., 3] >= 128

        old, new = mask(shape), mask(changed)
        if np.any(old & ~new):
            continue
        upper = [item for item in remaining if item.uid != shape.uid]
        opaque_ids = [rid for item in upper if item.kind == 'base' and item.fill[3] == 255
                      and not item.locked and not item.clipped_by_uid for rid in item.source_region_ids]
        x0, y0, x1, y1 = bounds
        if not np.isin(labels[y0:y1, x0:x1][new & ~old], opaque_ids + shape.source_region_ids).all():
            continue
        stack = [replace(item, draw_order=i) for i,item in enumerate([*lower_planes, changed, *upper])]
        candidate = replace(document, shapes=stack)
        rendered = render_svg(candidate.svg(display=True).encode())
        if _within_render_budget(baseline, rendered, 16):
            working = render_svg(candidate.svg().encode()) if document.display_size else rendered
            if not _within_render_budget(working_baseline, working, 24):
                continue
            working_render = working
            lower_planes.append(changed)
            remaining = upper
            saved += removed
    if not saved:
        return document, 0, working_render
    shapes = [replace(item, draw_order=i) for i,item in enumerate([*lower_planes, *remaining])]
    return replace(document, shapes=shapes), saved, working_render
