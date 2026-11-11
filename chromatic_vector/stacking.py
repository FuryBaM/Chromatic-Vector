"""Simplify hidden edges and fill covered holes in opaque poster paths."""
from dataclasses import replace
import re

import numpy as np
from shapely.geometry import Polygon

from .curve_fitting import sample_curves
from .hierarchy import PATH_DELIMITERS, PATH_PREFIX
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


def _stack_shape(document, shape, lower, upper, labels, scale, max_error, baseline, working_baseline):
    rings = ['M '+ring.strip() for ring in shape.geometry[len(PATH_PREFIX):-1].split('M ') if ring]
    straightened = [straighten_recesses(ring, scale) for ring in rings] if (
        shape.layer == 'base_planes' and shape.level == 'primary') else []
    shortcuts = sum(count for _, count in straightened)
    if not shortcuts and (not shape.holes or len(rings) < 2):
        return shape, 0, None

    covers = [item for item in upper if item.kind == 'base' and item.fill[3] == 255
              and not item.locked and not item.clipped_by_uid]
    opaque_ids = [rid for item in covers for rid in item.source_region_ids]
    x0, y0, x1, y1 = shape.bbox
    padding = int(np.ceil(max_error)) + 2
    x0, y0 = max(0, int(x0)-padding), max(0, int(y0)-padding)
    x1, y1 = min(document.width, int(x1)+padding), min(document.height, int(y1)+padding)
    bounds = x0, y0, x1, y1
    local_labels = labels[y0:y1, x0:x1]

    def alpha(parts, metrics=None):
        isolated = replace(document, shapes=parts, geometry_metrics=metrics or {})
        return render_svg(isolated.svg(bounds=bounds).encode())[..., 3]

    old_alpha = alpha([shape])
    old = old_alpha >= 128
    current_alpha = old_alpha
    cover_mask = None

    def try_rings(paths, holes):
        nonlocal cover_mask, current_alpha
        changed = replace(shape, geometry=PATH_PREFIX+' '.join(paths)+'"', holes=holes)
        new_alpha = alpha([changed])
        new = new_alpha >= 128
        added = new & ~old
        if np.any(old & ~new) or not np.isin(local_labels[added], opaque_ids + shape.source_region_ids).all():
            return None
        if holes < shape.holes:
            hole = new & (current_alpha < 128)
            # Removing a subpath can shift Cairo's edge coverage by a couple of alpha levels.
            if not hole.any() or np.any(new_alpha.astype(np.int16) + 2 < current_alpha):
                return None
            if cover_mask is None:
                cover_mask = alpha(covers, {'seam_underpaint': document.geometry_metrics.get('seam_underpaint')}) >= 128
            if not cover_mask[hole].all():
                return None
        stack = [replace(item, draw_order=i) for i, item in enumerate([*lower, changed, *upper])]
        candidate = replace(document, shapes=stack)
        rendered = render_svg(candidate.svg(display=True).encode())
        if not _within_render_budget(baseline, rendered, 16):
            return None
        working = render_svg(candidate.svg().encode()) if document.display_size else rendered
        if _within_render_budget(working_baseline, working, 24):
            current_alpha = new_alpha
            return changed, working
        return None

    changed, saved, working = shape, 0, None
    if shortcuts:
        paths = [path for path, _ in straightened]
        result = try_rings(paths, shape.holes)
        if result is not None:
            changed, working = result
            rings, saved = paths, shortcuts
    # Removing a hole can only add fill. Test individual rings so a visible
    # opening in the same shape does not prevent closing a covered one.
    for index in range(len(rings)-1, -1, -1):
        if not changed.holes or len(rings) < 2:
            break
        paths = rings[:index] + rings[index+1:]
        result = try_rings(paths, changed.holes-1)
        if result is not None:
            changed, working = result
            saved += len(PATH_DELIMITERS.findall(rings[index])) - 2
            rings = paths
    return changed, saved, working


def stack_poster(document, labels, scale, max_error, working_baseline):
    """Place simplified opaque planes under their covers, with a rendered guard."""
    baseline = render_svg(document.svg(display=True).encode()) if document.display_size else working_baseline
    working_render = working_baseline
    original_shapes = sorted(document.shapes, key=lambda shape: shape.draw_order)
    lower_planes = []
    remaining = original_shapes.copy()
    saved = 0
    for shape in original_shapes:
        if (shape.kind != 'base' or shape.fill[3] != 255 or shape.locked or shape.clipped_by_uid
                or len(shape.source_region_ids) != 1 or not shape.geometry.startswith(PATH_PREFIX)):
            continue
        if not shape.holes and (shape.layer != 'base_planes' or shape.level != 'primary'):
            continue
        upper = [item for item in remaining if item.uid != shape.uid]
        changed, removed, working = _stack_shape(document, shape, lower_planes, upper, labels, scale,
                                                 max_error, baseline, working_baseline)
        if working is not None:
            working_render = working
            lower_planes.append(changed)
            remaining = upper
            saved += removed
    if not lower_planes:
        return document, 0, working_render
    shapes = [replace(item, draw_order=i) for i,item in enumerate([*lower_planes, *remaining])]
    return replace(document, shapes=shapes), saved, working_render
