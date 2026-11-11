"""Ordered shared Bézier fits with pixel-uncertainty bounds.

Kept separate from the legacy fitter so precise SVG output remains byte-compatible.
"""
import math
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree


def sample_curves(curves, spacing=.25):
    chunks=[]
    for a,c1,c2,b in curves:
        length=np.linalg.norm(c1-a)+np.linalg.norm(c2-c1)+np.linalg.norm(b-c2)
        # No fixed 48-sample cap: a 1500px straight edge needs a dense guard too.
        count=max(2,int(math.ceil(length/spacing)))
        t=np.linspace(0,1,count+1)[1:,None];u=1-t
        chunks.append(u**3*a+3*u*u*t*c1+3*u*t*t*c2+t**3*b)
    return np.vstack((curves[0][0],*chunks)) if chunks else np.zeros((0,2))


def turning_cost(curves):
    """Measure abrupt direction changes at approximately one-pixel intervals."""
    points = sample_curves(curves)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if not len(lengths) or lengths.sum() < 2:
        return 0.
    distance = np.r_[0., np.cumsum(lengths)]
    positions = np.linspace(0, distance[-1], int(np.ceil(distance[-1]))+1)
    uniform = np.column_stack([np.interp(positions, distance, axis) for axis in points.T])
    tangents = np.diff(uniform, axis=0)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-12)
    if np.allclose(points[0], points[-1], rtol=0., atol=1e-8):
        tangents = np.vstack((tangents, tangents[0]))
    return float(np.square(np.diff(tangents, axis=0)).sum())


def _sharp_points(core, closed):
    # Identify actual corners from long straight runs, not 1px staircase turns.
    linear=cv2.approxPolyDP(core.astype(np.float32),0,closed).reshape(-1,2)
    sharp=set()
    for i,p in enumerate(linear):
        if not closed and i in (0,len(linear)-1):continue
        va=p-linear[i-1];vb=linear[(i+1)%len(linear)]-p
        da,db=np.linalg.norm(va),np.linalg.norm(vb)
        if min(da,db)>=3 and np.dot(va,vb)/(da*db)<.5:sharp.add(tuple(p))
    return sharp


def _ordered_reduce(raw, epsilon, smooth):
    closed=np.array_equal(raw[0],raw[-1])
    core=raw[:-1] if closed else raw
    sharp=_sharp_points(core,closed)
    if smooth>0 and len(core)>5:
        filtered=gaussian_filter1d(core,sigma=.85,axis=0,mode='wrap' if closed else 'nearest')
        pts=core+(filtered-core)*smooth
    else:pts=core.copy()
    pins=[0]+[i for i,p in enumerate(core) if tuple(p) in sharp]
    if closed:
        # Split in traversal order. Never prepend an absent closed-ring anchor to
        # an arbitrarily rotated approxPolyDP result (that creates backtracking).
        pins.append(int(np.argmax(np.sum((core-core[0])**2,axis=1))))
        pts=np.vstack((pts,pts[0]));pins.append(len(core))
    else:pins.append(len(core)-1)
    pins=sorted(set(pins))
    for i in pins:pts[i]=raw[i]
    groups=[]
    for a,b in zip(pins,pins[1:]):
        section=cv2.approxPolyDP(pts[a:b+1].astype(np.float32),epsilon,False).reshape(-1,2).astype(float)
        section[0]=pts[a];section[-1]=pts[b]
        groups.extend(section if not groups else section[1:])
    result=np.asarray(groups,float)
    if closed and len(result)<4:return raw.copy(),sharp
    return result,sharp


def _cubics(points, smooth, sharp):
    closed=np.array_equal(points[0],points[-1]);n=len(points)-1
    tangent=np.zeros_like(points)
    for i in (range(n) if closed else range(1,n)):
        if tuple(points[i]) in sharp:continue
        prev=points[(i-1)%n] if closed else points[i-1]
        nxt=points[(i+1)%n] if closed else points[i+1]
        va=points[i]-prev;vb=nxt-points[i]
        da,db=np.linalg.norm(va),np.linalg.norm(vb)
        if min(da,db)<1e-9:continue
        # Keep unambiguous cusps. Moderate turns on a sampled arc stay curved.
        if min(da,db)>3 and np.dot(va,vb)/(da*db)<.3:continue
        t=(nxt-prev)*.5*smooth
        mag=np.linalg.norm(t);limit=min(da,db)*1.2
        tangent[i]=t*min(1.,limit/max(1e-12,mag))
    if closed:tangent[-1]=tangent[0]
    return [(points[i],points[i]+tangent[i]/3,points[i+1]-tangent[i+1]/3,points[i+1]) for i in range(n)]


def fit_shared_curve(raw, simplify, smooth, max_error):
    raw=np.asarray(raw,float)
    exact=simplify==0 and smooth==0
    thin=np.array_equal(raw[0],raw[-1]) and min(np.ptp(raw,axis=0))<=1
    if exact or max_error==0:
        pts,sharp=_ordered_reduce(raw,0,0)
        return _cubics(pts,0,sharp),0
    source_tree=cKDTree(raw)
    for attempt,(epsilon,strength) in enumerate(((min(simplify,max_error),smooth),
             (min(simplify,max_error)*.6,smooth*.65),(min(simplify,max_error)*.3,smooth*.3),(0,0))):
        if thin:strength=0;epsilon=min(epsilon,.45)
        pts,sharp=_ordered_reduce(raw,epsilon,strength)
        curves=_cubics(pts,strength,sharp)
        if attempt==3:return curves,3
        if _within_error(raw, source_tree, curves, max_error):return curves,attempt
    raise AssertionError('unreachable')


def _within_error(raw, source_tree, curves, max_error):
    samples = sample_curves(curves)
    # Project to source segments so sampling phase cannot reject a straight fit.
    _, indices = source_tree.query(samples, k=min(4, len(raw)))
    indices = np.asarray(indices).reshape(len(samples), -1)
    starts = np.concatenate((np.clip(indices-1, 0, len(raw)-2), np.clip(indices, 0, len(raw)-2)), axis=1)
    a = raw[starts]
    v = raw[starts+1] - a
    t = np.clip(np.sum((samples[:, None, :] - a) * v, axis=-1) /
                np.maximum(1e-15, np.sum(v*v, axis=-1)), 0, 1)
    forward = np.linalg.norm(samples[:, None, :] - (a + t[..., None]*v), axis=-1).min(axis=1).max()
    backward = cKDTree(samples).query(raw)[0].max()
    return max(forward, backward) <= max_error


def _fit_span(points, start_tangent, end_tangent):
    """Fit handle lengths with fixed positions and tangent directions."""
    a, b = points[0], points[-1]
    chord = np.linalg.norm(b - a)
    if len(points) <= 2:
        return (a, a, b, b), 0, 0.
    if chord < 1e-8:
        return (a, a, b, b), len(points)//2, np.inf
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    t = np.r_[0., np.cumsum(lengths)][:, None] / lengths.sum()
    u = 1 - t
    h1, h2 = 3*u*u*t, 3*u*t*t
    base = (u**3 + h1)*a + (t**3 + h2)*b
    handles = np.stack((h1*start_tangent, -h2*end_tangent), axis=-1).reshape(-1, 2)
    offsets = (points - base).ravel()
    weights = np.linalg.lstsq(handles, offsets, rcond=None)[0]
    weights = np.clip(weights, chord*.06, chord*.65)
    errors = np.linalg.norm((offsets - handles @ weights).reshape(-1, 2), axis=1)
    split = int(np.argmax(errors[1:-1])) + 1
    return (a, a + weights[0]*start_tangent, b - weights[1]*end_tangent, b), split, errors[split]


def _poster_spans(raw, smooth, tolerance):
    closed = np.array_equal(raw[0], raw[-1])
    core = raw[:-1] if closed else raw
    filtered = gaussian_filter1d(core, sigma=2., axis=0, mode='wrap' if closed else 'nearest')
    points = core + (filtered - core)*smooth
    sharp = _sharp_points(core, closed)
    corners = {i for i, point in enumerate(core) if tuple(point) in sharp}
    pins = corners | {0, len(raw)-1}
    fixed = list(corners | {0})
    points[fixed] = raw[fixed]
    if closed:
        pins.add(int(np.argmax(np.sum((core-core[0])**2, axis=1))))
        points = np.vstack((points, points[0]))
    else:
        points[-1] = raw[-1]

    indices = np.arange(len(core))
    if closed:
        previous, following = np.roll(filtered, 2, axis=0), np.roll(filtered, -2, axis=0)
    else:
        previous = filtered[np.maximum(indices-2, 0)]
        following = filtered[np.minimum(indices+2, len(core)-1)]
    tangents = following - previous
    tangents /= np.maximum(1e-12, np.linalg.norm(tangents, axis=1, keepdims=True))
    if closed:
        tangents = np.vstack((tangents, tangents[0]))

    ordered = sorted(pins)
    pending = list(reversed(list(zip(ordered, ordered[1:]))))
    curves = []
    while pending:
        start, end = pending.pop()
        span = points[start:end+1]
        left = tangents[start]
        right = tangents[end]
        if start % len(core) in corners:
            left = span[min(3, len(span)-1)] - span[0]
            left = left / max(1e-12, np.linalg.norm(left))
        if end % len(core) in corners:
            right = span[-1] - span[max(0, len(span)-4)]
            right = right / max(1e-12, np.linalg.norm(right))
        curve, split, error = _fit_span(span, left, right)
        if error > tolerance:
            # Keep pathological staircase chains from repeatedly splitting one point off.
            margin = len(span)//8
            split = min(max(split, margin), len(span)-1-margin)
            mid = start + split
            pending.extend(((mid, end), (start, mid)))
        else:
            curves.append(curve)
    return curves


def fit_poster_curve(raw, simplify, smooth, max_error):
    """Fit long spans to the edge trend, with corners and junctions pinned."""
    raw = np.asarray(raw, float)
    closed = np.array_equal(raw[0], raw[-1])
    short_axis = min(np.ptp(raw, axis=0))
    thin = short_axis == 0 or (closed and short_axis <= 1)
    if len(raw) < 8 or thin or not (simplify and smooth and max_error):
        return fit_shared_curve(raw, simplify, smooth, max_error)

    source_tree = cKDTree(raw)
    tolerance = min(simplify, max_error)
    # A gentler fit keeps smooth spans even if the first pass moves a tight bend too far.
    settings = ((smooth, tolerance*.58), (smooth*.72, tolerance*.43))
    for attempt, (strength, error) in enumerate(settings):
        curves = _poster_spans(raw, strength, error)
        if _within_error(raw, source_tree, curves, max_error):
            return curves, attempt
    return fit_shared_curve(raw, simplify, smooth, max_error)


# Supersampled ink fit.
INK_EPSILON = .45   # polygon tolerance as a share of the geometric tolerance
INK_CORNER = .4     # a turn is a corner when its runs bow less than this share of an even arc
INK_BEVEL = 6.      # longest bevel, in detail pixels, that is rebuilt into a single apex
INK_CUT = .3        # deepest cut below the apex, as a share of that bevel length
INK_QUANTUM = 1.    # bow, in working pixels, that the pixel lattice can hide at one joint
ROUND_BEND = .67    # handle reach of an evenly curved bend, close to a circular arc


def _line(a, b):
    return (a, a.copy(), b.copy(), b)


def _is_line(curve):
    return np.allclose(curve[0], curve[1]) and np.allclose(curve[2], curve[3])


def _ink_polygon_curves(polygon, closed, tightness):
    """Turn a polygon into straight runs, crisp corners and tangent-continuous bends.

    As in Potrace, the curve touches every polygon edge at its midpoint. A vertex whose
    tightness reaches one stays a corner; any other is replaced by a Bézier whose handles
    reach that share of the way along the adjoining half-edges.
    """
    count = len(polygon)
    pieces = []
    if closed:
        triples = [(j-1, j, (j+1) % count) for j in range(count)]
        triples = triples[1:] + triples[:1]
    else:
        triples = [(j-1, j, j+1) for j in range(1, count-1)]
        pieces.append(_line(polygon[0], (polygon[0]+polygon[1])/2))
    for i, j, k in triples:
        before, vertex, after = polygon[i], polygon[j], polygon[k]
        start, end = (before+vertex)/2, (vertex+after)/2
        if tightness[j] >= 1:
            pieces.extend((_line(start, vertex), _line(vertex, end)))
        else:
            pieces.append((start, start + (vertex-start)*tightness[j], end + (vertex-end)*tightness[j], end))
    if not closed:
        pieces.append(_line((polygon[-2]+polygon[-1])/2, polygon[-1]))

    # Join the two halves of an edge that runs straight from one corner to the next.
    def continues(first, second):
        if not (_is_line(first) and _is_line(second)):
            return False
        along, extra = first[3] - first[0], second[3] - second[0]
        return abs(along[0]*extra[1] - along[1]*extra[0]) < 1e-9 and np.dot(along, extra) > 0

    curves = []
    for piece in pieces:
        if curves and continues(curves[-1], piece):
            curves[-1] = _line(curves[-1][0], piece[3])
        else:
            curves.append(piece)
    if closed and len(curves) > 2 and continues(curves[-1], curves[0]):
        # A closed contour may start anywhere, so let it start at the corner instead.
        curves[0] = _line(curves.pop()[0], curves[0][3])
    return curves


def _polygon_indices(points, polygon, closed):
    """Chain index of every polygon vertex, in traversal order; None if that is ambiguous."""
    indices = cKDTree(points).query(polygon)[1]
    if closed:
        indices = np.roll(indices, -int(np.argmin(indices)))
    else:
        indices[0], indices[-1] = 0, len(points)-1
    return indices if np.all(np.diff(indices) > 0) else None


def _run_shape(run):
    """Fitted line of a run, its chord length, and how far the run bows away from that line.

    The bow is the sagitta of the best parabola through the middle of the run, scaled to the
    whole run. A parabola ignores the sawtooth of a digital straight line. Runs too short to
    measure report ``None``.
    """
    chord = run[-1] - run[0]
    length = np.linalg.norm(chord)
    trim = len(run)//8
    middle = run[trim:len(run)-trim]
    if len(middle) < 5 or length < 1e-9:
        return ((run[0]+run[-1])/2, chord/max(1e-12, length)), length, None
    center = middle.mean(axis=0)
    offsets = middle - center
    along = np.linalg.eigh(offsets.T @ offsets)[1][:, 1]
    if along @ chord < 0:
        along = -along
    position = offsets @ along
    half = max(1e-9, np.abs(position).max())
    bend = np.polyfit(position/half, offsets[:, 0]*along[1] - offsets[:, 1]*along[0], 2)[0]
    return (center, along), length, abs(bend)*(length/(2*half))**2


def _tightness(first, second):
    """Decide whether the turn between two runs happens along them or at their joint.

    Runs of an evenly curved bend bow by an eighth of chord times turn. If the measured bow
    is only a small share of that, the runs are straight and the joint is a corner.
    """
    (_, along_a), length_a, bow_a = first
    (_, along_b), length_b, bow_b = second
    even = np.arccos(np.clip(along_a @ along_b, -1, 1))/8
    expected = (length_a + length_b)*even
    if expected < 1e-9:
        return ROUND_BEND
    measured = sum(length*even if bow is None else bow for length, bow in ((length_a, bow_a), (length_b, bow_b)))
    share = measured/expected
    # A long flat lattice run can belong to a gentle arc, so allow for what it may hide.
    if measured + INK_QUANTUM <= INK_CORNER*expected:
        return 1.
    return ROUND_BEND + (.95 - ROUND_BEND)*float(np.clip((1 - share)/(1 - INK_CORNER), 0, 1))


def _meet(first, second, near):
    """Intersection of two fitted lines; nearly parallel ones only pull ``near`` onto them."""
    (center_a, along_a), (center_b, along_b) = first, second
    cross = along_a[0]*along_b[1] - along_a[1]*along_b[0]
    if abs(cross) < .14:
        return (center_a + along_a*np.dot(near-center_a, along_a) +
                center_b + along_b*np.dot(near-center_b, along_b))/2
    gap = center_b - center_a
    return center_a + along_a*((gap[0]*along_b[1] - gap[1]*along_b[0])/cross)


def _adjusted_polygon(points, indices, closed, limit, short):
    """Move polygon vertices to the intersections of lines fitted to the adjoining runs.

    Vertices picked on the contour sit on the outside of every bend, so a curve inscribed in
    that polygon shrinks convex shapes and rounds corners the raster had already softened.
    Fitted lines pass through the middle of their runs, and where two long runs are joined
    only by a short bevel they meet at the apex the bevel cut off. Returns the vertices and
    the tightness of the bend at each of them.
    """
    count = len(indices)
    edges = count if closed else count-1

    def chain(first, last):
        return points[first:last+1] if last > first else np.vstack((points[first:], points[:last+1]))

    shapes = [_run_shape(chain(indices[edge], indices[(edge+1) % count])) for edge in range(edges)]
    lines = [shape[0] for shape in shapes]
    bevel = [shapes[edge][1] < short and (closed or 0 < edge < edges-1) for edge in range(edges)]
    keep = [True]*edges
    first_long = next((edge for edge in range(edges) if not bevel[edge]), None)
    if first_long is not None:
        group = []
        for edge in [(first_long + step) % edges for step in range(edges)] + [None]:
            if edge is not None and bevel[edge]:
                group.append(edge)
                continue
            before, after = (group[0]-1) % edges if group else 0, (group[-1]+1) % edges if group else 0
            if group and before != after and keep.count(True) - len(group) >= (3 if closed else 2):
                cross = lines[before][1][0]*lines[after][1][1] - lines[before][1][1]*lines[after][1][0]
                arc = chain(indices[group[0]], indices[(group[-1]+1) % count])
                apex = _meet(lines[before], lines[after], arc.mean(axis=0))
                # Only a small cut between two clearly different directions is a softened corner.
                if (abs(cross) >= .42 and np.linalg.norm(arc[-1] - arc[0]) <= 2*short
                        and np.linalg.norm(arc - apex, axis=1).min() <= INK_CUT*short):
                    for removed in group:
                        keep[removed] = False
            group = []
    kept = [edge for edge in range(edges) if keep[edge]]
    joints = list(zip(kept, kept[1:] + kept[:1])) if closed else list(zip(kept, kept[1:]))
    vertices = [] if closed else [points[indices[0]]]
    tightness = [] if closed else [1.]
    for arriving, leaving in joints:
        tail, head = points[indices[(arriving+1) % count]], points[indices[leaving]]
        anchor = (tail + head)/2
        move = _meet(lines[arriving], lines[leaving], anchor) - anchor
        reach = limit + np.linalg.norm(tail - head)/2
        vertices.append(anchor + move*min(1., reach/max(np.linalg.norm(move), 1e-12)))
        tightness.append(_tightness(shapes[arriving], shapes[leaving]))
    if not closed:
        vertices.append(points[indices[-1]])
        tightness.append(1.)
    return np.asarray(vertices, float), tightness


def fit_ink_curve(raw, simplify, smooth, max_error, scale=1.):
    """Fit a supersampled contour as confident straight runs, corners and smooth bends.

    ``scale`` is the number of working pixels that count as one pixel of contour detail.
    """
    raw = np.asarray(raw, float)
    closed = np.array_equal(raw[0], raw[-1])
    short_axis = min(np.ptp(raw, axis=0))
    thin = short_axis == 0 or (closed and short_axis <= 1)
    if len(raw) < 8 or thin or not (simplify and smooth and max_error):
        return fit_shared_curve(raw, simplify, smooth, max_error)
    core = raw[:-1] if closed else raw
    # Remove only the lattice staircase; the polygon decides what else is noise.
    filtered = gaussian_filter1d(core, sigma=1.5, axis=0, mode='wrap' if closed else 'nearest')
    points = core + (filtered - core)*smooth
    if not closed:
        points[0], points[-1] = raw[0], raw[-1]
    source_tree = cKDTree(raw)
    tolerance = min(simplify, max_error)
    for attempt, epsilon in enumerate((tolerance*INK_EPSILON, tolerance*INK_EPSILON*.55)):
        polygon = cv2.approxPolyDP(points.astype(np.float32), epsilon, closed).reshape(-1, 2).astype(float)
        if len(polygon) < (3 if closed else 2):
            continue
        tightness = [ROUND_BEND]*len(polygon)
        indices = _polygon_indices(points, polygon, closed)
        if indices is not None and len(polygon) > 2:
            polygon, tightness = _adjusted_polygon(points, indices, closed, limit=2*epsilon, short=INK_BEVEL*scale)
        if not closed:
            polygon[0], polygon[-1] = raw[0], raw[-1]
        curves = ([_line(polygon[0], polygon[1])] if len(polygon) == 2
                  else _ink_polygon_curves(polygon, closed, tightness))
        if _within_error(raw, source_tree, curves, max_error):
            return curves, attempt
    return fit_poster_curve(raw, simplify, smooth, max_error)
