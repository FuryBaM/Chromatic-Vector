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
