"""One shared geometric curve per physical region boundary.

All region rings are built from references to the same canonical boundary chains,
so adjacent SVG regions reuse *identical* Bezier control points in reverse order.
This prevents independently-smoothed neighbor edges from drifting apart.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import math
import cv2
import numpy as np
from scipy.spatial import cKDTree

Point = tuple[int, int]
Edge = tuple[Point, Point]


def _fmt(value: float) -> str:
    return f"{float(value):.3f}".rstrip('0').rstrip('.') or '0'


def _xy(p: np.ndarray) -> str:
    return f"{_fmt(p[0])} {_fmt(p[1])}"


def _edge(a: Point, b: Point) -> Edge:
    return (a, b) if a < b else (b, a)


def _corner_indices(pts: np.ndarray, angle_deg: float = 54.0) -> list[int]:
    """Detect deliberately sharp turns, not every one-pixel raster staircase."""
    n = len(pts)
    if n < 5:
        return []
    result = []
    for j in range(2, n-2):
        before = pts[j] - pts[max(0, j-3)]
        after = pts[min(n-1, j+3)] - pts[j]
        # Make sure both flanks are substantial. Short jaggies stay smoothable.
        length_a, length_b = np.linalg.norm(before), np.linalg.norm(after)
        if min(length_a, length_b) < 2.2:
            continue
        cos = float(np.dot(before, after) / (length_a * length_b))
        if cos < math.cos(math.radians(angle_deg)):
            result.append(j)
    # Merge close forced points to avoid overfitting a cluster of square pixels.
    chosen = []
    for j in result:
        if not chosen or j - chosen[-1] >= 4:
            chosen.append(j)
    return chosen


def adaptive_filter_py(points: np.ndarray, closed: bool, drop_tolerance: float,
                       smooth_strength: float, sharp_turn_threshold: float = 0.65) -> np.ndarray:
    """Reference of the game's adaptive skip/relax filter on a shared chain."""
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n <= 2:
        return pts.copy()
    out = [pts[0].copy()]
    anchor = pts[0].copy()
    stop = n if closed else n-1
    for i in range(1, stop):
        current, next_p = pts[i], pts[(i+1)%n]
        va, vb = current-anchor, next_p-current
        da, db = np.linalg.norm(va), np.linalg.norm(vb)
        turn = 0.0 if min(da,db) < 1e-6 else 1.0-float(np.clip(np.dot(va,vb)/(da*db),-1,1))
        chord = next_p-anchor
        dc = np.linalg.norm(chord)
        deviation = np.linalg.norm(current-anchor) if dc < 1e-6 else abs(float(chord[0]*(current-anchor)[1]-chord[1]*(current-anchor)[0]))/dc
        if turn < sharp_turn_threshold and deviation <= drop_tolerance:
            continue
        if turn < sharp_turn_threshold and smooth_strength > 0 and da+db > 1e-6:
            target = anchor + (next_p-anchor)*(da/(da+db))
            factor = smooth_strength*(1.0-min(1.0,turn/max(sharp_turn_threshold,1e-6)))
            anchor = current + (target-current)*factor
        else:
            anchor = current.copy()
        out.append(anchor.copy())
    if not closed:
        out.append(pts[-1].copy())
    return np.array(out, dtype=np.float64)


def game_adaptive(points: np.ndarray, closed: bool, epsilon: float, strength: float) -> np.ndarray:
    """Native port when compiled, deterministic Python fallback otherwise."""
    from .native import filter_native_polyline
    raw = points[:-1] if closed else points
    result = filter_native_polyline(raw, closed, epsilon, strength)
    if result is None:
        result = adaptive_filter_py(raw, closed, epsilon, strength)
    min_count = 3 if closed else 2
    if len(result) < min_count:
        result = np.array(raw, dtype=np.float64)
    # The source game's adaptive filter can move vertices. Keep the original
    # region-junction endpoints fixed; for cyclic boundaries pin the first point.
    result[0] = raw[0]
    if closed:
        return np.vstack([result, result[0]])
    result[-1] = raw[-1]
    return result

def simplify_chain(points: np.ndarray, epsilon: float, closed: bool,
                   simplifier: str = "game") -> np.ndarray:
    """Preserve endpoints / hard corners while discarding short staircase noise."""
    if epsilon <= 0 or len(points) < 4:
        return points.astype(np.float64, copy=True)
    pts = np.asarray(points, dtype=np.float64)
    if simplifier == "game":
        pts = game_adaptive(pts, closed, epsilon, 0.55)
    elif simplifier != "rdp":
        raise ValueError("simplifier must be game or rdp")
    if closed:
        core = cv2.approxPolyDP(pts[:-1].astype(np.float32), epsilon, True).reshape(-1, 2)
        if len(core) < 3:
            return pts.copy()
        # A closed chain has one canonical anchor shared by both regions.
        anchor = pts[0]
        if not np.any(np.all(np.isclose(core, anchor), axis=1)):
            # Force the canonical anchor, otherwise reconstructing the original
            # edge chain would start at a point absent from fitted geometry.
            core = np.vstack([anchor, core])
        aidx = int(np.argmin(np.sum((core-anchor)**2, axis=1)))
        core = np.roll(core, -aidx, axis=0)
        return np.vstack([core, core[:1]]).astype(np.float64)
    forced = [0, *_corner_indices(pts), len(pts)-1]
    forced = sorted(set(forced))
    groups = []
    for a, b in zip(forced, forced[1:]):
        section = cv2.approxPolyDP(pts[a:b+1].astype(np.float32), epsilon, False).reshape(-1, 2)
        if len(section) < 2:
            section = pts[[a, b]]
        if groups:
            section = section[1:]
        groups.extend(section)
    return np.asarray(groups, dtype=np.float64)


def fit_cubics(points: np.ndarray, smooth: float, closed: bool,
               preserve_angle_deg: float = 57.0) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Return canonical cubic segments; reverse them exactly for opposite region."""
    points = np.asarray(points, dtype=np.float64)
    n = len(points)-1
    if n <= 0:
        return []
    # Tangents are derivatives, not control points. Setting endpoint tangents to
    # zero pins shared multi-region junctions to exactly the same locations.
    tangents = np.zeros_like(points)
    if smooth > 0 and n >= 2:
        for i in range(n if closed else 1, n if closed else n):
            prev = points[i-1] if i > 0 else points[n-1]
            nxt = points[i+1]
            va, vb = points[i]-prev, nxt-points[i]
            da, db = np.linalg.norm(va), np.linalg.norm(vb)
            if da < 1e-8 or db < 1e-8:
                continue
            ca = float(np.dot(va, vb)/(da*db))
            # Keep meaningful long-segment corners, smooth small pixel steps.
            if min(da, db) > 2.2 and ca < math.cos(math.radians(preserve_angle_deg)):
                continue
            # Non-uniform Catmull-Rom style derivative, limited to avoid big overshoots.
            t = smooth * (nxt-prev)*0.5
            max_mag = min(da, db)*1.5
            mag = np.linalg.norm(t)
            if mag > max_mag:
                t *= max_mag/mag
            tangents[i] = t
        if closed:
            tangents[n] = tangents[0]
    cubics = []
    for i in range(n):
        a, b = points[i], points[i+1]
        c1 = a + tangents[i]/3
        c2 = b - tangents[i+1]/3
        cubics.append((a, c1, c2, b))
    return cubics


def _cubic_samples(cubics, spacing: float = 1.0) -> np.ndarray:
    """Approximate cubic geometry densely enough for pixel-scale error checks."""
    chunks = []
    for a, c1, c2, b in cubics:
        # Tangent lengths contribute to curvature and determine sample density.
        control_length = (np.linalg.norm(c1-a) + np.linalg.norm(c2-c1) +
                          np.linalg.norm(b-c2))
        count = min(48, max(2, int(math.ceil(control_length / spacing))))
        t = np.linspace(0, 1, count+1, dtype=np.float64)[1:, None]
        u = 1-t
        pieces = u**3*a + 3*u**2*t*c1 + 3*u*t**2*c2 + t**3*b
        chunks.append(pieces)
    if not chunks:
        return np.zeros((0, 2), dtype=np.float64)
    return np.vstack((cubics[0][0], *chunks))


def _max_distance_from_points(points: np.ndarray, polyline: np.ndarray,
                              stop_at: float = float('inf')) -> float:
    """Max distance to a piecewise-linear curve, vectorized in fixed batches."""
    if len(polyline) < 2:
        return float('inf')
    a = polyline[:-1]
    v = polyline[1:] - a
    norm = np.maximum(np.sum(v*v, axis=-1), 1e-14)
    worst = 0.0
    # Avoid allocating quadratic arrays for very long boundary chains.
    for offset in range(0, len(points), 32):
        p = points[offset:offset+32]
        delta = p[:, None, :] - a[None, :, :]
        t = np.clip(np.einsum('mni,ni->mn', delta, v)/norm[None,:], 0, 1)
        closest = a[None,:,:] + t[:,:,None]*v[None,:,:]
        distances = np.sqrt(np.sum((p[:,None,:]-closest)**2, axis=-1))
        worst = max(worst, float(distances.min(axis=1).max()))
        if worst > stop_at:
            break
    return worst


def _guarded_fitting(raw: np.ndarray, simplify: float, smooth: float,
                     simplifier: str, max_error: float):
    """Constrain both raw-to-vector and vector-to-raw geometric error.

    A rejected fit is retried with less simplification and curvature. The
    topology-preserving last resort is the original polygonal pixel boundary.
    """
    closed = bool(np.array_equal(raw[0], raw[-1]))
    if max_error <= 0:
        pts = simplify_chain(raw, simplify, closed, simplifier)
        return fit_cubics(pts, smooth, closed), 0
    # Don't destroy details when the whole shape is only a couple of pixels thick.
    bounds = raw.max(axis=0)-raw.min(axis=0)
    epsilon = min(simplify, max_error)
    if min(bounds) <= 3 or len(raw) <= 5:
        epsilon = 0.0
    tries = [(epsilon, smooth), (epsilon*0.5, smooth*0.4), (0, smooth*0.25), (0, 0)]
    for attempt, (eps, tension) in enumerate(tries):
        reduced = simplify_chain(raw, eps, closed, simplifier)
        curves = fit_cubics(reduced, tension, closed)
        if attempt == len(tries)-1:
            return curves, attempt
        samples = _cubic_samples(curves, spacing=0.75)
        # Conservative sampled upper bounds: the nearest sampled point or
        # original grid vertex is at least as far as the exact polyline.
        # KD-trees keep the check O(N log N), even for thousand-pixel chains.
        backward = float(cKDTree(samples).query(raw, k=1)[0].max())
        if backward > max_error:
            continue
        forward = float(cKDTree(raw).query(samples, k=1)[0].max())
        if forward <= max_error:
            return curves, attempt
    raise AssertionError('Unreachable geometry guard fallback')


@dataclass
class Chain:
    vertices: list[Point]
    paired_regions: tuple[int, ...]
    closed: bool
    cubics: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]


class SharedBoundaryGeometry:
    def __init__(self, oriented_edges: np.ndarray, simplify: float = 0.75,
                 smooth: float = 0.7, simplifier: str = "game",
                 max_error: float = 0.0,
                 region_styles: dict[int, tuple[float, float]] | None = None,
                 chain_fitter=None):
        if simplify < 0 or not (0 <= smooth <= 1):
            raise ValueError('simplify must be >= 0 and smooth in [0, 1]')
        regions: dict[Edge, set[int]] = defaultdict(set)
        for rid, x0, y0, x1, y1 in oriented_edges:
            a, b = (int(x0), int(y0)), (int(x1), int(y1))
            regions[_edge(a, b)].add(int(rid))
        physical = sorted(regions)
        adjacent = defaultdict(list)
        for idx, (a, b) in enumerate(physical):
            adjacent[a].append(idx)
            adjacent[b].append(idx)
        pairs = [tuple(sorted(regions[e])) for e in physical]
        if any(len(p) > 2 for p in pairs):
            raise ValueError('More than two regions touch one physical edge')

        def passes(v: Point, pair: tuple[int, ...]) -> bool:
            return len(adjacent[v]) == 2 and all(pairs[i] == pair for i in adjacent[v])

        chains: list[Chain] = []
        correction_counts = [0, 0, 0, 0]
        lookup: dict[Edge, tuple[int, int]] = {}
        visited: set[int] = set()
        # Trace junction-terminated paths first; only then isolated cycles.
        # Otherwise a first edge picked in the middle of an open chain
        # would split it and leave a visited continuation.
        traversal_order = sorted(range(len(physical)),
            key=lambda i: (passes(physical[i][0], pairs[i]) and
                           passes(physical[i][1], pairs[i]), i))
        for first in traversal_order:
            if first in visited:
                continue
            aa, bb = physical[first]
            pair = pairs[first]
            # On closed chains both ends are pass-throughs. Use canonical anchor.
            if not passes(aa, pair):
                start = aa
            elif not passes(bb, pair):
                start = bb
            else:
                start = min(aa, bb)
            verts = [start]
            current = first
            while True:
                if current in visited:
                    raise ValueError('Boundary chain revisited unexpectedly')
                visited.add(current)
                a, b = physical[current]
                nxt = b if verts[-1] == a else a
                verts.append(nxt)
                if not passes(nxt, pair) or nxt == start:
                    break
                options = [i for i in adjacent[nxt] if i != current and i not in visited]
                if len(options) != 1:
                    raise ValueError('Invalid continuation in boundary graph')
                current = options[0]
            closed = verts[0] == verts[-1]
            raw = np.array(verts, dtype=np.float64)
            # A shared physical boundary is fitted exactly once. Both sides ask
            # for a style, but the stricter one wins to avoid seams between layers.
            eps, tension = simplify, smooth
            if region_styles:
                for rid in pair:
                    if rid in region_styles:
                        local_eps, local_smooth = region_styles[rid]
                        eps, tension = min(eps, local_eps), min(tension, local_smooth)
            cubics, correction_level = (chain_fitter(raw, eps, tension, max_error) if chain_fitter is not None else
                                        _guarded_fitting(raw, eps, tension, simplifier, max_error))
            correction_counts[correction_level] += 1
            chain_id = len(chains)
            chains.append(Chain(verts, pair, closed, cubics))
            for k in range(len(verts)-1):
                lookup[_edge(verts[k], verts[k+1])] = (chain_id, k)
        if len(visited) != len(physical):
            raise ValueError('Not all physical boundaries were assigned')
        self.lookup = lookup
        self.chains = chains
        self.anchors = {chain.vertices[0] for chain in chains}
        self.anchors.update(chain.vertices[-1] for chain in chains)
        self.original_edge_count = len(physical)
        self.curve_count = sum(len(c.cubics) for c in chains)
        self.geometry_corrections = correction_counts[1] + correction_counts[2] + correction_counts[3]
        self.geometry_fallbacks = correction_counts[3]
        self.max_error = max_error

    def _command(self, bezier, reverse: bool) -> str:
        a, c1, c2, b = bezier
        if reverse:
            a, c1, c2, b = b, c2, c1, a
        if np.allclose(c1, a) and np.allclose(c2, b):
            return 'L ' + _xy(b)
        return 'C ' + _xy(c1) + ' ' + _xy(c2) + ' ' + _xy(b)

    def ring_path(self, points: np.ndarray) -> str:
        vertices = [tuple(map(int, p)) for p in points]
        n = len(vertices)
        if n < 3:
            return ''
        # A traced ring can start in the middle of a common chain; rotate to a
        # global junction/anchor so each chain is appended in a complete piece.
        starts = [i for i, v in enumerate(vertices) if v in self.anchors]
        if not starts:
            raise ValueError('No shared-chain anchor found in ring')
        shift = starts[0]
        vertices = vertices[shift:] + vertices[:shift]
        first_id,k=self.lookup[_edge(vertices[0],vertices[1])]
        first=self.chains[first_id]
        forward=first.vertices[k]==vertices[0]
        actual_start=first.cubics[0][0] if forward else first.cubics[-1][-1]
        # An isolated cycle has no junction: Chaikin may move its arbitrary
        # anchor. Start at actual fitted geometry on both opposing traversals.
        commands = ['M ' + _xy(actual_start)]
        i = 0
        while i < n:
            a, b = vertices[i], vertices[(i+1) % n]
            chain_id, k = self.lookup[_edge(a, b)]
            chain = self.chains[chain_id]
            forward = chain.vertices[k] == a and chain.vertices[k+1] == b
            if forward and k != 0:
                raise ValueError('Ring began inside an unfinished shared chain')
            if not forward and k != len(chain.vertices)-2:
                raise ValueError('Ring began inside an unfinished reverse chain')
            run = len(chain.vertices)-1
            actual_end = vertices[(i+run) % n]
            expected_end = chain.vertices[-1] if forward else chain.vertices[0]
            if actual_end != expected_end:
                raise ValueError('Ring crosses a shared boundary partway through')
            if forward:
                commands.extend(self._command(c, False) for c in chain.cubics)
            else:
                commands.extend(self._command(c, True) for c in reversed(chain.cubics))
            i += run
        if i != n:
            raise ValueError('Ring chain accounting mismatch')
        commands.append('Z')
        return ' '.join(commands)

    def paths(self, rings: dict[int, list[np.ndarray]]) -> dict[int, list[str]]:
        return {rid: [self.ring_path(r) for r in loops] for rid, loops in rings.items()}
