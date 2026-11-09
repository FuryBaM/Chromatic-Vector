"""Visual geometry regressions: a vector must stay smooth when rendered larger."""
from io import BytesIO
import json
import cairosvg
import numpy as np
import pytest
from PIL import Image
from scipy.spatial import cKDTree
from chromatic_vector.cli import vectorize
from chromatic_vector.contours import extract_edges
from chromatic_vector.geometry import safe_shared
from chromatic_vector.curve_fitting import fit_poster_curve, fit_shared_curve, sample_curves


def test_long_straight_edge_is_not_rejected_by_sampling_cap():
    raw=np.column_stack((np.arange(2001,dtype=float),np.zeros(2001)))
    curves,attempt=fit_shared_curve(raw,.8,.7,.8)
    assert len(curves)==1 and attempt==0
    assert np.max(np.abs(sample_curves(curves)[:,1]))==0


def test_closed_circle_has_curves_and_continuous_anchor_tangent():
    y,x=np.indices((85,95));labels=(((x+.5-47)/29)**2+((y+.5-42)/26)**2<=1).astype(np.int32)
    graph,_=safe_shared(extract_edges(labels),simplify=1.1,smooth=.9,max_error=1.1)
    chain=next(c for c in graph.chains if c.paired_regions==(0,1))
    assert chain.closed
    assert len(chain.cubics)<len(chain.vertices)//4
    assert sum(not np.array_equal(c1,a) for a,c1,c2,b in chain.cubics)>4
    a,c1,c2,b=chain.cubics[0];last=chain.cubics[-1]
    np.testing.assert_allclose(c1-a,last[-1]-last[-2],atol=1e-9)
    samples=sample_curves(chain.cubics)
    # The analytic source ellipse provides a grid-independent shape reference.
    radius=np.sqrt(((samples[:,0]-47)/29)**2+((samples[:,1]-42)/26)**2)
    assert np.max(np.abs(radius-1))<.06


def test_poster_ellipse_uses_long_spans_with_aligned_tangents():
    y, x = np.indices((85, 95))
    labels = (((x+.5-47)/29)**2 + ((y+.5-42)/26)**2 <= 1).astype(np.int32)
    edges = extract_edges(labels)
    graph, metrics = safe_shared(edges, simplify=1.1, smooth=.9, max_error=1.1, poster=True)
    chain = next(c for c in graph.chains if c.closed and c.paired_regions == (0, 1))
    assert 3 <= len(chain.cubics) <= 6
    assert metrics['intersection_fallbacks'] == 0
    for left, right in zip(chain.cubics, chain.cubics[1:] + chain.cubics[:1]):
        incoming, outgoing = left[-1] - left[-2], right[1] - right[0]
        assert incoming @ outgoing > 0
        np.testing.assert_allclose(incoming / np.linalg.norm(incoming),
                                   outgoing / np.linalg.norm(outgoing), atol=1e-8)
    points = sample_curves(chain.cubics)
    radius = np.sqrt(((points[:, 0]-47)/29)**2 + ((points[:, 1]-42)/26)**2)
    assert np.max(np.abs(radius-1)) < .035


def test_poster_retries_a_tight_bend_without_restoring_the_staircase():
    # A raster scarf edge has two long diagonals and a tight return in between.
    steps = 'UUURURURRURURRRDRRRRRRRRRULLLLLLUURURRURUUUR'
    directions = {'U': (0, -1), 'D': (0, 1), 'L': (-1, 0), 'R': (1, 0)}
    raw = np.cumsum([(0, 0), *(directions[step] for step in steps)], axis=0)
    curves, attempt = fit_poster_curve(raw, 1.1, .9, 1.1)
    assert attempt == 1
    assert len(curves) <= 10
    samples = sample_curves(curves)
    np.testing.assert_array_equal(samples[[0, -1]], raw[[0, -1]])
    assert cKDTree(samples).query(raw)[0].max() <= 1.1


@pytest.mark.parametrize('poster', [False, True])
def test_neural_unsafe_chain_recovers_smooth_bounded_contour(monkeypatch, poster):
    from collections import Counter
    from chromatic_vector import neural

    class Context:
        def __init__(self):
            self.records = []
            self.counters = Counter()

        def score_merge_pairs(self, chains, styles):
            pass

        def report(self):
            return self.records

    def unsafe_prediction(raw, pair, context, simplify, smooth, max_error):
        if pair != (0, 1):
            return fit_shared_curve(raw, simplify, smooth, max_error)[0]
        context.records.append({'accepted': True})
        if poster:
            return fit_shared_curve(raw, 0, 0, 0)[0]  # Simple, but a visible staircase.
        points = np.array([raw[0], [58, 63], [30, 63], [58, 22], raw[0]])
        return [(a, a, b, b) for a, b in zip(points, points[1:])]

    monkeypatch.setattr(neural, 'fit_neural_chain', unsafe_prediction)
    y, x = np.indices((85, 95))
    labels = (((x+.5-47)/29)**2 + ((y+.5-42)/26)**2 <= 1).astype(np.int32)
    edges = extract_edges(labels)
    options = dict(simplify=1.1, smooth=.9, max_error=1.1, poster=poster)
    bounded, _ = safe_shared(edges, **options)
    recovered, metrics = safe_shared(edges, method='neural', neural_context=Context(), **options)
    expected = next(chain.cubics for chain in bounded.chains if chain.paired_regions == (0, 1))
    actual = next(chain.cubics for chain in recovered.chains if chain.paired_regions == (0, 1))
    np.testing.assert_array_equal(actual, expected)
    assert metrics['intersection_fallbacks'] == (0 if poster else 1)
    assert metrics['lattice_fallbacks'] == 0
    assert metrics['neural'][0]['emitted'] is False
    assert metrics['neural'][0]['final_reason'] == ('chain_smoothness' if poster else 'topology_fallback')


def test_large_opaque_render_is_antialiased_editable_and_has_no_seams(tmp_path):
    y,x=np.indices((85,95));mask=((x+.5-47)/29)**2+((y+.5-42)/26)**2<=1
    rgba=np.full((85,95,4),255,np.uint8);rgba[mask,:3]=(30,80,130)
    source=tmp_path/'circle.png';svg=tmp_path/'circle.svg';report=tmp_path/'circle.json'
    Image.fromarray(rgba).save(source)
    vectorize(source,svg,mode='illustrative',base_colors=2,target_shapes=0,detail_budget=0,
              primitive_fit='off',max_side=0,simplify=1.1,smooth=.9,max_error=1.1,shape_report=report)
    text=svg.read_text()
    assert 'crispEdges' not in text and '<image' not in text
    assert 'C ' in text
    rendered=np.array(Image.open(BytesIO(cairosvg.svg2png(url=str(svg),scale=4))).convert('RGBA'))
    assert np.all(rendered[...,3]==255)
    assert len(np.unique(rendered[...,:3].reshape(-1,3),axis=0))>50
    assert json.loads(report.read_text())['geometry']['unhandled_topology_violations']==0


def test_closed_concave_contour_retains_traversal_order():
    labels=np.zeros((70,90),np.int32);labels[8:60,8:27]=1;labels[43:60,8:76]=1
    graph,metrics=safe_shared(extract_edges(labels),simplify=.8,smooth=0,max_error=.8)
    chain=next(c for c in graph.chains if c.paired_regions==(0,1))
    assert metrics['guard_fallbacks']==0
    assert len(chain.cubics)<12
    from shapely.geometry import LineString
    assert LineString(sample_curves(chain.cubics)).is_simple
