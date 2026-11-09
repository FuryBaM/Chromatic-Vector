import numpy as np
from chromatic_vector.contours import extract_edges, trace_rings, signed_area


def assert_exact_areas(labels):
    edges = extract_edges(labels)
    rings = trace_rings(edges)
    for rid in np.unique(labels):
        area = sum(signed_area(ring) for ring in rings[int(rid)])
        assert abs(area - np.count_nonzero(labels == rid)) < 1e-5
    assert sum(len(rings[int(rid)]) for rid in np.unique(labels)) >= 1


def test_single_solid():
    assert_exact_areas(np.zeros((10, 12), dtype=np.int32))


def test_hole():
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[2:6, 2:6] = 1
    assert_exact_areas(labels)
    rings = trace_rings(extract_edges(labels))
    assert len(rings[0]) == 2


def test_diagonal_contacts():
    labels = np.array([[0, 1], [1, 0]], dtype=np.int32)
    assert_exact_areas(labels)
    rings = trace_rings(extract_edges(labels))
    assert len(rings[0]) == 2
    assert len(rings[1]) == 2


def test_random_regions():
    rng = np.random.default_rng(0)
    for h, w in [(4, 4), (15, 20), (28, 33)]:
        labels = rng.integers(0, 5, size=(h, w), dtype=np.int32)
        assert_exact_areas(labels)
