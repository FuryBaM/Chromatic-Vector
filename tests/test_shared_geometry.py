from xml.etree import ElementTree as ET
import numpy as np
import pytest
from chromatic_vector.contours import extract_edges, trace_rings
from chromatic_vector.shared import SharedBoundaryGeometry, adaptive_filter_py
from chromatic_vector.native import filter_native_polyline
from chromatic_vector.svg import write_svg


def test_shared_topology_and_svg(tmp_path):
    samples = [
        np.zeros((10, 12), dtype=np.int32),
        np.array([[0, 1], [1, 0]], dtype=np.int32),
        np.pad(np.ones((5, 5), dtype=np.int32), 2),
    ]
    for seed in range(5):
        samples.append(np.random.default_rng(seed).integers(0, 5, (20, 27), dtype=np.int32))
    for labels in samples:
        edges = extract_edges(labels)
        rings = trace_rings(edges)
        geometry = SharedBoundaryGeometry(edges, simplify=.75, smooth=.8)
        paths = geometry.paths(rings)
        assert sum(len(p) for p in paths.values()) == sum(map(len, rings.values()))
        assert geometry.original_edge_count >= geometry.curve_count
        for key, (chain_id, edge_index) in geometry.lookup.items():
            chain = geometry.chains[chain_id]
            assert key == tuple(sorted((chain.vertices[edge_index], chain.vertices[edge_index+1])))
        filename = tmp_path / 'test.svg'
        palette = np.tile(np.array([[150, 50, 120]], dtype=np.uint8), (int(labels.max())+1,1))
        write_svg(filename, labels.shape[1], labels.shape[0], rings, palette, shared_paths=paths)
        doc = ET.parse(filename)
        ns = {'s': 'http://www.w3.org/2000/svg'}
        assert len(doc.findall('.//s:path', ns)) == len(rings)
        assert not doc.findall('.//s:image', ns)


def test_canonical_cubic_is_reversible():
    labels = np.zeros((30, 30), dtype=np.int32)
    labels[5:25, 8:22] = 1
    edges = extract_edges(labels)
    graph = SharedBoundaryGeometry(edges, simplify=1, smooth=.95)
    for chain in graph.chains:
        forward = chain.cubics
        reversed_cmd = [graph._command(c, True) for c in reversed(forward)]
        assert len(reversed_cmd) == len(forward)
        for (a, c1, c2, b), opposite in zip(reversed(forward), reversed_cmd):
            # Same coordinates/control points, only traversal direction reversed.
            assert opposite == graph._command((b, c2, c1, a), False)
    # The closed path is traced in both regions but refers to the same chain.
    shared = [chain for chain in graph.chains if chain.paired_regions == (0, 1)]
    assert len(shared) == 1 and shared[0].closed


def test_native_filter_matches_reference():
    rng = np.random.default_rng(7)
    for closed in (False, True):
        for count in (3, 5, 30, 130):
            points = rng.uniform(-5, 15, (count, 2)).astype(np.float32)
            output = filter_native_polyline(points, closed, .9, .55, .65)
            if output is None:
                pytest.skip('Native C++ library not built')
            reference = adaptive_filter_py(points, closed, .9, .55, .65)
            assert len(output) == len(reference)
            np.testing.assert_allclose(output, reference, rtol=1e-5, atol=1e-4)
