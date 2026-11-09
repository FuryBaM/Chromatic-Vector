import numpy as np
import cv2
from PIL import Image
import pytest
from chromatic_vector.palette import connected_regions, edge_aware_refine
from chromatic_vector.corrections import apply_color_overlay
from chromatic_vector.shared import SharedBoundaryGeometry, _cubic_samples, _max_distance_from_points
from chromatic_vector.contours import extract_edges


def test_manual_rgba_overlay_and_exact_color(tmp_path):
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[..., :] = (38, 41, 42)
    seg = connected_regions(np.zeros((8, 8), dtype=np.int32), rgb)
    mask = np.zeros((8, 8, 4), dtype=np.uint8)
    mask[1:4, 2:5, :] = (240, 155, 50, 255)
    path = tmp_path / 'overlay.png'
    Image.fromarray(mask).save(path)
    result, stats = apply_color_overlay(seg, rgb, path)
    assert stats == {'corrected_pixels': 9, 'correction_colors': 1}
    np.testing.assert_array_equal(result.region_colors[result.labels[2, 3]], [240, 155, 50])
    np.testing.assert_array_equal(result.region_colors[result.labels[0, 0]], [38, 41, 42])
    assert int(result.sizes.sum()) == 64
    assert result.labels[2, 3] != result.labels[0, 0]


def test_correction_overlay_requires_exact_canvas(tmp_path):
    rgb = np.zeros((6, 8, 3), dtype=np.uint8)
    seg = connected_regions(np.zeros((6, 8), dtype=np.int32), rgb)
    path = tmp_path / 'overlay.png'
    Image.new('RGBA', (7, 6)).save(path)
    with pytest.raises(ValueError, match='size'):
        apply_color_overlay(seg, rgb, path)


def test_edge_refinement_is_deterministic_and_preserves_strong_edges():
    rgb = np.zeros((22, 22, 3), dtype=np.uint8)
    rgb[:, 11:] = (235, 225, 205)
    ids = np.zeros((22, 22), dtype=np.int32)
    ids[:, 11:] = 1
    ids[4, 4] = 1  # erroneous isolated palette assignment
    fixed_a = edge_aware_refine(rgb, ids, passes=2)
    fixed_b = edge_aware_refine(rgb, ids, passes=2)
    assert np.array_equal(fixed_a, fixed_b)
    assert fixed_a[4, 4] == 0
    assert np.all(fixed_a[:, :11] == 0)
    assert np.all(fixed_a[:, 11:] == 1)


def test_error_bounded_shared_curve_preserves_both_sides():
    labels = np.zeros((38, 45), dtype=np.int32)
    labels[8:30, 8:25] = 1
    labels[17:20, 25:40] = 1
    graph = SharedBoundaryGeometry(extract_edges(labels), simplify=2.8,
                                   smooth=.95, max_error=.65)
    assert graph.geometry_corrections >= 0
    for chain in graph.chains:
        samples = _cubic_samples(chain.cubics)
        raw = np.asarray(chain.vertices, dtype=np.float64)
        assert _max_distance_from_points(raw, samples) < .66
        assert _max_distance_from_points(samples, raw) < .66
