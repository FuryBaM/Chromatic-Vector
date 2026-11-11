"""Supersampled poster path: edge restoration, faint-seam merging, ink contours, export."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image
from scipy.spatial import cKDTree

from chromatic_vector.base_segmentation import merge_planes
from chromatic_vector.cli import vectorize
from chromatic_vector.curve_fitting import fit_ink_curve, sample_curves
import chromatic_vector.enhance as enhance
from chromatic_vector.enhance import (auto_supersample, coherence_shock, ink_lines, lift_blacks,
                                      restore_blacks, supersample)
from chromatic_vector.flatten import importance_map
from chromatic_vector.hierarchy import VectorDocument, VectorShape, layer_and_compact
from chromatic_vector.metrics import render_svg
from chromatic_vector.modes import IllustrativeConfig
from chromatic_vector.palette import connected_regions
from chromatic_vector.stacking import stack_poster, straighten_recesses

REFERENCE = Path(__file__).resolve().parents[1] / 'references' / 'head_shoulder_crop.png'
SVG = '{http://www.w3.org/2000/svg}'


def test_stacked_poster_shortcuts_only_a_covered_recess():
    path = 'M 0 0 L 8 0 L 8 4 L 12 4 L 12 0 L 20 0 L 20 20 L 0 20 Z'
    simpler, removed = straighten_recesses(path, 1)
    assert removed == 4 and 'L 20 0' in simpler
    lower = VectorShape('lower', [0], 'primary', 'base_planes', (35, 40, 45, 255),
                        f'fill-rule="evenodd" d="{path}"', 'path', (0, 0, 20, 20), 384)
    upper = VectorShape('upper', [1], 'detail', 'details', (220, 170, 65, 255),
                        'x="8" y="0" width="4" height="4"', 'rect', (8, 0, 12, 4), 16, draw_order=1)
    labels = np.zeros((20, 20), np.int32)
    labels[:4, 8:12] = 1
    doc = VectorDocument(20, 20, [lower, upper], [], {})
    before = render_svg(doc.svg().encode())
    stacked, saved, after = stack_poster(doc, labels, scale=1, max_error=1, working_baseline=before)
    assert saved == removed
    np.testing.assert_array_equal(after, before)
    np.testing.assert_array_equal(render_svg(stacked.svg().encode()), before)
    uncovered = VectorDocument(20, 20, [lower], [], {})
    assert stack_poster(uncovered, labels, 1, 1, render_svg(uncovered.svg().encode()))[1] == 0


def test_stacked_poster_fills_only_holes_covered_by_the_actual_upper_shapes():
    outline = 'M 0 0 L 28 0 L 28 20 L 0 20 Z'
    covered = 'M 4 5 L 4 11 L 10 11 L 10 5 Z'
    partly_covered = 'M 17 5 L 17 11 L 24 11 L 24 5 Z'
    lower = VectorShape('lower', [0], 'secondary', 'shadows', (35, 40, 45, 255),
                        f'fill-rule="evenodd" d="{outline} {covered} {partly_covered}"',
                        'path', (0, 0, 28, 20), 482, draw_order=2, holes=2)
    covers = [VectorShape(f'detail-{rid}', [rid], 'detail', 'details', (220, 170, 65, 255),
                          f'x="{x}" y="5" width="3" height="6"', 'rect', (x, 5, x+3, 11),
                          18, draw_order=order)
              for rid, x, order in ((1, 4, 0), (2, 7, 1), (3, 17, 3))]
    labels = np.zeros((20, 28), np.int32)
    labels[5:11, 4:7] = 1
    labels[5:11, 7:10] = 2
    labels[5:11, 17:24] = 3
    doc = VectorDocument(28, 20, [lower, *covers], [], {})
    before = render_svg(doc.svg().encode())
    stacked, saved, after = stack_poster(doc, labels, 1, 1, before)

    assert saved == 3
    assert stacked.shapes[0].holes == 1
    assert stacked.shapes[0].geometry == f'fill-rule="evenodd" d="{outline} {partly_covered}"'
    assert [shape.uid for shape in stacked.shapes[1:]] == [shape.uid for shape in covers]
    np.testing.assert_array_equal(after, before)
    np.testing.assert_array_equal(render_svg(stacked.svg().encode()), before)


def test_layer_compaction_combines_disjoint_fills_and_preserves_paint_order():
    shapes = []
    for index, (bounds, layer, rgb) in enumerate([
            ((0, 0, 80, 40), 'background', (15, 16, 18)),
            ((5, 5, 20, 20), 'details', (221, 178, 76)),
            ((13, 13, 29, 26), 'shadows', (82, 109, 167)),
            ((53, 6, 70, 20), 'accents', (221, 178, 76))]):
        x0, y0, x1, y1 = bounds
        vertices = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        path = ' L '.join(f'{x} {y}' for x, y in vertices)
        shapes.append(VectorShape(f'shape-{index}', [index], 'detail', layer, (*rgb, 255),
                                  f'fill-rule="evenodd" d="M {path} Z"', 'path', bounds,
                                  (x1-x0)*(y1-y0), draw_order=index))
    doc = VectorDocument(80, 40, shapes, [], {'seam_underpaint': True}, display_size=(20, 10))
    packed = layer_and_compact(doc, padding=2)
    assert len(packed.shapes) == 3
    assert packed.shapes[0].layer == 'background'
    assert set(packed.shapes[1].source_region_ids) == {1, 3}
    for display in (False, True):
        np.testing.assert_array_equal(render_svg(packed.svg(display=display).encode()),
                                      render_svg(doc.svg(display=display).encode()))


def lattice_chain(polygon):
    """Unit-step boundary of a filled polygon mask, as a closed chain of lattice vertices."""
    mask = np.zeros((140, 140), np.uint8)
    cv2.fillPoly(mask, [np.asarray(polygon, np.int32)], 1)
    from chromatic_vector.contours import extract_edges
    from chromatic_vector.geometry import safe_shared
    graph, _ = safe_shared(extract_edges(mask.astype(np.int32)), 0, 0, 0)
    chain = next(item for item in graph.chains if item.paired_regions == (0, 1))
    return np.asarray(chain.vertices, float)


def test_auto_supersample_targets_small_sources_only():
    assert auto_supersample(230, 245) == 4
    assert auto_supersample(500, 320) == 2
    assert auto_supersample(1600, 900) == 1
    assert auto_supersample(1, 1) == 4


def test_shock_filter_collapses_a_blurred_edge_without_overshoot_or_flat_noise():
    step = np.full((48, 96, 3), 40, np.uint8)
    step[:, 48:] = 190
    blurred = cv2.GaussianBlur(step, (0, 0), 4)
    rng = np.random.default_rng(5)
    # Faint texture far from the edge must not grow into structure.
    blurred[:, :24] = np.clip(blurred[:, :24].astype(int) + rng.integers(-1, 2, (48, 24, 1)), 0, 255)
    sharp = coherence_shock(blurred, 4)
    row = sharp[24, :, 0].astype(int)
    before = blurred[24, :, 0].astype(int)
    assert np.abs(np.diff(row)).max() > 2.5*np.abs(np.diff(before)).max()
    assert row.min() >= before.min() and row.max() <= before.max()
    assert np.count_nonzero((row > 60) & (row < 170)) <= .5*np.count_nonzero((before > 60) & (before < 170))
    np.testing.assert_array_equal(sharp[:, :16], blurred[:, :16])


def test_supersample_keeps_tones_inside_the_source_range():
    rgb = np.full((12, 16, 3), 11, np.uint8)
    rgb[3:9, 5:11] = (150, 140, 128)
    enlarged = supersample(rgb, 4)
    assert enlarged.shape == (48, 64, 3)
    restored = np.array([restore_blacks(color) for color in enlarged.reshape(-1, 3)])
    # Cubic lobes may not invent a tone darker than the background or lighter than the plane.
    assert restored.min() >= 10 and restored.max() <= 151
    np.testing.assert_array_equal(restore_blacks(lift_blacks(np.array([[11, 60, 200]], np.uint8))[0]), (11, 60, 200))


def test_ink_lines_redraw_a_blurred_stroke_thinner_and_darker_and_spare_everything_else():
    canvas = np.full((96, 240, 3), (185, 166, 145), np.uint8)
    canvas[:, 40:43] = (24, 20, 16)        # hairline, three working pixels wide
    canvas[:, 110:150] = (24, 20, 16)      # a wide dark shape
    canvas[:, 200:] = (90, 84, 78)         # a plain edge
    blurred = cv2.GaussianBlur(canvas, (0, 0), 3.2)
    blurred[:, 70:90] = np.clip(blurred[:, 70:90].astype(int) + (np.arange(20) % 2*4)[None, :, None], 0, 255)
    result = ink_lines(blurred, 4)
    before, after = blurred[48, :, 1].astype(int), result[48, :, 1].astype(int)

    def stroke(row):
        depth = 166 - row[20:62]
        return depth.max(), np.count_nonzero(depth > depth.max()/2), int(np.argmax(depth))

    (depth_before, width_before, centre_before), (depth_after, width_after, centre_after) = stroke(before), stroke(after)
    assert depth_after > 1.4*depth_before
    assert .5*width_before <= width_after < width_before and centre_after == centre_before
    assert after.min() >= 10
    # The flanks go back to the tone beside the stroke; nothing outside it moves.
    assert np.all(after[20:62] <= 167)
    np.testing.assert_array_equal(after[66:240], before[66:240])
    np.testing.assert_array_equal(after[:16], before[:16])


def test_ink_lines_ignore_specks_and_keep_black_gaps_at_their_width():
    canvas = np.full((160, 200, 3), 170, np.uint8)
    cv2.circle(canvas, (40, 40), 3, (40, 40, 40), -1)      # a speck, not a line
    canvas[90:, 118:126] = 6                               # a gap that is already ink black
    blurred = cv2.GaussianBlur(canvas, (0, 0), 2.5)
    result = ink_lines(blurred, 4)
    np.testing.assert_array_equal(result[20:60, 20:60], blurred[20:60, 20:60])
    row_before, row_after = blurred[130, 90:154, 0].astype(int), result[130, 90:154, 0].astype(int)
    assert np.count_nonzero(row_after < 90) >= np.count_nonzero(row_before < 90) - 1
    assert np.abs(row_after - row_before).max() <= 20


def test_ink_lines_carry_a_stroke_through_its_faint_stretch():
    canvas = np.full((120, 400, 3), 170, np.uint8)
    canvas[58:61, 20:380] = 30
    canvas[58:61, 170:230] = 120           # the same line, washed out in the middle
    blurred = cv2.GaussianBlur(canvas, (0, 0), 2.2)
    result = ink_lines(blurred, 4)
    strong, faint = result[59, 60:140, 0].astype(int), result[59, 185:215, 0].astype(int)
    assert strong.max() < 70
    # Left alone the faint stretch stays near 145 and breaks the line in two.
    assert blurred[59, 185:215, 0].min() > 135
    assert faint.max() < 70


def test_ink_lines_keep_neighbouring_shapes_apart(tmp_path, monkeypatch):
    truth = np.full((4*96, 4*128, 3), (22, 24, 29), np.uint8)
    cv2.ellipse(truth, (256, 192), (220, 150), -12, 0, 360, (185, 166, 145), -1, lineType=cv2.LINE_AA)
    curves = []
    for offset in (-70, 0, 70):
        points = np.array([(20 + step*24, 205 + offset - int(50*np.sin(step/19*np.pi))) for step in range(20)])
        cv2.polylines(truth, [points], False, (21, 18, 15), 5, lineType=cv2.LINE_AA)
        curves.append(points)
    source = tmp_path / 'source.png'
    small = np.asarray(Image.fromarray(truth).resize((128, 96), Image.Resampling.BOX))
    Image.fromarray(cv2.GaussianBlur(small, (0, 0), .7)).save(source)
    light = {}
    for gain in (1., enhance.INK_GAIN):
        monkeypatch.setattr(enhance, 'INK_GAIN', gain)
        svg = tmp_path / f'gain-{gain}.svg'
        stats = vectorize(source, svg, mode='illustrative', preset='poster', base_colors=6, target_shapes=0, max_side=0)
        assert stats['supersample'] == 4 and stats['geometry']['unhandled_topology_violations'] == 0
        import cairosvg
        from io import BytesIO
        large = np.asarray(Image.open(BytesIO(cairosvg.svg2png(url=str(svg), output_width=512))).convert('L'), float)
        inside = [point for points in curves for point in points[4:-4]]
        light[gain] = np.array([large[y-2:y+3, x-2:x+3].min() for x, y in inside])
        bands = cv2.connectedComponents((large > 120).astype(np.uint8), connectivity=4)[0] - 1
    # Every separator is dark along its whole length, so the four bands of the plane stay apart.
    assert light[enhance.INK_GAIN].max() < 75
    assert light[enhance.INK_GAIN].mean() < .75*light[1.].mean()
    assert bands == 4


def test_black_lift_compresses_invisible_differences_only():
    lifted = lift_blacks(np.array([[2, 11, 40, 120, 250]], np.uint8))[0].astype(int)
    assert lifted[1] - lifted[0] <= 5
    assert lifted[3] - lifted[2] >= 75
    assert np.all(np.diff(lifted) > 0)


def test_faint_seams_merge_after_the_shape_target_is_met():
    rgb = np.zeros((12, 18, 3), np.uint8)
    rgb[:, :6] = 80; rgb[:, 6:12] = 86; rgb[:, 12:] = 240
    labels = np.tile(np.repeat(np.arange(3, dtype=np.int32), 6), (12, 1))
    seg = connected_regions(labels, rgb)
    alpha = np.full(labels.shape, 255, np.uint8); locked = np.zeros(labels.shape, bool)
    importance = importance_map(rgb, alpha)
    config = IllustrativeConfig.resolve('poster', target_shapes=3)
    kept, stats, _ = merge_planes(seg, rgb, importance, alpha, locked, config)
    assert stats['after'] == 3 and stats['faint_seams_joined'] == 0
    merged, stats, _ = merge_planes(seg, rgb, importance, alpha, locked, config, min_contrast=.04)
    assert stats['after'] == 2 and stats['faint_seams_joined'] == 1
    assert merged.labels[0, 0] == merged.labels[0, 8] != merged.labels[0, 17]


def test_ink_fit_rebuilds_bevelled_corners_as_straight_runs():
    # A raster has softened every corner of this quadrilateral into a short bevel.
    corners = np.array([(20, 22), (118, 30), (104, 112), (28, 96)], float)
    bevelled = []
    for index, corner in enumerate(corners):
        for other in (corners[index-1], corners[(index+1) % 4]):
            direction = (other - corner)/np.linalg.norm(other - corner)
            bevelled.append(corner + 3*direction)
    raw = lattice_chain(np.rint(bevelled))
    curves, attempt = fit_ink_curve(raw, 3.3, .95, 3.4, scale=2.)
    assert attempt == 0
    assert len(curves) == 4
    assert all(np.allclose(a, c1) and np.allclose(c2, b) for a, c1, c2, b in curves)
    apexes = np.array([curve[0] for curve in curves])
    assert cKDTree(apexes).query(corners)[0].max() < 2.
    assert cKDTree(sample_curves(curves)).query(raw)[0].max() <= 3.4


def test_ink_fit_keeps_round_shapes_round_and_unbiased():
    angle = np.linspace(0, 2*np.pi, 90, endpoint=False)
    circle = np.column_stack((70 + 46*np.cos(angle), 70 + 46*np.sin(angle)))
    raw = lattice_chain(np.rint(circle))
    curves, _ = fit_ink_curve(raw, 3.3, .95, 3.4, scale=2.)
    assert not any(np.allclose(a, c1) and np.allclose(c2, b) for a, c1, c2, b in curves)
    for left, right in zip(curves, curves[1:] + curves[:1]):
        incoming, outgoing = left[3] - left[2], right[1] - right[0]
        np.testing.assert_allclose(incoming/np.linalg.norm(incoming), outgoing/np.linalg.norm(outgoing), atol=1e-8)
    radius = np.linalg.norm(sample_curves(curves) - 70, axis=1)
    # Inscribing the curve in on-contour vertices would shrink the circle by about a pixel.
    assert abs(radius.mean() - 46) < .45
    assert np.ptp(radius) < 2.2


def test_ink_fit_pins_junction_endpoints_of_an_open_chain():
    x = np.arange(0, 61)
    raw = [(0, 30)]
    for column in x[1:]:
        target = int(round(30 - 14*np.sin(column/60*np.pi)))
        while raw[-1][1] != target:
            raw.append((raw[-1][0], raw[-1][1] + np.sign(target - raw[-1][1])))
        raw.append((column, target))
    raw = np.asarray(raw, float)
    curves, _ = fit_ink_curve(raw, 3.3, .95, 3.4, scale=2.)
    np.testing.assert_array_equal(curves[0][0], raw[0])
    np.testing.assert_array_equal(curves[-1][3], raw[-1])
    for left, right in zip(curves, curves[1:]):
        np.testing.assert_array_equal(left[3], right[0])


@pytest.mark.parametrize('kwargs', [{'supersample': 5}, {'supersample': -1},
                                    {'supersample': 2, 'contour_method': 'neural'}])
def test_supersample_configuration_is_validated(kwargs):
    with pytest.raises(ValueError):
        IllustrativeConfig.resolve('poster', **kwargs)
    with pytest.raises(ValueError):
        IllustrativeConfig.resolve('detail', supersample=2)


def test_poster_supersamples_the_reference_and_exports_source_dimensions(tmp_path):
    svg = tmp_path / 'poster.svg'
    debug = tmp_path / 'debug'
    report = tmp_path / 'poster.json'
    stats = vectorize(REFERENCE, svg, mode='illustrative', preset='poster', max_side=0,
                      debug_dir=debug, shape_report=report)
    assert stats['supersample'] == 4
    assert (stats['width'], stats['height']) == (920, 980)
    assert stats['display_dimensions'] == [230, 245]
    assert stats['config']['target_shapes'] == 560 and stats['shape_count'] <= 560
    assert stats['accepted_patches'] == 0
    # Fidelity is measured against the untouched source on its own pixel grid.
    metrics = stats['metrics_after']
    assert metrics['edge_f1']['1'] > .925
    # Ink lines are deliberately darker than the blurred source, which costs some mean error.
    assert metrics['rgb_mae'] < 5.7
    assert metrics['weighted_edge_loss'] < .045
    assert metrics['silhouette_iou'] > .86
    assert metrics['alpha_mae'] == 0
    geometry = stats['geometry']
    assert geometry['unhandled_topology_violations'] == 0
    assert geometry['lattice_fallbacks'] <= 3
    root = ET.parse(svg).getroot()
    assert (root.get('width'), root.get('height'), root.get('viewBox')) == ('230', '245', '0 0 920 980')
    assert '<image' not in svg.read_text(encoding='utf-8') and not root.findall(f'.//{SVG}clipPath')
    rendered = render_svg(svg)
    assert rendered.shape == (245, 230, 4) and np.all(rendered[..., 3] == 255)
    source = np.asarray(Image.open(REFERENCE).convert('RGB'), float)
    # The mouth and the wing emblem keep their structure without residual overlays.
    for window in (np.s_[86:100, 150:166], np.s_[100:140, 70:122]):
        assert np.abs(rendered[..., :3][window] - source[window]).mean() < 18
    # The backdrop stays the source black instead of a lifted or crushed tone.
    np.testing.assert_allclose(rendered[4, 4, :3], source[4, 4], atol=3)
    assert Image.open(debug / 'source.png').size == (230, 245)
    assert Image.open(debug / 'enhanced.png').size == (920, 980)
    data = json.loads(report.read_text(encoding='utf-8'))
    straight = sum(np.allclose(curve[0], curve[1]) and np.allclose(curve[2], curve[3])
                   for chain in data['boundaries'] for curve in chain['cubics'])
    total = sum(len(chain['cubics']) for chain in data['boundaries'])
    assert total < 9000 and straight > .1*total


def test_supersampled_poster_is_closer_to_clean_vector_art(tmp_path):
    truth = np.full((4*96, 4*128, 3), (21, 25, 34), np.uint8)
    cv2.fillPoly(truth, [np.array([(40, 300), (212, 52), (260, 100), (372, 40), (470, 250), (300, 350), (180, 250)])],
                 (223, 179, 109), lineType=cv2.LINE_AA)
    cv2.fillPoly(truth, [np.array([(250, 150), (306, 182), (262, 240), (214, 200)])], (21, 25, 34), lineType=cv2.LINE_AA)
    cv2.circle(truth, (120, 90), 34, (66, 138, 178), -1, lineType=cv2.LINE_AA)
    source = tmp_path / 'source.png'
    Image.fromarray(truth).resize((128, 96), Image.Resampling.BOX).save(source)
    errors = {}
    for label, scale in (('lattice', 1), ('supersampled', 0)):
        svg = tmp_path / f'{label}.svg'
        stats = vectorize(source, svg, mode='illustrative', preset='poster', base_colors=4,
                          target_shapes=0, max_side=0, supersample=scale)
        assert stats['supersample'] == (1 if scale else 4)
        assert stats['geometry']['unhandled_topology_violations'] == 0
        import cairosvg
        from io import BytesIO
        large = np.asarray(Image.open(BytesIO(cairosvg.svg2png(url=str(svg), output_width=512))).convert('RGB'), float)
        errors[label] = np.abs(large - truth).mean()
    assert errors['supersampled'] < 1.2
    assert errors['supersampled'] < .5*errors['lattice']


def test_alpha_and_corrections_stay_on_the_source_lattice(tmp_path):
    rgba = np.full((30, 35, 4), (90, 100, 110, 255), np.uint8)
    rgba[:, 17:, :3] = 235
    opaque = tmp_path / 'opaque.png'; Image.fromarray(rgba).save(opaque)
    mask = np.zeros_like(rgba); mask[7:13, 4:10] = (240, 10, 20, 255)
    correction = tmp_path / 'correction.png'; Image.fromarray(mask).save(correction)
    stats = vectorize(opaque, tmp_path / 'a.svg', mode='illustrative', preset='poster', max_side=0,
                      base_colors=2, target_shapes=0, detail_budget=0, correction_mask=correction)
    assert stats['supersample'] == 1
    rgba[:4, :4, 3] = 0
    cutout = tmp_path / 'cutout.png'; Image.fromarray(rgba).save(cutout)
    stats = vectorize(cutout, tmp_path / 'b.svg', mode='illustrative', preset='poster', max_side=0,
                      base_colors=2, target_shapes=0, detail_budget=0, supersample=3)
    assert stats['supersample'] == 1
    assert any('opaque source' in warning for warning in stats['warnings'])
    assert render_svg(tmp_path / 'b.svg').shape == (30, 35, 4)
