from xml.etree import ElementTree as ET
import numpy as np
from chromatic_vector.palette import connected_regions
from chromatic_vector.contours import trace_rings, extract_edges
from chromatic_vector.svg import write_svg


def test_disconnected_same_color_and_svg(tmp_path):
    ids = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int32)
    rgb = np.zeros((3, 3, 3), dtype=np.uint8)
    rgb[ids == 0] = [245, 100, 10]
    rgb[ids == 1] = [10, 15, 20]
    seg = connected_regions(ids, rgb)
    assert len(seg.sizes) == 5  # 4 disconnected corners and centre cross
    rings = trace_rings(extract_edges(seg.labels))
    dest = tmp_path / 'test.svg'
    write_svg(dest, 3, 3, rings, seg.region_colors)
    xml = ET.parse(dest)
    ns = {'svg': 'http://www.w3.org/2000/svg'}
    paths = xml.findall('.//svg:path', ns)
    assert len(paths) == 5
    assert not xml.findall('.//svg:image', ns)
