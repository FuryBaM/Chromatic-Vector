"""v0.4 shape/layer/primitive invariants."""
from xml.etree import ElementTree as ET
import json
import numpy as np
from chromatic_vector.palette import connected_regions
from chromatic_vector.shape_optimizer import merge_similar_shapes
from chromatic_vector.shape_graph import build_document, layer_geometry_styles
from chromatic_vector.contours import extract_edges, trace_rings, signed_area
from chromatic_vector.shared import SharedBoundaryGeometry
from chromatic_vector.svg_shapes import write_layered_svg

NS = {'svg': 'http://www.w3.org/2000/svg'}


def build(labels, colors):
    edges = extract_edges(labels)
    loops = trace_rings(edges)
    doc = build_document(labels, np.asarray(colors, dtype=np.uint8), loops)
    geometry = SharedBoundaryGeometry(edges, .8, .8, region_styles=layer_geometry_styles(doc,.8,.8), max_error=.6)
    return doc, geometry.paths(loops)


def test_svg_has_layers_true_rectangle_compound_and_metadata(tmp_path):
    labels = np.zeros((20, 23),dtype=np.int32)
    labels[3:13, 4:18] = 1
    labels[6:9, 8:13] = 2
    doc, paths=build(labels, [[10,10,10],[200,50,30],[255,200,70]])
    assert doc.background_id == 0
    assert doc.shapes[0].layer == 'background'
    assert doc.shapes[1].parent_id == 0
    assert doc.shapes[2].parent_id == 1
    assert doc.shapes[1].holes
    assert doc.shapes[2].primitive == 'rectangle-candidate'
    output = tmp_path/'example.svg'
    result=write_layered_svg(output,doc,paths)
    parsed=ET.parse(output)
    groups=parsed.findall('.//svg:g', NS)
    assert len(groups) == 5
    assert groups[0].attrib['id']=='layer-background'
    assert parsed.findall('.//svg:rect',NS)
    assert parsed.findall('.//svg:path',NS)
    assert not parsed.findall('.//svg:image',NS)
    assert parsed.find('.//*[@id="shape-2"]').attrib['data-parent-id']=='1'
    assert result['rectangle'] >= 1


def test_shape_union_only_edge_adjacent_and_similar():
    ids=np.array([[0,0,1,2,2], [0,0,1,2,2], [0,0,3,3,3]],dtype=np.int32)
    colors={0:[100,100,100],1:[105,105,105],2:[240,20,20],3:[103,103,103]}
    rgb=np.zeros((*ids.shape,3),dtype=np.uint8)
    for i,c in colors.items():rgb[ids==i]=c
    seg=connected_regions(ids,rgb)
    new, stats=merge_similar_shapes(seg,rgb,tolerance=15,max_small_area=4)
    assert stats.joined>=1
    assert np.all(new.labels[:,0]==new.labels[:,1])
    # Red pixels must not be joined with grayscale.
    assert new.labels[0,4] != new.labels[0,1]
    # The union is pixel-exact in area, before fitting.
    loops=trace_rings(extract_edges(new.labels))
    for rid in np.unique(new.labels):
        assert abs(sum(signed_area(x) for x in loops[int(rid)])-np.sum(new.labels==rid))<1e-5
    same, none=merge_similar_shapes(seg,rgb,tolerance=0,max_small_area=4)
    assert none.joined==0
    np.testing.assert_array_equal(same.labels,seg.labels)


def test_shared_boundaries_even_when_layer_styles_disagree():
    ids=np.array([[0,0,0,1,1,1], [0,0,2,2,1,1], [0,2,2,1,1,1], [2,2,2,1,1,1]],dtype=np.int32)
    e=extract_edges(ids); loops=trace_rings(e)
    doc=build_document(ids,np.array([[0,0,0],[225,225,225],[90,100,150]],dtype=np.uint8),loops)
    g=SharedBoundaryGeometry(e,1.0,.95,region_styles=layer_geometry_styles(doc,1,.95))
    paths=g.paths(loops)
    assert all(paths.values())
    for chain in g.chains:
        for a,c1,c2,b in chain.cubics:
            assert g._command((a,c1,c2,b),True)==g._command((b,c2,c1,a),False)


def test_random_region_document_counts():
    rng=np.random.default_rng(10)
    for _ in range(5):
        palette=rng.integers(0,5,(18,26),dtype=np.int32)
        rgb=np.stack((palette*40,palette*30,palette*20),axis=-1).astype(np.uint8)
        seg=connected_regions(palette,rgb)
        loops=trace_rings(extract_edges(seg.labels))
        doc=build_document(seg.labels,seg.region_colors,loops)
        assert len(doc.shapes)==len(seg.sizes)
        assert sum(map(len,doc.layers.values()))==len(doc.shapes)
        for shape in doc.shapes.values():
            if shape.parent_id is not None:
                assert shape.nesting_depth>doc.shapes[shape.parent_id].nesting_depth
