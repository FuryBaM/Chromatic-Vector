"""v0.5 shape-budget, topology and edge-protection regression tests."""
import numpy as np
from chromatic_vector.palette import connected_regions
from chromatic_vector.agglomerate import agglomerate_to_budget
from chromatic_vector.contours import extract_edges, trace_rings, signed_area


def test_shape_budget_reduces_region_count_and_preserves_area():
    y,x=np.mgrid[:25,:40]
    ids=(x//4 + y//5)%4
    colors=np.array([[25,23,20],[34,31,28],[42,39,35],[54,50,48]],dtype=np.uint8)
    image=colors[ids]
    seg=connected_regions(ids,image)
    reduced, stats=agglomerate_to_budget(seg,image,target=6,max_added_mse=200,
                                         edge_protection=0,hard_edge=255)
    assert 1<=stats.after<=6
    assert stats.target_met
    assert stats.joined==stats.before-stats.after
    assert np.all(reduced.labels >= 0)
    assert sum(reduced.sizes)==25*40
    loops=trace_rings(extract_edges(reduced.labels))
    for rid in range(len(reduced.sizes)):
        assert abs(sum(signed_area(r) for r in loops[rid])-reduced.sizes[rid])<1.e-6


def test_strong_edges_protected_and_zero_budget():
    img=np.zeros((24,40,3),dtype=np.uint8)
    img[:,:20]=[16,16,20]
    img[:,20:]=[247,242,236]
    seg=connected_regions(np.arange(40,dtype=np.int32)[None,:].repeat(24,axis=0),img)
    reduced,stats=agglomerate_to_budget(seg,img,target=1,max_added_mse=1000)
    # Even though the target is 1, the strong midline must survive.
    assert reduced.labels[10,5]!=reduced.labels[10,35]
    assert not stats.target_met
    unchanged,st=agglomerate_to_budget(seg,img,target=1,max_added_mse=0)
    assert unchanged.labels[10,5]!=unchanged.labels[10,35]
    assert st.added_mse==0


def test_deterministic_and_no_merge_when_already_under_target():
    ids=np.array([[0,0,1,1,2,2],[0,3,3,1,2,4],[5,3,3,6,6,4]],dtype=np.int32)
    colors=np.array([[11,11,11],[15,15,15],[40,40,40],[14,14,14],
                     [43,43,43],[13,13,13],[39,39,39]],dtype=np.uint8)
    rgb=colors[ids]
    seg=connected_regions(ids,rgb)
    a,sa=agglomerate_to_budget(seg,rgb,target=3,max_added_mse=100)
    b,sb=agglomerate_to_budget(seg,rgb,target=3,max_added_mse=100)
    np.testing.assert_array_equal(a.labels,b.labels)
    assert sa==sb
    c,sc=agglomerate_to_budget(seg,rgb,target=99)
    np.testing.assert_array_equal(c.labels,seg.labels)
    assert sc.joined==0


def test_manual_correction_survives_global_optimization(tmp_path):
    from PIL import Image
    from chromatic_vector.cli import vectorize
    rgb=np.full((32,32,3),33,dtype=np.uint8)
    rgb[5:25,15:30]=[40,41,45]
    src=tmp_path/'in.png'
    Image.fromarray(rgb).save(src)
    mask=np.zeros((32,32,4),dtype=np.uint8)
    mask[10:19,8:13]=[238,64,29,255]
    patch=tmp_path/'patch.png';Image.fromarray(mask).save(patch)
    result=tmp_path/'out.svg'
    stats=vectorize(src,result,colors=3,min_area=1,max_side=0,
                    correction_mask=patch,target_shapes=1,
                    merge_mse_budget=500,merge_tolerance=10)
    import xml.etree.ElementTree as ET
    root=ET.parse(result)
    fills=[x.get('fill') for x in root.iter() if x.get('fill')]
    assert '#ee401d' in fills
    assert stats['corrected_pixels']==9*5
    assert stats['shape_count']>1  # strong painted boundary must remain
