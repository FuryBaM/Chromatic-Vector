"""Source-kernel checks and topology regressions for the optional UG adapter."""
import json
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from PIL import Image
from chromatic_vector.unipolar import chaikin,fit_unipolar,simplify
from chromatic_vector.shared import adaptive_filter_py
from chromatic_vector.contours import extract_edges,trace_rings
from chromatic_vector.geometry import safe_shared
from chromatic_vector.cli import vectorize
from chromatic_vector.metrics import render_svg


def test_source_chaikin_weights_and_open_endpoints():
    p=np.array([[0.,0.],[8.,0.],[8.,8.]])
    np.testing.assert_array_equal(chaikin(p,False),[[0,0],[2,0],[6,0],[8,2],[8,6],[8,8]])
    closed=chaikin(p,True)
    np.testing.assert_array_equal(closed[-2:],[[6,6],[2,2]])


def test_source_closed_rdp_order():
    p=np.array([[0,0],[1,0],[2,0],[2,1],[2,2],[1,2],[0,2],[0,1]],float)
    np.testing.assert_array_equal(simplify(p,True,.1),[[0,0],[2,0],[2,2],[0,2]])


def test_current_ug_threshold_differs_from_legacy_cv():
    p=np.array([[0,0],[1,0],[1,1]],float)
    legacy=adaptive_filter_py(p,False,.28,.22,.65)
    current=adaptive_filter_py(p,False,.28,.22,1.5)
    np.testing.assert_array_equal(legacy[1],[1,0])
    assert current[1,0]<1 and current[1,1]>0

@pytest.mark.parametrize('method',['unipolar-adaptive','unipolar-chaikin'])
def test_shared_curves_reverse_and_svg_starts_on_fitted_anchor(method):
    labels=np.zeros((32,34),np.int32);labels[7:25,8:26]=1
    edges=extract_edges(labels);rings=trace_rings(edges)
    graph,stats=safe_shared(edges,simplify=1.1,smooth=.9,max_error=1.1,method=method)
    paths=graph.paths(rings)
    chain=next(c for c in graph.chains if c.paired_regions==(0,1))
    expected='M '+' '.join(f'{float(v):.3f}'.rstrip('0').rstrip('.') or '0' for v in chain.cubics[0][0])
    assert paths[1][0].startswith(expected+' ')
    assert stats['unhandled_topology_violations']==0
    for c in chain.cubics:
        a,b,d,e=c
        assert graph._command(c,True)==graph._command((e,d,b,a),False)

@pytest.mark.parametrize('method',['unipolar-adaptive','unipolar-chaikin'])
def test_method_preserves_alpha_and_locked_swatches(tmp_path,method):
    rgba=np.full((22,26,4),(80,100,130,255),np.uint8)
    rgba[:,:7,3]=0;rgba[:,7:15,3]=128
    correction=np.zeros_like(rgba);correction[3:18,5:20]=(230,30,70,255)
    source=tmp_path/'a.png';mask=tmp_path/'mask.png';out=tmp_path/'a.svg'
    Image.fromarray(rgba).save(source);Image.fromarray(correction).save(mask)
    stats=vectorize(source,out,mode='illustrative',contour_method=method,base_colors=2,target_shapes=0,
                    detail_budget=0,max_side=0,correction_mask=mask)
    result=render_svg(out)
    np.testing.assert_array_equal(result[...,3],rgba[...,3])
    np.testing.assert_array_equal(result[6:15,16:19,:3],np.broadcast_to((230,30,70),(9,3,3)))
    assert stats['config']['contour_method']==method
    assert stats['geometry']['unhandled_topology_violations']==0


def test_native_filter_matches_source_parameter_reference():
    from chromatic_vector.native import filter_native_polyline
    p=np.array([[0,0],[1,0],[1,1],[2,1],[2,2],[3,2]],np.float32)
    native=filter_native_polyline(p,False,.28,.22,1.5)
    if native is None:pytest.skip('Native C++ library not built')
    np.testing.assert_allclose(native,adaptive_filter_py(p,False,.28,.22,1.5),atol=1e-5)
