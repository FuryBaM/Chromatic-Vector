"""Visual geometry regressions: a vector must stay smooth when rendered larger."""
from io import BytesIO
import json
import cairosvg
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from chromatic_vector.cli import vectorize
from chromatic_vector.contours import extract_edges
from chromatic_vector.geometry import safe_shared
from chromatic_vector.curve_fitting import fit_shared_curve, sample_curves


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
