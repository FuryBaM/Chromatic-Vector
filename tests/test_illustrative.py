import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
import numpy as np
import pytest
from PIL import Image
from chromatic_vector.modes import IllustrativeConfig
from chromatic_vector.flatten import flatten_rgba
from chromatic_vector.base_segmentation import propose, merge_planes, rag_statistics
from chromatic_vector.cli import vectorize
from chromatic_vector.metrics import render_svg, objective
from chromatic_vector.shape_fitter import ellipse_fit
from chromatic_vector.geometry import safe_shared
from chromatic_vector.contours import extract_edges, trace_rings, signed_area
from chromatic_vector.hierarchy import containment_parents
from chromatic_vector.shape_graph import build_document


def run(tmp_path, rgba, **kwargs):
    source=tmp_path/'source.png'; dest=tmp_path/'result.svg';report=tmp_path/'report.json'
    Image.fromarray(rgba).save(source)
    stats=vectorize(source,dest,mode='illustrative',target_shapes=0,max_side=0,
                    detail_budget=0,shape_report=report,**kwargs)
    return stats,json.loads(report.read_text()),render_svg(dest),dest

@pytest.mark.parametrize('kwargs',[{'base_colors':1},{'base_colors':257},{'target_shapes':-1},
    {'detail_budget':-1},{'boundary_preservation':1.1},{'boundary_preservation':float('nan')},
    {'residual_threshold':float('inf')},{'max_shape_levels':0},{'refine_passes':6},{'primitive_fit':'fake'}])
def test_config_rejects_invalid(kwargs):
    with pytest.raises(ValueError): IllustrativeConfig.resolve(**kwargs)


def test_presets_explicit_override():
    c=IllustrativeConfig.resolve('draft',base_colors=22,detail_budget=7)
    assert (c.base_colors,c.detail_budget,c.refine_passes)==(22,7,1)

@pytest.mark.parametrize('alpha',[0,1,64,128,254,255])
def test_uniform_alpha_roundtrip(tmp_path,alpha):
    rgba=np.full((12,13,4),(37,104,170,alpha),np.uint8)
    stats,report,out,path=run(tmp_path,rgba)
    assert np.array_equal(out[...,3],rgba[...,3])
    if alpha:
        assert tuple(report['shapes'][0]['fill'])==(37,104,170,alpha)
        from chromatic_vector.metrics import composite
        np.testing.assert_allclose(composite(out),composite(rgba),atol=1.1)
    assert stats['shape_count']==(alpha>0)
    assert '<image' not in path.read_text()


def test_alpha_hole_rim_and_hidden_rgb(tmp_path):
    rgba=np.full((20,23,4),(22,90,150,0),np.uint8)
    rgba[3:17,3:20]=(190,60,20,128);rgba[7:13,8:15,3]=0
    _,r,out,_=run(tmp_path,rgba)
    np.testing.assert_array_equal(out[...,3],rgba[...,3])
    flat,_=flatten_rgba(rgba)
    changed=rgba.copy();changed[rgba[...,3]==0,:3]=(255,0,255)
    flat2,_=flatten_rgba(changed)
    np.testing.assert_array_equal(flat,flat2)
    assert r['geometry']['unhandled_topology_violations']==0


def test_flatten_source_immutable_and_thin_edges():
    rng=np.random.default_rng(4)
    rgba=np.full((40,50,4),(130,130,130,255),np.uint8)
    rgba[...,:3]=np.clip(rgba[...,:3].astype(int)+rng.integers(-5,6,(40,50,3)),0,255)
    rgba[:,24:26,:3]=0
    before=rgba.copy();flat,importance=flatten_rgba(rgba)
    np.testing.assert_array_equal(rgba,before)
    assert flat[:,:20].std()<rgba[:,:20,:3].std()
    assert flat[:,24:26].mean()<8
    assert importance[:,23:27].max()>.8


def test_hard_boundary_and_manual_locks(tmp_path):
    rgba=np.full((30,35,4),(90,100,110,255),np.uint8)
    rgba[:,17:,:3]=235
    mask=np.zeros_like(rgba);mask[7:13,4:10]=(240,10,20,255)
    cp=tmp_path/'correction.png';Image.fromarray(mask).save(cp)
    _,r,out,_=run(tmp_path,rgba,correction_mask=cp,base_colors=2,refine_passes=2)
    np.testing.assert_array_equal(out[9,6,:3],(240,10,20))
    assert r['corrected_pixels']==36 and any(s['locked'] for s in r['shapes'])
    assert out[:,18:,:3].mean()>220


def test_rag_weak_merges_strong_stays():
    rgb=np.zeros((12,18,3),np.uint8);rgb[:,:6]=80;rgb[:,6:12]=83;rgb[:,12:]=240
    labels=np.tile(np.repeat(np.arange(3,dtype=np.int32),6),(12,1))
    from chromatic_vector.palette import connected_regions
    seg=connected_regions(labels,rgb)
    alpha=np.full(labels.shape,255,np.uint8);locked=np.zeros(labels.shape,bool)
    from chromatic_vector.flatten import importance_map
    imp=importance_map(rgb,alpha)
    conf=IllustrativeConfig.resolve(target_shapes=1)
    merged,stats,rag=merge_planes(seg,rgb,imp,alpha,locked,conf)
    assert stats['after']==2 and stats['joined']==1
    assert merged.labels[0,0]==merged.labels[0,8]!=merged.labels[0,17]
    assert stats['protected_edges']==1

@pytest.mark.parametrize('fixture',['rect','nested','junction','diagonal','thin','random'])
def test_topology_and_canonical_reverse(fixture):
    labels=np.zeros((25,29),np.int32)
    if fixture=='rect': labels[3:20,5:24]=1
    elif fixture=='nested': labels[2:23,2:27]=1;labels[6:18,6:23]=2;labels[9:14,9:17]=3
    elif fixture=='junction': labels[:,14:]=1;labels[12:,:14]=2
    elif fixture=='diagonal': labels[8:12,8:12]=1;labels[12:16,12:16]=1
    elif fixture=='thin': labels[3:22,13:14]=1
    else: labels=np.random.default_rng(3).integers(0,5,labels.shape,dtype=np.int32)
    graph,stats=safe_shared(extract_edges(labels))
    paths=graph.paths(trace_rings(extract_edges(labels)))
    assert set(paths)==set(np.unique(labels))
    assert stats['unhandled_topology_violations']==0
    for chain in graph.chains:
        for c in chain.cubics:
            a,b,d,e=c
            assert graph._command(c,True)==graph._command((e,d,b,a),False)
        np.testing.assert_array_equal(chain.cubics[0][0],chain.vertices[0])
        np.testing.assert_array_equal(chain.cubics[-1][-1],chain.vertices[-1])


def test_containment_not_adjacency():
    labels=np.zeros((32,34),np.int32);labels[3:29,3:31]=1;labels[8:24,8:26]=2;labels[12:20,12:22]=3
    rings=trace_rings(extract_edges(labels));doc=build_document(labels,np.zeros((4,3),np.uint8),rings)
    parents=containment_parents(doc)
    assert parents=={1:0,2:1,3:2}

@pytest.mark.parametrize('cut',[False,True])
def test_ellipse_positive_negative(cut):
    yy,xx=np.indices((61,71));mask=((xx+.5-35)/24)**2+((yy+.5-30)/19)**2<=1
    if cut:mask[20:40,48:]=False
    rings=trace_rings(extract_edges(mask.astype(np.int32)))
    fit=ellipse_fit(mask,rings[1][0])
    assert (fit is not None)==(not cut)


def test_actual_ellipse_shared_hole(tmp_path):
    yy,xx=np.indices((70,80));mask=((xx+.5-40)/25)**2+((yy+.5-35)/20)**2<=1
    rgba=np.full((70,80,4),255,np.uint8);rgba[mask,:3]=(20,80,130)
    _,r,out,path=run(tmp_path,rgba,base_colors=2)
    assert r['primitives'].get('ellipse')==1
    assert ET.parse(path).findall('.//{http://www.w3.org/2000/svg}ellipse')
    assert ' A ' in path.read_text()


def test_determinism_and_native_parity(tmp_path):
    rgba=np.full((30,40,4),255,np.uint8);rgba[4:26,5:35,:3]=(40,75,110)
    rgba[10:19,13:23,:3]=(210,170,90)
    _,a,out,path=run(tmp_path,rgba,backend='python',base_colors=4)
    svg=path.read_text()
    from chromatic_vector.native import extract_native_edges
    if extract_native_edges(np.zeros((1,1),np.int32)) is None:
        pytest.skip('Native C++ library not built')
    _,b,out2,path=run(tmp_path,rgba,backend='native',base_colors=4)
    assert svg==path.read_text()
    np.testing.assert_array_equal(out,out2)
    for key in ('shapes','boundaries','metrics_after','config'): assert a[key]==b[key]


def test_residual_refinement_acceptance_and_correction(tmp_path):
    rgba=np.full((50,60,4),(90,100,120,255),np.uint8)
    rgba[8:40,8:45,:3]=(120,130,145)
    rgba[12:23,15:35,:3]=(210,180,145)
    rgba[30:35,20:44,:3]=(45,55,70)
    source=tmp_path/'input.png';Image.fromarray(rgba).save(source)
    path=tmp_path/'refined.svg';report=tmp_path/'refined.json'
    stats=vectorize(source,path,mode='illustrative',base_colors=2,detail_budget=12,target_shapes=0,
                    residual_threshold=.01,max_side=0,shape_report=report)
    data=json.loads(report.read_text());ref=data['refinement']
    assert ref['objective_after']<=ref['objective_before']
    assert ref['accepted_patches']>0
    for entry in ref['patch_log']:
        if entry['accepted']: assert entry['objective_delta']<0
    shapes=data['shapes'];uids={s['uid'] for s in shapes}
    assert len(uids)==len(shapes)
    for s in shapes:
        if s['kind']=='overlay': assert s['clipped_by_uid'] in uids and s['composition_parent_uid'] in uids
    assert len(ET.parse(path).findall('.//{http://www.w3.org/2000/svg}clipPath'))>0
    base_path=tmp_path/'base.svg'
    vectorize(source,base_path,mode='illustrative',base_colors=2,detail_budget=0,target_shapes=0,max_side=0)
    base=render_svg(base_path);output=render_svg(path)
    outside=np.ones(rgba.shape[:2],bool)
    for entry in ref['patch_log']:
        if entry['accepted']:
            x,y,x1,y1=entry['bbox'];outside[y:y1,x:x1]=False
    np.testing.assert_array_equal(base[outside],output[outside])


def test_cli_invalid_message(tmp_path):
    source=tmp_path/'a.png';Image.new('RGB',(4,4),'white').save(source)
    result=subprocess.run([sys.executable,'-m','chromatic_vector',str(source),str(tmp_path/'a.svg'),
        '--mode','illustrative','--boundary-preservation','2'],capture_output=True,text=True)
    assert result.returncode==2 and '--boundary-preservation' in result.stderr


def test_python_fallback_and_explicit_native_error(tmp_path,monkeypatch):
    import chromatic_vector.pipeline as pipeline
    monkeypatch.setattr(pipeline,'extract_native_edges',lambda labels:None)
    rgba=np.full((9,11,4),(40,80,120,255),np.uint8)
    stats,_,_,_=run(tmp_path,rgba,backend='auto')
    assert stats['backend']=='python'
    with pytest.raises(RuntimeError,match='Native backend requested'):
        run(tmp_path,rgba,backend='native')


def test_legacy_cli_and_programmatic_output(tmp_path):
    rgba=np.full((15,18,4),255,np.uint8);rgba[3:12,4:14,:3]=(30,100,150)
    source=tmp_path/'input.png';Image.fromarray(rgba).save(source)
    old=tmp_path/'legacy.svg';new=tmp_path/'explicit.svg'
    common=[str(source),'OUT','--quality','precise','--target-shapes','2000','--edge-protection','2.5','--backend','python']
    subprocess.run([sys.executable,'-m','chromatic_vector',*[str(old) if x=='OUT' else x for x in common]],check=True,capture_output=True)
    subprocess.run([sys.executable,'-m','chromatic_vector',*[str(new) if x=='OUT' else x for x in common],'--mode','precise'],check=True,capture_output=True)
    assert old.read_bytes()==new.read_bytes()


def test_locked_color_survives_real_refinement(tmp_path):
    rgba=np.full((35,44,4),(60,65,70,255),np.uint8)
    rgba[5:30,7:36,:3]=(150,130,100);rgba[10:18,10:24,:3]=(210,200,180)
    correction=np.zeros_like(rgba);correction[8:13,8:13]=(250,20,40,255)
    source=tmp_path/'src.png';mask=tmp_path/'mask.png';out=tmp_path/'out.svg'
    Image.fromarray(rgba).save(source);Image.fromarray(correction).save(mask)
    stats=vectorize(source,out,mode='illustrative',base_colors=2,detail_budget=15,residual_threshold=.01,
                    correction_mask=mask,max_side=0,target_shapes=0)
    rendered=render_svg(out)
    np.testing.assert_array_equal(rendered[9:12,9:12,:3],np.broadcast_to((250,20,40),(3,3,3)))
    assert stats['corrected_pixels']==25


def test_large_background_queue_stays_bounded():
    from chromatic_vector.palette import connected_regions
    rgb=np.clip(110+np.random.default_rng(19).integers(-3,4,(28,36,3)),0,255).astype(np.uint8)
    labels=np.arange(28*36,dtype=np.int32).reshape(28,36)
    seg=connected_regions(labels,rgb)
    alpha=np.full(labels.shape,255,np.uint8)
    result,stats,rag=merge_planes(seg,rgb,np.zeros(labels.shape,np.float32),alpha,
                                np.zeros(labels.shape,bool),IllustrativeConfig.resolve(target_shapes=1))
    assert stats['after']==1
    assert stats['peak_queue_entries']<len(rag)*8
    assert result.sizes.sum()==labels.size
    np.testing.assert_allclose(result.region_colors[0],np.rint(rgb.mean(axis=(0,1))),atol=0)

@pytest.mark.parametrize('levels',[1,2,3])
def test_max_shape_levels_and_primitive_off(tmp_path,levels):
    rgba=np.full((13,17,4),255,np.uint8);rgba[3:10,4:13,:3]=30
    _,report,_,_=run(tmp_path,rgba,max_shape_levels=levels,primitive_fit='off')
    allowed={'primary','secondary','detail'} if levels==3 else {'primary','secondary'} if levels==2 else {'primary'}
    assert set(report['shape_levels'])<=allowed
    assert set(report['primitives'])=={'path'}


def test_native_uint32_region_ids():
    from chromatic_vector.native import extract_native_edges
    labels=np.array([[0,65536],[90000,65536]],np.uint32)
    native=extract_native_edges(labels)
    if native is None:pytest.skip('Native C++ library not built')
    expected=extract_edges(labels)
    ordered=lambda a:a[np.lexsort(tuple(a[:,j] for j in range(4,-1,-1)))]
    np.testing.assert_array_equal(ordered(native),ordered(expected))


def test_correction_preserves_distinct_alpha_strata(tmp_path):
    rgba=np.full((18,23,4),(80,100,140,255),np.uint8)
    rgba[:,:7,3]=0;rgba[:,7:14,3]=128
    correction=np.full_like(rgba,(230,40,60,255))
    path=tmp_path/'mask.png';Image.fromarray(correction).save(path)
    _,report,out,_=run(tmp_path,rgba,correction_mask=path)
    np.testing.assert_array_equal(out[...,3],rgba[...,3])
    assert {s['fill'][3] for s in report['shapes']}=={128,255}
    assert all(s['fill'][:3]==[230,40,60] for s in report['shapes'])


def test_exact_polygon_is_real_svg_element(tmp_path):
    rgba=np.full((45,55,4),255,np.uint8)
    rgba[5:38,5:20,:3]=(40,80,140);rgba[24:38,5:48,:3]=(40,80,140)
    _,report,_,path=run(tmp_path,rgba,base_colors=2,smooth=0)
    assert report['primitives'].get('polygon')==1
    assert ET.parse(path).findall('.//{http://www.w3.org/2000/svg}polygon')
