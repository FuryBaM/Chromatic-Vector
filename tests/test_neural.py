"""Learned-model quality, CPU runtime, guard and cache-invalidation regressions."""
from pathlib import Path
import json,subprocess,sys,hashlib
import numpy as np
import pytest
from PIL import Image
from chromatic_vector.neural import BezierModel,NeuralContext,conservative_merge,_render_controls
from chromatic_vector.neural_data import make_dataset
from chromatic_vector.metrics import render_svg
from chromatic_vector.palette import connected_regions
from chromatic_vector.cli import vectorize


def test_trained_model_beats_straight_lines_on_unseen_vectors():
    model=BezierModel();data=make_dataset(96,721031)
    p,c,h,m,l=data;errors=[];straight=[];corner=[];merge=[]
    for i in range(len(p)):
        handles,pc,pm=model.predict(p[i],c[i])
        corner.append((pc>.5)==(l[i,0]>.5));merge.append((pm>.5)==(l[i,1]>.5))
        if l[i,2]:
            errors.append(np.abs(_render_controls(handles)-m[i]).mean())
            straight.append(np.abs(_render_controls([.3,0,.7,0])-m[i]).mean())
    assert np.mean(errors)<np.mean(straight)*.45
    assert np.mean(corner)>.85 and np.mean(merge)>.90
    assert model.meta['synthetic_only'] and not model.meta['photo_fixtures_used_in_training']


def test_prediction_depends_on_input_raster():
    model=BezierModel();p,c,h,m,l=make_dataset(32,44281)
    valid=np.flatnonzero(l[:,2]);predictions=[model.predict(p[i],c[i])[0] for i in valid]
    assert np.std(np.array(predictions)[:,1])>.05


def test_missing_and_mismatched_model_fail_explicitly(tmp_path):
    with pytest.raises(RuntimeError,match='missing'):BezierModel(tmp_path/'absent.onnx')
    model=BezierModel();p=tmp_path/'model.onnx';p.write_bytes(model.path.read_bytes()+b'corrupt')
    (tmp_path/'metadata.json').write_text(json.dumps(model.meta))
    with pytest.raises(RuntimeError,match='checksum'):BezierModel(p)


def test_runtime_does_not_import_torch():
    code="import sys;sys.modules['torch']=None;from chromatic_vector.neural import BezierModel;import numpy as np;m=BezierModel();m.predict(np.zeros((5,64,64),np.float32),np.zeros(8,np.float32))"
    subprocess.run([sys.executable,'-c',code],check=True,capture_output=True)


def test_weak_seam_merge_rebuilds_dense_palette_and_preserves_locks():
    rgb=np.full((28,32,3),110,np.uint8)
    labels=np.zeros((28,32),np.int32);labels[9:15,11:17]=1
    seg=connected_regions(labels,rgb);alpha=np.full(labels.shape,255,np.uint8)
    imp=np.zeros(labels.shape,np.float32);locked=np.zeros(labels.shape,bool)
    result,stats=conservative_merge(seg,rgb,alpha,locked,imp,{(0,1):1.},.9)
    assert stats['merged']==1 and np.array_equal(np.unique(result.labels),np.arange(len(result.sizes)))
    assert int(result.sizes.sum())==labels.size
    assert np.array_equal(result.region_colors[result.labels],rgb)
    locked[11:14,12:15]=True
    same,rejected=conservative_merge(seg,rgb,alpha,locked,imp,{(0,1):1.},.9)
    assert rejected['merged']==0 and any(p['reason']=='protected' for p in rejected['proposals'])
    np.testing.assert_array_equal(same.labels,seg.labels)


def test_model_scores_artificial_island_for_merge():
    rgb=np.full((32,36,3),105,np.uint8);labels=np.zeros((32,36),np.int32);labels[12:18,14:20]=1
    seg=connected_regions(labels,rgb);model=BezierModel();ctx=NeuralContext(rgb,seg.labels,seg.region_colors,model)
    from chromatic_vector.contours import extract_edges
    from chromatic_vector.shared import SharedBoundaryGeometry
    graph=SharedBoundaryGeometry(extract_edges(seg.labels),simplify=0,smooth=0)
    ctx.score_merge_pairs(graph.chains,{})
    # Classification uses .5. Automatic union deliberately demands .97 and
    # may reject a correctly classified but insufficiently confident proposal.
    assert ctx.merge_hints[(0,1)]>.5


def test_neural_svg_contains_real_guarded_predictions(tmp_path):
    y,x=np.indices((95,130));t=np.clip((x+.5)/130,0,1)
    curve=45+24*np.sin(t*np.pi*2)
    mask=y+.5>curve
    rgba=np.full((95,130,4),(45,55,65,255),np.uint8);rgba[mask,:3]=(165,150,125)
    source=tmp_path/'input.png';out=tmp_path/'out.svg';report=tmp_path/'report.json'
    Image.fromarray(rgba).save(source)
    stats=vectorize(source,out,mode='illustrative',contour_method='neural',base_colors=2,target_shapes=0,
                    detail_budget=0,max_side=0,simplify=1.1,smooth=.9,max_error=1.1,shape_report=report)
    r=json.loads(report.read_text());neural=r['neural']
    assert neural['inference_used'] and neural['counters']['emitted_segments']>0
    assert neural['output_contains_neural_segments']
    for entry in neural['records']:
        if entry.get('emitted'):assert entry['accepted'] and entry['objective_delta']<0
    assert '<image' not in out.read_text()
    assert r['geometry']['unhandled_topology_violations']==0
    assert np.all(render_svg(out)[...,3]==255)


def test_neural_alpha_and_correction_are_exact(tmp_path):
    rgba=np.full((18,24,4),(80,100,120,255),np.uint8);rgba[:,:7,3]=0;rgba[:,7:14,3]=128
    correction=np.full_like(rgba,(230,40,60,255));source=tmp_path/'in.png';mask=tmp_path/'mask.png';out=tmp_path/'out.svg'
    Image.fromarray(rgba).save(source);Image.fromarray(correction).save(mask)
    stats=vectorize(source,out,mode='illustrative',contour_method='neural',correction_mask=mask,
                    target_shapes=0,detail_budget=0,max_side=0)
    result=render_svg(out);np.testing.assert_array_equal(result[...,3],rgba[...,3])
    np.testing.assert_array_equal(result[4:12,15:22,:3],np.broadcast_to([230,40,60],(8,7,3)))
    assert not stats['neural']['output_contains_neural_segments']

@pytest.mark.parametrize('kwargs',[dict(neural_merge=True),dict(neural_model='missing.onnx'),dict(contour_method='neural',neural_optimize_steps=25)])
def test_neural_options_are_validated(tmp_path,kwargs):
    source=tmp_path/'input.png';Image.new('RGB',(12,12),'white').save(source)
    with pytest.raises(ValueError):vectorize(source,tmp_path/'out.svg',**kwargs)
