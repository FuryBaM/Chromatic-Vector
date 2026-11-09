"""Image-conditioned Bézier proposals from a genuinely trained local ONNX CNN.

Inference uses OpenCV CPU only. No downloaded/pretrained weights, no Torch runtime.
Every proposal is bounded, compared with a raster objective and topology-audited.
"""
from dataclasses import dataclass,field
from pathlib import Path
from collections import Counter
import hashlib,json,time
import cv2
import numpy as np
from .neural_data import SIZE,X,Y,curve_samples
from .curve_fitting import fit_shared_curve,sample_curves
from .unipolar import _within

DEFAULT_MODEL=Path(__file__).resolve().parents[1]/'models/bezier_patch_v1/model.onnx'


class BezierModel:
    def __init__(self,path=None):
        path=Path(path) if path else DEFAULT_MODEL
        if not path.exists():
            packaged=Path(__file__).parent/'assets/bezier_patch_v1/model.onnx'
            if not path.exists() and path==DEFAULT_MODEL and packaged.exists():path=packaged
        if not path.exists():raise RuntimeError('Neural model missing: '+str(path)+'; train with python training/train_bezier.py or provide --neural-model')
        self.path=path;self.sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        meta_path=path.with_name('metadata.json')
        if not meta_path.exists():raise RuntimeError('Neural model metadata.json is required beside the ONNX file')
        self.meta=json.loads(meta_path.read_text())
        if self.meta.get('architecture')!='PatchBezier-v1' or self.meta.get('patch_size')!=64:
            raise RuntimeError('Unsupported neural network architecture or patch size')
        if self.meta.get('sha256')!=self.sha256:raise RuntimeError('Neural model checksum does not match metadata.json')
        if self.meta.get('schema')!=1:raise RuntimeError('Unsupported neural model metadata schema')
        self.net=cv2.dnn.readNetFromONNX(str(path));self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        if int(cv2.__version__.split('.')[0])<5:self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def predict(self,patch,context):
        p=np.ascontiguousarray(patch[None],np.float32);c=np.ascontiguousarray(context[None],np.float32)
        self.net.setInput(p,'patch');self.net.setInput(c,'context')
        z=self.net.forward().reshape(-1)
        sigmoid=lambda v:1/(1+np.exp(-np.clip(v,-40,40)))
        controls=np.array([.15+.30*sigmoid(z[0]),.80*np.tanh(z[1]),.55+.30*sigmoid(z[2]),.80*np.tanh(z[3])],float)
        return controls,float(sigmoid(z[4])),float(sigmoid(z[5]))


@dataclass
class NeuralContext:
    rgb: np.ndarray
    labels: np.ndarray
    colors: np.ndarray
    model: BezierModel
    optimize_steps: int = 5
    records: list = field(default_factory=list)
    merge_hints: dict = field(default_factory=dict)
    counters: Counter = field(default_factory=Counter)
    seconds: float = 0.

    def __post_init__(self):
        values=self.rgb.astype(np.float32)/255
        self.global_mean=values.mean(axis=(0,1));self.global_std=float(values.std())

    def patch(self,raw,pair):
        start,end=raw[0],raw[-1];length=float(np.linalg.norm(end-start))
        tangent=(end-start)/max(length,1e-9);normal=np.array([-tangent[1],tangent[0]])
        xx,yy=np.meshgrid(X,Y)
        points=start+length*(xx[...,None]*tangent+yy[...,None]*normal)
        mx=(points[...,0]-.5).astype(np.float32);my=(points[...,1]-.5).astype(np.float32)
        rgb=cv2.remap(self.rgb,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE).astype(np.float32)/255
        ids=cv2.remap(self.labels.astype(np.float32),mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=-1).astype(np.int32)
        a,b=pair
        mask=np.full((SIZE,SIZE),.5,np.float32);mask[ids==a]=1;mask[ids==b]=0
        valid=(ids==a)|(ids==b)
        projected=(raw-start)@np.stack((tangent,normal),axis=1)/length
        pixel=(projected+np.array([.25,.75]))/1.5*(SIZE-1)
        edge=np.zeros((SIZE,SIZE),np.uint8)
        cv2.polylines(edge,[np.rint(pixel*16).astype(np.int32)],False,255,2,cv2.LINE_AA,4)
        edge=cv2.GaussianBlur(edge.astype(np.float32)/255,(3,3),.65)
        patch=np.concatenate((rgb.transpose(2,0,1),mask[None],edge[None]),axis=0)
        ca=self.colors[a].astype(np.float32)/255;cb=self.colors[b].astype(np.float32)/255
        context=np.array([*self.global_mean,self.global_std,float(np.linalg.norm(ca-cb)),float(mask.mean()),min(1,length/96),1],np.float32)
        # Compare color reconstruction only near this boundary and its two regions.
        radius=max(1,min(14,int(np.ceil(3*SIZE/(1.5*length)))))
        band=cv2.dilate((edge>.1).astype(np.uint8),np.ones((radius*2+1,radius*2+1),np.uint8))>0
        valid &=band
        yraw=np.interp(X,np.sort(projected[:,0]),projected[np.argsort(projected[:,0]),1])
        positive=(Y[:,None]>yraw)
        side_a=float(mask[positive].mean())>float(mask[~positive].mean()) if positive.any() and (~positive).any() else True
        return patch,context,(start,length,tangent,normal),dict(rgb=rgb,valid=valid,ca=ca,cb=cb,side_a=side_a,edge=edge)

    def score_merge_pairs(self,chains,styles):
        areas=np.bincount(self.labels.ravel(),minlength=len(self.colors));seen=set()
        for chain in chains:
            pair=chain.paired_regions
            if len(pair)!=2 or pair in seen:continue
            seen.add(pair);a,b=pair
            if min(areas[a],areas[b])>48:continue
            if styles and any(styles.get(r)==(0.,0.) for r in pair):continue
            if np.linalg.norm(self.colors[a].astype(float)-self.colors[b].astype(float))>18:continue
            raw=np.asarray(chain.vertices,float)
            direction=raw[-1]-raw[0]
            if np.linalg.norm(direction)<1:direction=raw[len(raw)//2]-raw[0]
            length=np.linalg.norm(direction)
            if length<1:continue
            direction/=length;center=raw[len(raw)//2]
            span=np.array([center-8*direction,center+8*direction])
            patch,features,frame,data=self.patch(span,pair)
            mask=patch[3]>.5
            patch[4]=cv2.GaussianBlur(cv2.morphologyEx(mask.astype(np.float32),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8)),(3,3),.65)
            _,corner,probability=self.model.predict(patch,features)
            self.merge_hints[pair]=probability
            self.counters['merge_pair_inferences']+=1

    def report(self):
        return dict(provider='local-onnx-cnn',architecture=self.model.meta['architecture'],model_sha256=self.model.sha256,
                    synthetic_only=self.model.meta['synthetic_only'],inference_used=bool(self.counters['inference_calls'] or self.counters['merge_pair_inferences']),
                    output_contains_neural_segments=bool(self.counters['emitted_segments']),counters=dict(self.counters),seconds=self.seconds,
                    merge_proposals=[dict(regions=list(k),probability=v) for k,v in sorted(self.merge_hints.items())],
                    records=self.records,training=self.model.meta['test_metrics'])


def _render_controls(h):
    x,y=curve_samples(h)
    yy=np.interp(X,x[0],y[0])
    return 1/(1+np.exp(np.clip(-(Y[:,None]-yy[None,:])/.018,-50,50)))


def _render_curve(curves,frame):
    start,length,tangent,normal=frame
    points=(sample_curves(curves)-start)@np.stack((tangent,normal),axis=1)/length
    order=np.argsort(points[:,0]);yy=np.interp(X,points[order,0],points[order,1])
    return 1/(1+np.exp(np.clip(-(Y[:,None]-yy[None,:])/.018,-50,50)))


def _color_loss(mask,data):
    if not data['side_a']:mask=1-mask
    image=mask[...,None]*data['ca']+(1-mask[...,None])*data['cb']
    valid=data['valid']
    return float(np.abs(image-data['rgb'])[valid].mean()) if valid.any() else float('inf')


def _physical(h,frame):
    start,length,tangent,normal=frame
    c1=start+length*(h[0]*tangent+h[1]*normal)
    c2=start+length*(h[2]*tangent+h[3]*normal)
    end=start+length*tangent
    return [(start.copy(),c1,c2,end)]


def fit_neural_span(raw,pair,context,simplify,smooth,max_error):
    reference,level=fit_shared_curve(raw,simplify,smooth,max_error)
    context.counters['spans']+=1
    length=float(np.linalg.norm(raw[-1]-raw[0]))
    if length<7 or len(raw)<9:
        context.counters['short_or_small']+=1;return reference
    direction=(raw[-1]-raw[0])/length;progress=(raw-raw[0])@direction
    if np.max(np.maximum.accumulate(progress)-progress)>1.5:
        context.counters['nonmonotone']+=1;return reference
    patch,features,frame,data=context.patch(raw,pair)
    if data['valid'].mean()<.12:
        context.counters['insufficient_context']+=1;return reference
    initial,corner,merge=context.model.predict(patch,features)
    context.counters['inference_calls']+=1
    key=tuple(sorted(pair))
    if key not in context.merge_hints:context.merge_hints[key]=merge
    entry=dict(regions=list(pair),start=raw[0].tolist(),end=raw[-1].tolist(),corner_probability=corner,
               merge_probability=merge,source_vertices=len(raw),baseline_segments=len(reference))
    if corner>.80:
        context.counters['corner_rejected']+=1;entry.update(accepted=False,reason='predicted_corner');context.records.append(entry);return reference
    # Bounded coordinate descent refines the prediction against the source raster.
    h=initial.copy();best=_color_loss(_render_controls(h),data)
    entry['neural_initial_raster_error']=best
    entry['straight_raster_error']=_color_loss(_render_controls(np.array([.3,0,.7,0])),data)
    for step in range(context.optimize_steps):
        amount=.025*(.7**step)
        changed=False
        for j in range(4):
            for sign in (-1,1):
                candidate=h.copy();candidate[j]+=sign*amount
                candidate[0]=np.clip(candidate[0],.08,.48);candidate[2]=np.clip(candidate[2],.52,.92)
                candidate[[1,3]]=np.clip(candidate[[1,3]],-.8,.8)
                value=_color_loss(_render_controls(candidate),data)+.002*float(np.mean((candidate-initial)**2))
                if value<best-1e-8:h=candidate;best=value;changed=True
        if not changed:break
    curves=_physical(h,frame)
    before=_color_loss(_render_curve(reference,frame),data)
    after=_color_loss(_render_controls(h),data)
    # Extra control points have a cost; geometry and source-color budgets are hard.
    gain=before+.0012*len(reference)-(after+.0012)
    accepted=bool(_within(raw,curves,max_error) and after<=before+.006 and gain>1e-6)
    entry.update(accepted=accepted,reason='objective_improved' if accepted else 'geometry_or_raster_budget',
                 raster_before=before,raster_after=after,objective_delta=-gain,handles=h.tolist(),
                 neural_initial_handles=initial.tolist(),candidate_segments=1)
    context.records.append(entry);context.counters['curve_candidates']+=1
    context.counters['accepted' if accepted else 'rejected']+=1
    return curves if accepted else reference


def fit_neural_chain(raw,pair,context,simplify,smooth,max_error):
    started=time.monotonic()
    raw=np.asarray(raw,float)
    if simplify==0 and smooth==0 or max_error==0 or len(pair)!=2:
        return fit_shared_curve(raw,simplify,smooth,max_error)[0]
    # Split at bounded arc-length intervals; shared junction endpoints never move.
    count=len(raw)-1
    spans=max(1,int(np.ceil(count/28)))
    if np.array_equal(raw[0],raw[-1]):spans=max(4,spans)
    cuts=np.unique(np.rint(np.linspace(0,count,spans+1)).astype(int))
    curves=[]
    for a,b in zip(cuts,cuts[1:]):
        if b>a:curves.extend(fit_neural_span(raw[a:b+1],pair,context,simplify,smooth,max_error))
    context.seconds+=time.monotonic()-started
    return curves


def conservative_merge(seg,rgb,alpha,locked,importance,probabilities,preservation):
    """Apply only high-confidence weak-seam hints under original hard constraints.

    This is one disjoint union pass, <=24 merges, <=2 added RGB MSE/channel.
    The caller must invalidate/rebuild every geometry cache and render-gate the result.
    """
    from .base_segmentation import rag_statistics
    from .palette import Segmentation
    graph=rag_statistics(seg.labels,rgb,importance,alpha,locked,preservation)
    n=len(seg.sizes);parent=np.arange(n,dtype=np.int32)
    area=seg.sizes.astype(np.float64)
    sums=np.stack([np.bincount(seg.labels.ravel(),weights=rgb[...,c].ravel(),minlength=n) for c in range(3)],axis=1)
    budget=2.*rgb.size;added=0.;used=set();log=[]
    for (a,b),probability in sorted(probabilities.items(),key=lambda p:(-p[1],p[0])):
        if probability<.97:continue
        edge=graph.get((a,b));reason=None
        if edge is None:reason='not_adjacent'
        elif edge.protected:reason='protected'
        elif a in used or b in used:reason='already_changed'
        elif min(area[a],area[b])>48:reason='not_small'
        elif edge.importance_sum/max(1,edge.length)>.35:reason='important_edge'
        delta=0.
        if reason is None:
            distance=float(np.linalg.norm(sums[a]/area[a]-sums[b]/area[b]))
            delta=float(area[a]*area[b]/(area[a]+area[b])*distance**2)
            if distance>18:reason='color_distance'
            elif added+delta>budget:reason='distortion_budget'
        accepted=reason is None
        log.append(dict(regions=[a,b],probability=float(probability),accepted=accepted,reason=reason or 'weak_seam'))
        if not accepted:continue
        keep,drop=(a,b) if area[a]>=area[b] else (b,a)
        parent[drop]=keep;used.update((a,b));added+=delta
        if len(used)>=48:break
    _,index=np.unique(parent,return_inverse=True)
    labels=index[seg.labels].astype(np.int32);sizes=np.bincount(labels.ravel())
    totals=np.stack([np.bincount(labels.ravel(),weights=rgb[...,c].ravel()) for c in range(3)],axis=1)
    colors=np.rint(totals/sizes[:,None]).clip(0,255).astype(np.uint8)
    return Segmentation(labels,labels,colors,sizes),dict(before=n,after=len(sizes),merged=n-len(sizes),
        added_mse=added/rgb.size,proposals=log,render_accepted=False)


def preview_partition(seg,alpha,shared,rings):
    """Actual SVG base render for global merge acceptance; no raster embeddings."""
    from html import escape
    from .hierarchy import VectorDocument,VectorShape
    colors_alpha=np.zeros(len(seg.sizes),np.uint8);colors_alpha[seg.labels.ravel()]=alpha.ravel()
    paths=shared.paths(rings);h,w=seg.labels.shape;shapes=[]
    for rid in sorted(paths):
        if colors_alpha[rid]==0:continue
        shapes.append(VectorShape(f'probe-{rid}',[rid],'primary','base_planes',
            (*map(int,seg.region_colors[rid]),int(colors_alpha[rid])),
            'fill-rule="evenodd" d="'+escape(' '.join(paths[rid]),quote=True)+'"','path',
            (0,0,w,h),int(seg.sizes[rid]),draw_order=len(shapes)))
    opaque=bool(np.all(alpha==255));settings=dict(alpha_exact=not opaque,seam_underpaint=opaque)
    if opaque:
        color=seg.region_colors[int(np.argmax(seg.sizes))]
        settings['opaque_backdrop']='#'+''.join(f'{int(v):02x}' for v in color)
    from .metrics import render_svg
    return render_svg(VectorDocument(w,h,shapes,[],settings).svg().encode()),len(shapes)
