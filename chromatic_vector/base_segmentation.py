"""Spatially connected palette proposals and versioned, hard-protected RAG merges.

All merges operate on the partition. Alpha and manual locks are hard constraints;
protected boundary statistics survive union without dilution by weak edges.
"""
from dataclasses import dataclass
import heapq
import cv2
import numpy as np
from .palette import Segmentation, quantize, edge_aware_refine, connected_regions, _oklab

@dataclass
class BoundaryStats:
    length: int
    gradient_sum: float
    importance_sum: float
    strong_count: int
    protected: bool


def propose(flat, reference, alpha, config, locked, correction_rgb):
    # Quantize only visible colors; invisible pixels have an isolated palette ID.
    visible = alpha > 0
    work = flat.copy()
    if not visible.any():
        ids = np.zeros(alpha.shape,np.int32)
    else:
        sample = flat[visible].reshape(1,-1,3)
        colors = min(config.base_colors, sample.shape[1])
        if colors < 2:
            ids = np.zeros(alpha.shape,np.int32)
        else:
            # Reuse deterministic v0.5 OKLab quantization without hidden pixels.
            palette_ids = quantize(sample, colors, config.seed).ravel()
            # Get centers in OKLab then assign on the full registered raster.
            lab = _oklab(flat)
            visible_lab = lab[visible]
            centers = np.array([visible_lab[palette_ids==i].mean(axis=0) if np.any(palette_ids==i)
                                else visible_lab[0] for i in range(colors)],np.float32)
            ids = np.zeros(alpha.shape,np.int32)
            for y in range(0,len(flat),64):
                ids[y:y+64] = np.sum((lab[y:y+64,:,None,:]-centers[None,None,:,:])**2,axis=-1).argmin(axis=-1)
            ids = edge_aware_refine(work,ids,passes=1 if config.preset=='detail' else 2)
    ids[~visible] = int(ids.max())+1
    # Exact opacity strata: no alpha quantization, including antialiased rims.
    combined = ids.astype(np.int64)*256+alpha
    if locked.any():
        packed = (correction_rgb[...,0].astype(np.int64)<<16)+(correction_rgb[...,1].astype(np.int64)<<8)+correction_rgb[...,2]
        combined[locked] = (int(combined.max())+1)+packed[locked]*256+alpha[locked]
    _, dense = np.unique(combined, return_inverse=True)
    return connected_regions(dense.reshape(alpha.shape).astype(np.int32),reference)


def rag_statistics(labels, rgb, importance, alpha, locked, strength):
    n = int(labels.max())+1
    records = []
    for a,b,ra,rb,ia,ib in ((labels[:,:-1],labels[:,1:],rgb[:,:-1],rgb[:,1:],importance[:,:-1],importance[:,1:]),
                            (labels[:-1],labels[1:],rgb[:-1],rgb[1:],importance[:-1],importance[1:])):
        different = a != b
        aa,bb = a[different].astype(np.int64), b[different].astype(np.int64)
        keys = np.minimum(aa,bb)*n+np.maximum(aa,bb)
        grad = np.linalg.norm(_oklab(ra)[different]-_oklab(rb)[different],axis=-1)
        imp = np.maximum(ia,ib)[different]
        records.append((keys,grad,imp))
    key = np.concatenate([r[0] for r in records])
    if not len(key): return {}
    unique,inv,count = np.unique(key,return_inverse=True,return_counts=True)
    gs = np.bincount(inv,weights=np.concatenate([r[1] for r in records]))
    im = np.bincount(inv,weights=np.concatenate([r[2] for r in records]))
    strong = np.bincount(inv,weights=np.concatenate([r[1] for r in records]) > (.16-.10*strength))
    sizes=np.bincount(labels.ravel(),minlength=n)
    region_alpha = np.zeros(n,np.uint8); region_alpha[labels.ravel()] = alpha.ravel()
    region_lock = np.bincount(labels.ravel(),weights=locked.ravel(),minlength=n)>0
    result = {}
    for k,c,g,i,s in zip(unique,count,gs,im,strong):
        a,b = int(k//n),int(k%n)
        protected = (region_alpha[a]!=region_alpha[b] or region_lock[a] or region_lock[b] or
                     (c>=3 and s>=3 and s/c>=.4 and min(sizes[a],sizes[b])>=8 and g/c>(.13-.06*strength)))
        result[a,b] = BoundaryStats(int(c),float(g),float(i),int(s),bool(protected))
    return result


def merge_planes(seg, reference, importance, alpha, locked, config):
    n = len(seg.sizes)
    graph = rag_statistics(seg.labels,reference,importance,alpha,locked,config.boundary_preservation)
    initial = {pair:vars(stats).copy() for pair,stats in graph.items()}
    parent = np.arange(n,dtype=np.int32)
    version = np.zeros(n,np.int64)
    area = seg.sizes.copy()
    sums = np.stack([np.bincount(seg.labels.ravel(),weights=reference[...,c].ravel(),minlength=n) for c in range(3)],axis=-1)
    neighbors = {i:{} for i in range(n)}
    for (a,b),st in graph.items(): neighbors[a][b]=st; neighbors[b][a]=st
    queue = []
    scheduled = {}
    peak_queue = 0
    def push(a,b):
        if a>b: a,b=b,a
        st = neighbors[a].get(b)
        if st is None or st.protected: return
        token=(int(version[a]),int(version[b]))
        if scheduled.get((a,b))==token: return
        scheduled[a,b]=token
        d2 = float(np.sum((sums[a]/area[a]-sums[b]/area[b])**2))
        delta = float(area[a])*float(area[b])/(area[a]+area[b])*d2
        # Boundary strength + compactness discourage erasing long thin accents.
        score = delta*(1+4*st.importance_sum/max(1,st.length))*(1+.03*np.sqrt(min(area[a],area[b]))/st.length)
        heapq.heappush(queue,(score,a,b,int(version[a]),int(version[b]),delta))
    for a,b in graph: push(a,b)
    target = config.target_shapes or 1
    budget = {'detail': 65, 'draft': 130, 'poster': 320}[config.preset] * reference.size
    added = 0.; joined = 0
    while queue and n-joined > target:
        peak_queue=max(peak_queue,len(queue))
        _,a,b,va,vb,delta = heapq.heappop(queue)
        if scheduled.get((a,b))!=(va,vb): continue
        scheduled.pop((a,b),None)
        if not area[a] or not area[b]: continue
        if version[a]!=va or version[b]!=vb:
            # Refresh on demand. Do not enqueue the entire high-degree background
            # after every pixel-island union (quadratic memory in the old queue).
            push(a,b)
            continue
        if added+delta > budget: continue
        if area[a]<area[b] or (area[a]==area[b] and a>b): a,b=b,a
        parent[b]=a; area[a]+=area[b]; area[b]=0; sums[a]+=sums[b]
        added+=delta; joined+=1; version[a]+=1; version[b]+=1
        neighbors[a].pop(b,None)
        changed=set(neighbors[b])- {a}
        for other,st in list(neighbors[b].items()):
            if other==a: continue
            neighbors[other].pop(b,None)
            old = neighbors[a].get(other)
            if old:
                st = BoundaryStats(st.length+old.length,st.gradient_sum+old.gradient_sum,
                                   st.importance_sum+old.importance_sum,st.strong_count+old.strong_count,
                                   st.protected or old.protected)
            neighbors[a][other]=st; neighbors[other][a]=st
        neighbors[b].clear()
        for other in sorted(changed): push(a,other)
    for i in range(n):
        j=i
        while parent[j]!=j: j=int(parent[j])
        parent[i]=j
    _,index=np.unique(parent,return_inverse=True)
    labels=index[seg.labels].astype(np.int32)
    sizes=np.bincount(labels.ravel())
    roots=np.unique(parent)
    colors=np.rint(sums[roots]/area[roots,None]).clip(0,255).astype(np.uint8)
    result=Segmentation(labels,labels,colors,sizes)
    return result,dict(before=n,after=len(sizes),joined=joined,added_mse=added/reference.size,
                       protected_edges=sum(v['protected'] for v in initial.values()),
                       queue_policy='lazy_endpoint_refresh',peak_queue_entries=peak_queue),initial
