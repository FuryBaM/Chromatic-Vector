"""Edge-aware region agglomeration with an explicit shape budget.

This operates on the *shared* region-ID raster before contour tracing. A merge
therefore removes the common edge exactly once and never introduces independent
SVG overlaps or hairline seams. It is deterministic for equal input and seed.

The budget controls added RGB mean-squared error per channel relative to the
initial region reconstruction. It is not a guarantee on perceived similarity:
we report the actual RGB MAE after the final reconstruction too.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .palette import Segmentation


@dataclass
class AgglomerationStats:
    before: int
    after: int
    target: int
    joined: int
    added_mse: float
    color_mae_delta: float
    target_met: bool
    protected_edges: int


def _adjacency(labels: np.ndarray, rgb: np.ndarray, n: int):
    """Return edge length and mean gradient for every unordered adjacent pair."""
    keys = []
    contrasts = []
    for al, bl, ar, br in (
        (labels[:, :-1], labels[:, 1:], rgb[:, :-1], rgb[:, 1:]),
        (labels[:-1, :], labels[1:, :], rgb[:-1, :], rgb[1:, :]),
    ):
        m = al != bl
        if not np.any(m):
            continue
        a, b = al[m].astype(np.int64), bl[m].astype(np.int64)
        keys.append(np.minimum(a,b)*n + np.maximum(a,b))
        contrasts.append(np.mean(np.abs(ar[m].astype(np.int16) - br[m].astype(np.int16)),axis=1))
    if not keys:
        return []
    key = np.concatenate(keys)
    contrast = np.concatenate(contrasts)
    uniq, inverse, count = np.unique(key, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse,weights=contrast,minlength=len(uniq))
    return [(int(k//n),int(k%n),int(c),float(s)) for k,c,s in zip(uniq,count,sums)]


def agglomerate_to_budget(seg: Segmentation, rgb: np.ndarray, target: int,
                          max_added_mse: float = 32.0, edge_protection: float = 2.5,
                          hard_edge: float = 42.0, hard_color: float = 50.0,
                          protect_area: int = 6) -> tuple[Segmentation, AgglomerationStats]:
    """Fuse lowest-cost adjacent regions until target or distortion budget.

    Priorities penalize true input-image edges. Discontinuities with both a
    strong local gradient and a high mean-color difference are protected if the
    region is sizeable. Remaining merges minimize increased within-region SSE.

    Args:
        target: desired upper bound on number of regions, 1..original count
        max_added_mse: extra per-pixel per-channel squared RGB reconstruction error
        edge_protection: soft penalty on strong image edges, 0 turns it off
        hard_edge/hard_color/protect_area: prevent deletion of salient boundaries
    """
    if target < 1 or max_added_mse < 0 or edge_protection < 0:
        raise ValueError('target >= 1, max_added_mse >= 0, edge_protection >= 0 required')
    n = len(seg.sizes)
    if n <= target:
        return seg, AgglomerationStats(n,n,target,0,0.,0.,True,0)
    edges = _adjacency(seg.labels,rgb,n)
    # Exact unrounded pixel means are used for merge costs.
    flat=seg.labels.ravel()
    area=np.bincount(flat,minlength=n).astype(np.int64)
    sums=np.array([np.bincount(flat,weights=rgb[:,:,c].ravel(),minlength=n) for c in range(3)],dtype=np.float64).T
    parent=np.arange(n,dtype=np.int32)

    def root(i):
        j=i
        while parent[j]!=j:
            j=int(parent[j])
        while i!=j:
            k=int(parent[i]); parent[i]=j; i=k
        return j

    def pair_score(a,b,length,grad_sum):
        ma,mb=sums[a]/area[a],sums[b]/area[b]
        d2=float(np.sum((ma-mb)**2))
        delta=float(float(area[a])*float(area[b])/(area[a]+area[b])*d2)
        edge=grad_sum/length
        protected=(edge>=hard_edge and np.sqrt(d2)>=hard_color and
                   min(area[a],area[b])>=protect_area)
        factor=(1.+edge_protection*(edge/40.)**2)
        factor*=1.+.05*max(1.,min(area[a],area[b])**.5)/max(1.,length)
        return delta*factor,delta,protected

    joined=0
    added_sse=0.
    budget_sse=max_added_mse*float(rgb.shape[0]*rgb.shape[1]*3)
    protected_edges=0
    # Rebuild the adjacency *between root components* once per pass instead of
    # pushing every neighbor again after each tiny merge. This bounds the queue
    # size by the number of initial graph edges even for a huge background node.
    for _round in range(64):
        graph={}
        for a,b,length,grad in edges:
            a=root(a);b=root(b)
            if a==b: continue
            if b<a: a,b=b,a
            prev=graph.get((a,b))
            if prev is None: graph[(a,b)]=(length,grad)
            else: graph[(a,b)]=(prev[0]+length,prev[1]+grad)
        if not graph: break
        ranked=[]
        for (a,b),(length,grad_sum) in graph.items():
            priority,delta,protected=pair_score(a,b,length,grad_sum)
            if protected:
                protected_edges+=1
            else:
                ranked.append((priority,a,b,length,grad_sum))
        ranked.sort(key=lambda item:(item[0],item[1],item[2]))
        round_joined=0
        touched=set()
        for _,aa,bb,length,grad_sum in ranked:
            a=root(aa);b=root(bb)
            if a==b or a in touched or b in touched: continue
            # The current pair can have changed while the pass was running.
            # The true shared boundary is rebuilt at the next pass, so a
            # stale edge still serves as proof of physical adjacency.
            _,delta,protected=pair_score(a,b,length,grad_sum)
            if protected or added_sse+delta>budget_sse: continue
            if area[a]<area[b] or (area[a]==area[b] and a>b):a,b=b,a
            parent[b]=a
            area[a]+=area[b];area[b]=0
            sums[a]+=sums[b]
            added_sse+=delta
            joined+=1;round_joined+=1
            touched.add(a);touched.add(b)
            if joined>=n-target:break
        if joined>=n-target or round_joined==0:break

    # Compress transitive parent links.
    for j in range(n):
        k=j
        while parent[k]!=k: k=int(parent[k])
        parent[j]=k
    _,index=np.unique(parent,return_inverse=True)
    index=index.astype(np.int32)
    out=index[seg.labels]
    new_n=n-joined
    sizes=np.bincount(out.ravel(),minlength=new_n)
    color_sums=np.array([np.bincount(out.ravel(),weights=rgb[:,:,c].ravel(),minlength=new_n) for c in range(3)]).T
    colors=np.rint(color_sums/sizes[:,None]).clip(0,255).astype(np.uint8)
    before_mae=np.mean(np.abs(seg.region_colors[seg.labels].astype(np.float32)-rgb.astype(np.float32)))
    after_mae=np.mean(np.abs(colors[out].astype(np.float32)-rgb.astype(np.float32)))
    result=Segmentation(out,seg.palette_ids,colors,sizes)
    return result,AgglomerationStats(n,new_n,target,joined,added_sse/(rgb.shape[0]*rgb.shape[1]*3),float(after_mae-before_mae),new_n<=target,protected_edges)
