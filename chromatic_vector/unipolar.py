"""Standalone Unipolar-Gambit contour methods, with SVG topology/error guards.

Verified against FuryBaM/Unipolar-Gambit commit
6a826d823555baf789d68fced3825d5a26037527, BorderBuilder.cs:
AdaptiveFilterPolyline, SimplifyClosedPolyline, SmoothPolyline.
MIT notice: licenses/Unipolar-Gambit-MIT.txt. No Unity runtime dependency.
"""
import numpy as np
from scipy.spatial import cKDTree
from .shared import adaptive_filter_py
from .curve_fitting import _cubics, sample_curves, fit_shared_curve

METHODS=('bounded','unipolar-adaptive','unipolar-chaikin')


def chaikin(points, closed):
    """The source SmoothPolyline's exact 3/4, 1/4 corner-cutting pass."""
    p=np.asarray(points,float)
    if len(p)<3:return p.copy()
    a=p if closed else p[:-1];b=np.roll(p,-1,axis=0) if closed else p[1:]
    q=.75*a+.25*b;r=.25*a+.75*b
    result=np.stack((q,r),axis=1).reshape(-1,2)
    return result if closed else np.vstack((p[0],result,p[-1]))


def _rdp_open(points,tolerance):
    # Source uses distance to the infinite chord, not distance to pixel vertices.
    p=np.asarray(points,float)
    if len(p)<=2:return p.copy()
    keep=np.zeros(len(p),bool);keep[[0,-1]]=True;stack=[(0,len(p)-1)]
    while stack:
        a,b=stack.pop()
        if b<=a+1:continue
        v=p[b]-p[a];length=np.linalg.norm(v);delta=p[a+1:b]-p[a]
        d=np.linalg.norm(delta,axis=1) if length<1e-6 else np.abs(delta[:,0]*v[1]-delta[:,1]*v[0])/length
        index=int(np.argmax(d))+a+1
        if d[index-a-1]>tolerance:
            keep[index]=True;stack.extend(((a,index),(index,b)))
    return p[keep]


def simplify(points,closed,tolerance):
    p=np.asarray(points,float)
    if not closed or len(p)<=3:return _rdp_open(p,tolerance) if not closed else p.copy()
    split=int(np.argmax(np.sum((p-p[0])**2,axis=1)))
    a=_rdp_open(p[:split+1],tolerance)
    b=_rdp_open(np.vstack((p[split:],p[0])),tolerance)
    out=np.vstack((a,b[1:-1]))
    return out if len(out)>=3 else p.copy()


def _within(raw,curves,tolerance):
    samples=sample_curves(curves)
    if not len(samples):return False
    if cKDTree(samples).query(raw)[0].max()>tolerance:return False
    tree=cKDTree(raw);_,i=tree.query(samples,k=min(4,len(raw)))
    i=np.asarray(i).reshape(len(samples),-1)
    starts=np.concatenate((np.clip(i-1,0,len(raw)-2),np.clip(i,0,len(raw)-2)),axis=1)
    a=raw[starts];v=raw[starts+1]-a
    t=np.clip(np.sum((samples[:,None]-a)*v,axis=-1)/np.maximum(1e-15,np.sum(v*v,axis=-1)),0,1)
    distance=np.linalg.norm(samples[:,None]-(a+t[...,None]*v),axis=-1).min(axis=1)
    return bool(distance.max()<=tolerance)


def fit_unipolar(raw,tolerance,strength,max_error,method='unipolar-chaikin'):
    raw=np.asarray(raw,float)
    if max_error==0 or (tolerance==0 and strength==0):return fit_shared_curve(raw,0,0,0)
    closed=np.array_equal(raw[0],raw[-1]);core=raw[:-1] if closed else raw
    for attempt,factor in enumerate((1.,.6,.3)):
        if method=='unipolar-adaptive':
            # Current source's sharp-turn threshold is 1.5, not legacy CV's .65.
            pts=adaptive_filter_py(core,closed,min(tolerance,max_error)*factor,strength*factor,1.5)
        else:
            pts=simplify(core,closed,min(tolerance,max_error)*factor)
            if strength>0:
                # The kernel is exact; lower retries reduce displacement toward
                # the original polygon while keeping the same subdivision.
                cut=chaikin(pts,closed)
                if factor<1:
                    if closed:
                        anchor=np.repeat(pts,2,axis=0)
                    else:
                        anchor=np.vstack((pts[0],np.repeat(pts[:-1],2,axis=0),pts[-1]))
                    cut=anchor+(cut-anchor)*factor
                pts=cut
        if len(pts)<(3 if closed else 2):continue
        if closed:pts=np.vstack((pts,pts[0]))
        else:pts[0]=raw[0];pts[-1]=raw[-1]
        curves=_cubics(pts,strength*factor,set())
        if _within(raw,curves,max_error):return curves,attempt
    # Refuse an unbounded approximation; exact collinear reduction is safe.
    return fit_shared_curve(raw,0,0,0)[0],3
