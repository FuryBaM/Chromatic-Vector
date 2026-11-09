"""Ordered shared Bézier fits with pixel-uncertainty bounds.

Kept separate from the legacy fitter so precise SVG output remains byte-compatible.
"""
import math
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree


def sample_curves(curves, spacing=.25):
    chunks=[]
    for a,c1,c2,b in curves:
        length=np.linalg.norm(c1-a)+np.linalg.norm(c2-c1)+np.linalg.norm(b-c2)
        # No fixed 48-sample cap: a 1500px straight edge needs a dense guard too.
        count=max(2,int(math.ceil(length/spacing)))
        t=np.linspace(0,1,count+1)[1:,None];u=1-t
        chunks.append(u**3*a+3*u*u*t*c1+3*u*t*t*c2+t**3*b)
    return np.vstack((curves[0][0],*chunks)) if chunks else np.zeros((0,2))


def _ordered_reduce(raw, epsilon, smooth):
    closed=np.array_equal(raw[0],raw[-1])
    core=raw[:-1] if closed else raw
    # Identify actual corners from long straight runs, not 1px staircase turns.
    linear=cv2.approxPolyDP(core.astype(np.float32),0,closed).reshape(-1,2)
    sharp=set()
    for i,p in enumerate(linear):
        if not closed and i in (0,len(linear)-1):continue
        va=p-linear[i-1];vb=linear[(i+1)%len(linear)]-p
        da,db=np.linalg.norm(va),np.linalg.norm(vb)
        if min(da,db)>=3 and np.dot(va,vb)/(da*db)<.5:sharp.add(tuple(p))
    if smooth>0 and len(core)>5:
        filtered=gaussian_filter1d(core,sigma=.85,axis=0,mode='wrap' if closed else 'nearest')
        pts=core+(filtered-core)*smooth
    else:pts=core.copy()
    pins=[0]+[i for i,p in enumerate(core) if tuple(p) in sharp]
    if closed:
        # Split in traversal order. Never prepend an absent closed-ring anchor to
        # an arbitrarily rotated approxPolyDP result (that creates backtracking).
        pins.append(int(np.argmax(np.sum((core-core[0])**2,axis=1))))
        pts=np.vstack((pts,pts[0]));pins.append(len(core))
    else:pins.append(len(core)-1)
    pins=sorted(set(pins))
    for i in pins:pts[i]=raw[i]
    groups=[]
    for a,b in zip(pins,pins[1:]):
        section=cv2.approxPolyDP(pts[a:b+1].astype(np.float32),epsilon,False).reshape(-1,2).astype(float)
        section[0]=pts[a];section[-1]=pts[b]
        groups.extend(section if not groups else section[1:])
    result=np.asarray(groups,float)
    if closed and len(result)<4:return raw.copy(),sharp
    return result,sharp


def _cubics(points, smooth, sharp):
    closed=np.array_equal(points[0],points[-1]);n=len(points)-1
    tangent=np.zeros_like(points)
    for i in (range(n) if closed else range(1,n)):
        if tuple(points[i]) in sharp:continue
        prev=points[(i-1)%n] if closed else points[i-1]
        nxt=points[(i+1)%n] if closed else points[i+1]
        va=points[i]-prev;vb=nxt-points[i]
        da,db=np.linalg.norm(va),np.linalg.norm(vb)
        if min(da,db)<1e-9:continue
        # Keep unambiguous cusps. Moderate turns on a sampled arc stay curved.
        if min(da,db)>3 and np.dot(va,vb)/(da*db)<.3:continue
        t=(nxt-prev)*.5*smooth
        mag=np.linalg.norm(t);limit=min(da,db)*1.2
        tangent[i]=t*min(1.,limit/max(1e-12,mag))
    if closed:tangent[-1]=tangent[0]
    return [(points[i],points[i]+tangent[i]/3,points[i+1]-tangent[i+1]/3,points[i+1]) for i in range(n)]


def fit_shared_curve(raw, simplify, smooth, max_error):
    raw=np.asarray(raw,float)
    exact=simplify==0 and smooth==0
    thin=np.array_equal(raw[0],raw[-1]) and min(np.ptp(raw,axis=0))<=1
    if exact or max_error==0:
        pts,sharp=_ordered_reduce(raw,0,0)
        return _cubics(pts,0,sharp),0
    source_tree=cKDTree(raw)
    for attempt,(epsilon,strength) in enumerate(((min(simplify,max_error),smooth),
             (min(simplify,max_error)*.6,smooth*.65),(min(simplify,max_error)*.3,smooth*.3),(0,0))):
        if thin:strength=0;epsilon=min(epsilon,.45)
        pts,sharp=_ordered_reduce(raw,epsilon,strength)
        curves=_cubics(pts,strength,sharp)
        if attempt==3:return curves,3
        samples=sample_curves(curves)
        # Raw grid vertices are <= .5px apart from their true segment. Checking
        # point-to-segment projection avoids falsely rejecting a straight fit.
        _,indices=source_tree.query(samples,k=min(4,len(raw)))
        indices=np.asarray(indices).reshape(len(samples),-1)
        starts=np.concatenate((np.clip(indices-1,0,len(raw)-2),np.clip(indices,0,len(raw)-2)),axis=1)
        a=raw[starts];v=raw[starts+1]-a
        t=np.clip(np.sum((samples[:,None,:]-a)*v,axis=-1)/np.maximum(1e-15,np.sum(v*v,axis=-1)),0,1)
        forward=np.sqrt(np.sum((samples[:,None,:]-(a+t[...,None]*v))**2,axis=-1)).min(axis=1).max()
        backward=cKDTree(samples).query(raw)[0].max()
        if max(forward,backward)<=max_error:return curves,attempt
    raise AssertionError('unreachable')
