"""Real SVG primitives with measured mask IoU and boundary distance guards."""
from html import escape
import cv2
import numpy as np
from scipy.spatial import cKDTree
from .svg_shapes import _as_linear_polygon, _rectangle_of, _FMT


def ellipse_fit(mask, ring, policy='auto'):
    if policy=='off' or len(ring)<12 or min(mask.shape)<5: return None
    # Holes and cutouts cannot pass the high IoU and max-distance gates.
    padded=np.pad(mask.astype(np.uint8),1)
    count=cv2.connectedComponents(padded,connectivity=4)[0]-1
    contours,hierarchy=cv2.findContours(padded,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    if count!=1 or hierarchy is None or len(contours)!=1: return None
    try: (cx,cy),(dx,dy),angle=cv2.fitEllipse(ring.astype(np.float32))
    except cv2.error: return None
    rx,ry=dx/2,dy/2
    if min(rx,ry)<=1 or not np.isfinite((cx,cy,rx,ry,angle)).all(): return None
    yy,xx=np.indices(mask.shape,dtype=float);xx+=.5;yy+=.5
    rad=np.deg2rad(angle);cos,sin=np.cos(rad),np.sin(rad)
    x=(xx-cx)*cos+(yy-cy)*sin; y=-(xx-cx)*sin+(yy-cy)*cos
    fitted=(x/rx)**2+(y/ry)**2<=1
    iou=float((fitted & mask).sum()/max(1,(fitted | mask).sum()))
    t=np.linspace(0,2*np.pi,max(96,int(2*np.pi*max(rx,ry)*3)),endpoint=False)
    points=np.column_stack((cx+rx*np.cos(t)*cos-ry*np.sin(t)*sin,cy+rx*np.cos(t)*sin+ry*np.sin(t)*cos))
    distances=np.concatenate((cKDTree(points).query(ring)[0],cKDTree(ring).query(points)[0]))
    p95=float(np.quantile(distances,.95)); maximum=float(distances.max())
    min_iou=.995 if policy=='strict' else .985
    if iou<min_iou or p95>.75 or maximum>1.0: return None
    return dict(cx=float(cx),cy=float(cy),rx=float(rx),ry=float(ry),angle=float(angle),
                iou=iou,boundary_p95=p95,boundary_max=maximum)


def ellipse_attributes(fit):
    f=lambda x:f'{x:.6f}'
    cx,cy,rx,ry,angle=(fit[k] for k in ('cx','cy','rx','ry','angle'))
    return f'cx="{f(cx)}" cy="{f(cy)}" rx="{f(rx)}" ry="{f(ry)}" transform="rotate({f(angle)} {f(cx)} {f(cy)})"'


def ellipse_path(fit, positive=True):
    cx,cy,rx,ry,angle=(fit[k] for k in ('cx','cy','rx','ry','angle'))
    rad=np.deg2rad(angle); x=rx*np.cos(rad); y=rx*np.sin(rad)
    f=lambda v:f'{v:.6f}'
    return (f'M {f(cx+x)} {f(cy+y)} A {f(rx)} {f(ry)} {f(angle)} 1 {int(positive)} {f(cx-x)} {f(cy-y)} '
            f'A {f(rx)} {f(ry)} {f(angle)} 1 {int(positive)} {f(cx+x)} {f(cy+y)} Z')


def linear_primitive(paths,shape,policy):
    if policy!='off' and len(paths)==1 and not shape.holes and len(shape.outer)==1:
        poly=_as_linear_polygon(paths[0],shape.pixel_area)
        if poly is not None:
            if _rectangle_of(poly,shape.bbox):
                x,y,x1,y1=shape.bbox
                return 'rect',f'x="{x}" y="{y}" width="{x1-x}" height="{y1-y}"'
            return 'polygon','points="'+' '.join(f'{_FMT(x)},{_FMT(y)}' for x,y in poly)+'"'
    return 'path','fill-rule="evenodd" d="'+escape(' '.join(paths),quote=True)+'"'
