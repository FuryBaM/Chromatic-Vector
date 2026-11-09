"""Metrics on actual SVG renders, sRGB over white; no source filtering."""
from io import BytesIO
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from .palette import _oklab


def render_svg(svg: str | Path | bytes) -> np.ndarray:
    import cairosvg
    data = svg if isinstance(svg, bytes) else Path(svg).read_bytes()
    return np.asarray(Image.open(BytesIO(cairosvg.svg2png(bytestring=data))).convert('RGBA')).copy()


def composite(rgba):
    a = rgba[..., 3:4].astype(np.float32) / 255
    return rgba[..., :3].astype(np.float32)*a + 255*(1-a)


def edge_mask(rgb):
    gray = cv2.cvtColor(np.rint(rgb).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return cv2.Canny(gray, 40, 100) > 0


def quality_metrics(source, rendered):
    if source.shape != rendered.shape:
        raise ValueError('Metrics require identical RGBA canvas dimensions')
    a, b = composite(source), composite(rendered)
    ae, be = edge_mask(a), edge_mask(b)
    f1 = {}
    for tol in (1, 2, 3):
        k = np.ones((tol*2+1, tol*2+1), np.uint8)
        precision = float(np.count_nonzero(be & cv2.dilate(ae.astype(np.uint8), k).astype(bool)) / max(1, be.sum()))
        recall = float(np.count_nonzero(ae & cv2.dilate(be.astype(np.uint8), k).astype(bool)) / max(1, ae.sum()))
        f1[str(tol)] = 2*precision*recall/max(1e-12, precision+recall)
    # Alpha silhouette when available; otherwise fixed source-corner background.
    # Reusing the source reference prevents each output inventing its own mask.
    if np.any(source[...,3] != 255):
        af=source[...,3]>127; bf=rendered[...,3]>127
    else:
        background=np.median(np.array((a[0,0],a[0,-1],a[-1,0],a[-1,-1])),axis=0)
        af=np.linalg.norm(a-background,axis=-1)>24
        bf=np.linalg.norm(b-background,axis=-1)>24
    union=int((af|bf).sum())
    lab=_oklab(a.astype(np.uint8))
    gx=cv2.Sobel(lab,cv2.CV_32F,1,0,ksize=3)/4
    gy=cv2.Sobel(lab,cv2.CV_32F,0,1,ksize=3)/4
    weight=np.sqrt(np.sum(gx*gx+gy*gy,axis=-1))*ae
    missing=ae & ~(cv2.dilate(be.astype(np.uint8),np.ones((3,3),np.uint8))>0)
    edge_loss=float((weight*missing).sum()/max(1e-12,weight.sum()))
    return dict(rgb_mae=float(np.abs(a-b).mean()),
                oklab_delta=float(np.linalg.norm(_oklab(a.astype(np.uint8))-_oklab(b.astype(np.uint8)), axis=-1).mean()),
                edge_f1=f1, silhouette_iou=float((af & bf).sum()/union) if union else 1.,
                weighted_edge_loss=edge_loss,
                alpha_mae=float(np.abs(source[...,3].astype(float)-rendered[...,3]).mean()))


def residual(source, rendered, importance):
    a, b = composite(source), composite(rendered)
    color = np.linalg.norm(_oklab(a.astype(np.uint8))-_oklab(b.astype(np.uint8)), axis=-1)
    ea, eb = edge_mask(a), edge_mask(b)
    mismatch = (ea != eb).astype(np.float32)
    result = np.zeros(color.shape, np.float32)
    for scale, weight in ((1, .5), (2, .3), (4, .2)):
        size = (max(1, color.shape[1]//scale), max(1, color.shape[0]//scale))
        c = cv2.resize(color, size, interpolation=cv2.INTER_AREA)
        result += weight*cv2.resize(c, (color.shape[1],color.shape[0]), interpolation=cv2.INTER_LINEAR)
    return result*(1+.5*importance) + .08*mismatch + .2*np.abs(source[...,3].astype(float)-rendered[...,3])/255


def objective(source, rendered, importance, complexity=0):
    return float(residual(source, rendered, importance).mean() + .000003*complexity)
