"""Immutable source, alpha-safe edge-aware flattening and registered pyramids."""
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from .palette import _oklab


def pyramid(image):
    h,w = image.shape[:2]
    return [(scale, cv2.resize(image, (max(1,round(w*scale)),max(1,round(h*scale))),
                              interpolation=cv2.INTER_AREA)) for scale in (.25,.5,1.)]


def importance_map(rgb, alpha):
    lab = _oklab(rgb)
    result = np.zeros(alpha.shape, np.float32)
    for scale, weight in ((1.,.6),(.5,.3),(.25,.1)):
        h,w = alpha.shape
        size = (max(1,round(w*scale)),max(1,round(h*scale)))
        small = cv2.resize(lab,size,interpolation=cv2.INTER_AREA)
        gx = cv2.Sobel(small, cv2.CV_32F, 1,0,ksize=3)/4
        gy = cv2.Sobel(small, cv2.CV_32F, 0,1,ksize=3)/4
        grad = np.sqrt(np.sum(gx*gx+gy*gy,axis=-1))
        result += weight*cv2.resize(grad,(w,h),interpolation=cv2.INTER_LINEAR)
    result = np.clip(result/.3,0,1)
    boundary = np.zeros(alpha.shape,bool)
    boundary[:,1:] |= alpha[:,1:] != alpha[:,:-1]
    boundary[:,:-1] |= alpha[:,1:] != alpha[:,:-1]
    boundary[1:] |= alpha[1:] != alpha[:-1]
    boundary[:-1] |= alpha[1:] != alpha[:-1]
    result[boundary] = 1
    result[alpha==0] = 0
    return result.astype(np.float32)


def flatten_rgba(rgba, preset='detail'):
    rgb, alpha = rgba[...,:3].copy(), rgba[...,3]
    visible = alpha > 0
    # Hidden RGB cannot influence either clustering or the bilateral kernel.
    if visible.any() and not visible.all():
        indices = distance_transform_edt(~visible,return_distances=False,return_indices=True)
        rgb[~visible] = rgb[tuple(indices[:,~visible])]
    elif not visible.any():
        rgb[:] = 0
    importance = importance_map(rgb,alpha)
    flat = cv2.bilateralFilter(rgb,7,38 if preset=='draft' else 28,5)
    flat = cv2.bilateralFilter(flat,7,32 if preset=='draft' else 24,5)
    # Protect source edge seeds and every alpha transition from smoothing.
    alpha_edge = np.zeros(alpha.shape,np.uint8)
    alpha_edge[:,1:] |= (alpha[:,1:] != alpha[:,:-1])
    alpha_edge[1:] |= (alpha[1:] != alpha[:-1])
    alpha_edge = cv2.dilate(alpha_edge,np.ones((3,3),np.uint8)) > 0
    flat[(importance > .75) | alpha_edge] = rgb[(importance > .75) | alpha_edge]
    flat[~visible] = 0
    return flat, importance
