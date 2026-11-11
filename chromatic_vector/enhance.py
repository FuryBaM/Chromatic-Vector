"""Supersampled poster source: smooth upscaling followed by edge-collapsing shock filtering.

A small or soft raster stores every contour as a ramp several pixels wide. Quantizing that
ramp directly yields staircase bands, and thin seams or highlights vanish between palette
levels. Working on an enlarged copy whose ramps are collapsed to steps at their inflection
lines lets the ordinary partition trace subpixel contours and keep narrow ink lines.
"""
import cv2
import numpy as np

MAX_SUPERSAMPLE = 4
# Long side, in working pixels, that automatic supersampling aims for.
AUTO_WORKING_SIDE = 1400


def auto_supersample(width, height):
    return int(np.clip(AUTO_WORKING_SIDE // max(1, width, height), 1, MAX_SUPERSAMPLE))


def coherence_shock(rgb, scale=1., *, iterations=None, strength=.6):
    """Coherence-enhancing shock filter (Weickert 2003), gated to real edges.

    Each pass moves a pixel toward the local maximum where the smoothed luminance is concave
    across the dominant structure direction, and toward the local minimum where it is convex.
    Ramps therefore turn into steps, ridges into even bright strokes and troughs into even
    dark strokes. Flat areas have no reliable sign and are left alone.
    """
    scale = float(max(1., scale))
    sigma = .5 * scale
    rho = .625 * scale
    # Luminance slope (per working pixel) below which the sign of curvature is noise.
    gate = 4. / scale
    if iterations is None:
        # About one source pixel of travel: enough to close a blur ramp without eating shapes.
        iterations = int(round(scale))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    image = rgb.astype(np.float32)
    for _ in range(iterations):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        smooth = cv2.GaussianBlur(gray, (0, 0), sigma)
        gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3) / 8
        gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3) / 8
        jxx = cv2.GaussianBlur(gx * gx, (0, 0), rho)
        jxy = cv2.GaussianBlur(gx * gy, (0, 0), rho)
        jyy = cv2.GaussianBlur(gy * gy, (0, 0), rho)
        # Dominant eigenvector of the structure tensor: the direction of greatest contrast.
        theta = .5 * np.arctan2(2 * jxy, jxx - jyy)
        c, s = np.cos(theta), np.sin(theta)
        vxx = cv2.Sobel(smooth, cv2.CV_32F, 2, 0, ksize=3)
        vxy = cv2.Sobel(smooth, cv2.CV_32F, 1, 1, ksize=3)
        vyy = cv2.Sobel(smooth, cv2.CV_32F, 0, 2, ksize=3)
        across = c * c * vxx + 2 * c * s * vxy + s * s * vyy
        weight = np.clip((np.sqrt(jxx + jyy) - gate) / gate, 0, 1)[..., None] * strength
        target = np.where((across < 0)[..., None], cv2.dilate(image, kernel), cv2.erode(image, kernel))
        image += (target - image) * weight
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


# sRGB level below which tonal differences are treated as one ink black.
BLACK_TOE = 12.
_TOE_GAIN = 255. / np.hypot(255., BLACK_TOE)


def lift_blacks(rgb):
    """Compress near-black tones so invisible halos cannot split the darkest plane."""
    return np.clip(np.rint(np.hypot(rgb.astype(np.float32), BLACK_TOE) * _TOE_GAIN), 0, 255).astype(np.uint8)


def restore_blacks(color):
    """Invert :func:`lift_blacks` for a fill color."""
    value = np.asarray(color, np.float64) / _TOE_GAIN
    restored = np.sqrt(np.maximum(value * value - BLACK_TOE * BLACK_TOE, 0.))
    return tuple(int(v) for v in np.clip(np.rint(restored), 0, 255))


def supersample(rgb, scale):
    """Enlarge opaque RGB by an integer factor and restore crisp contours."""
    if scale <= 1:
        return rgb.copy()
    enlarged = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    # Cubic lobes must not invent tones outside the local source range.
    kernel = np.ones((3, 3), np.uint8)
    size = (enlarged.shape[1], enlarged.shape[0])
    low = cv2.resize(cv2.erode(rgb, kernel), size, interpolation=cv2.INTER_NEAREST)
    high = cv2.resize(cv2.dilate(rgb, kernel), size, interpolation=cv2.INTER_NEAREST)
    enlarged = np.clip(enlarged, low, high)
    return lift_blacks(coherence_shock(enlarged, scale))
