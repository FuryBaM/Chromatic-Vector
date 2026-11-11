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


# Distance, in source pixels, that a shock front may travel: enough to close a blur ramp
# into a step without letting neighbouring shapes eat each other.
SHOCK_TRAVEL = 1.


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
        iterations = int(round(SHOCK_TRAVEL*scale))
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


# Ink lines. INK_GAIN: strongest deepening of a dark stroke. INK_NARROW: how much thinner it
# may become. INK_RADIUS: half-width, in source pixels, of the widest stroke treated.
# INK_CONTRAST: share of the surrounding tone a trough must lose to count as a drawn line.
# INK_LENGTH: shortest run, in source pixels, that is a line rather than a speck.
# INK_REACH: distance, in source pixels, over which a stroke lends its strength to its own
# weaker stretches. INK_FLOOR: tone of full ink.
INK_GAIN = 3.
INK_NARROW = 1.75
INK_RADIUS = 1.5
INK_CONTRAST = (.2, .4)
INK_LENGTH = 8.
INK_REACH = 6.
INK_FLOOR = 10.


def ink_lines(rgb, scale):
    """Redraw blurred dark strokes as thin, evenly dark lines.

    Blur lifts a thin dark line toward the tones beside it and spreads it out, unevenly along
    its length, so it quantizes into a pale band that breaks up and lets the shapes it
    separated run together. Every trough narrower than the structuring element gets its
    ink back: the middle is deepened and the flanks are returned to the tone beside the
    stroke. The strength of a stroke is carried along its own ridge, so a faint stretch
    between two clear ones is drawn like them instead of fading out. A stroke narrows only as
    far as it can still deepen, so gaps that are already black keep their width. Short dips
    are texture, not lines, and stay as they are, as do wide dark shapes and plain edges,
    which are no troughs at all.
    """
    size = 2*int(round(INK_RADIUS*scale)) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    image = rgb.astype(np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    filled = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    level = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    depth = level - gray
    # The deepest point within reach sets one factor for the whole cross-section.
    deepest = np.maximum(cv2.dilate(depth, kernel), 1e-6)
    share = np.clip(depth/deepest, 0, 1)
    contrast = deepest/np.maximum(level, 1.)
    trust = (np.clip((contrast - INK_CONTRAST[0])/(INK_CONTRAST[1] - INK_CONTRAST[0]), 0, 1)
             * np.clip((deepest - 6.)/6., 0, 1))
    # The axis of a stroke: darker than both points a stroke-radius away, in some direction.
    # Unlike the share of the local maximum, this does not break where a stroke turns faint,
    # so strength can travel through stretches too weak to be inked on their own evidence.
    reach = size//2
    axis = np.zeros(gray.shape, bool)
    for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
        pair = np.zeros((size, size), np.uint8)
        pair[reach + dy*reach, reach + dx*reach] = pair[reach - dy*reach, reach - dx*reach] = 1
        axis |= cv2.erode(gray, pair) - gray >= np.maximum(3., .4*depth)
    ridge = axis & (depth >= 6.) & (depth >= .5*INK_CONTRAST[0]*level)
    if INK_LENGTH > 0:
        _, components, stats, _ = cv2.connectedComponentsWithStats(ridge.astype(np.uint8), connectivity=8)
        long = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]) >= INK_LENGTH*scale
        long[0] = False
        ridge = long[components]
    strength = np.where(ridge & (trust > 0), deepest, 0).astype(np.float32)
    confidence = np.where(ridge, trust, 0).astype(np.float32)
    step = np.ones((3, 3), np.uint8)
    for _ in range(int(round(INK_REACH*scale))):
        # Geodesic growth: values travel along the ridge and never jump to another stroke.
        strength = np.where(ridge, cv2.dilate(strength, step), 0)
        confidence = np.where(ridge, cv2.dilate(confidence, step), 0)
    strength, confidence = cv2.dilate(strength, kernel), cv2.dilate(confidence, kernel)
    gain = np.clip(INK_GAIN*strength/deepest, 1., np.maximum((level - INK_FLOOR)/deepest, 1.))
    # For a Gaussian profile the power n**2-1 narrows it n times.
    power = np.minimum(INK_NARROW**2, gain**2) - 1
    image = filled - (filled - image)*(1 + confidence*(gain*share**power - 1))[..., None]
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


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
    if INK_GAIN > 1:
        enlarged = ink_lines(enlarged, scale)
    return lift_blacks(coherence_shock(enlarged, scale))
