"""Portable ctypes adapter for the standalone C++ shared-boundary scanner."""
import ctypes as ct
from functools import lru_cache
import os
from pathlib import Path
import numpy as np


class CvPoint(ct.Structure):
    _fields_ = [("x", ct.c_float), ("y", ct.c_float)]


class CvEdge(ct.Structure):
    _fields_ = [
        ("region_id", ct.c_uint32),
        ("x0", ct.c_int32),
        ("y0", ct.c_int32),
        ("x1", ct.c_int32),
        ("y1", ct.c_int32),
    ]


def _candidates():
    env = os.getenv("CHROMATIC_CONTOURS_LIB")
    if env:
        yield Path(env)
    root = Path(__file__).resolve().parents[1]
    for library in ("libchromatic_contours.so", "chromatic_contours.dll", "libchromatic_contours.dylib"):
        yield root / "build" / library
        yield root / "build" / "Release" / library
        yield root / "build" / "Debug" / library


@lru_cache(maxsize=1)
def _load():
    for path in _candidates():
        if path.exists():
            dll = ct.CDLL(str(path))
            dll.cv_abi_version.restype = ct.c_uint32
            if dll.cv_abi_version() != 2:
                raise RuntimeError("Wrong chromatic_contours ABI")
            dll.cv_extract_shared_edges.argtypes = [ct.POINTER(ct.c_uint32), ct.c_uint32, ct.c_uint32,
                                                    ct.POINTER(CvEdge), ct.c_uint64]
            dll.cv_extract_shared_edges.restype = ct.c_uint64
            dll.cv_adaptive_filter_polyline.argtypes = [ct.POINTER(CvPoint), ct.c_uint64, ct.c_uint8,
                                                        ct.c_float, ct.c_float, ct.c_float,
                                                        ct.POINTER(CvPoint), ct.c_uint64]
            dll.cv_adaptive_filter_polyline.restype = ct.c_uint64
            return dll
    return None


def extract_native_edges(labels: np.ndarray):
    """Returns None if no native library is installed, otherwise Nx5 edges."""
    dll = _load()
    if dll is None:
        return None
    if np.any(labels < 0):
        raise ValueError("Region IDs must be non-negative")
    labels = np.ascontiguousarray(labels, dtype=np.uint32)
    h, w = labels.shape
    ptr = labels.ctypes.data_as(ct.POINTER(ct.c_uint32))
    needed = dll.cv_extract_shared_edges(ptr, w, h, None, 0)
    edges = np.empty((needed, 5), dtype=np.int32)
    dest = edges.ctypes.data_as(ct.POINTER(CvEdge))
    written = dll.cv_extract_shared_edges(ptr, w, h, dest, needed)
    if written != needed:
        raise RuntimeError("Native edge count changed between calls")
    return edges


def filter_native_polyline(points: np.ndarray, closed: bool, drop_tolerance: float,
                           smooth_strength: float, sharp_turn_threshold: float = 0.65):
    """Return a ported game-style adaptive polyline, or None if native is absent."""
    dll = _load()
    if dll is None:
        return None
    points = np.ascontiguousarray(points, dtype=np.float32)
    n = len(points)
    ptr = points.ctypes.data_as(ct.POINTER(CvPoint))
    result = np.empty_like(points)
    dest = result.ctypes.data_as(ct.POINTER(CvPoint))
    written = dll.cv_adaptive_filter_polyline(ptr, n, bool(closed),
        drop_tolerance, smooth_strength, sharp_turn_threshold, dest, n)
    if not 0 < written <= n:
        raise RuntimeError("Native adaptive filter returned invalid point count")
    return result[:written].astype(np.float64)
