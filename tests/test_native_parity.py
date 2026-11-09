import numpy as np
import pytest
from chromatic_vector.native import extract_native_edges
from chromatic_vector.contours import extract_edges, trace_rings, signed_area


def as_set(edges):
    return set(map(tuple, edges.tolist()))


def test_native_equals_python():
    rng = np.random.default_rng(4)
    for size in ((1,1), (2,2), (15,11), (31,23)):
        arr = rng.integers(0, 8, size=size, dtype=np.int32)
        cpp = extract_native_edges(arr)
        if cpp is None:
            pytest.skip("Optional native library not built")
        python = extract_edges(arr)
        assert len(cpp) == len(python)
        assert as_set(cpp) == as_set(python)
        loops = trace_rings(cpp)
        for rid in np.unique(arr):
            assert abs(sum(signed_area(x) for x in loops[int(rid)]) - np.count_nonzero(arr == rid)) < 1e-4
