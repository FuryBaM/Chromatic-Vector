# Chromatic Vector — neural prototype (0.7.0.dev1)

**Обученная локальная CNN и результаты:** [README_NEURAL.md](README_NEURAL.md).
Исходный v0.6.1 и необязательные методы Unipolar сохранены.

## v0.6.1

**Optional source-verified Unipolar methods:** see [README_UNIPOLAR.md](README_UNIPOLAR.md)
for `--contour-method unipolar-chaikin` / `unipolar-adaptive` and direct 400% SVG comparisons.

Standalone CPU raster-to-editable-SVG vectorizer, Python with optional C++17 acceleration.
The new `illustrative` mode builds coarse visual planes, protects source boundaries,
assigns geometric levels and refines selected details using the rendered SVG residual.
Legacy `balanced` / `precise` behavior is preserved.

v0.6.1 fixes the pixel-staircase appearance: closed contours now have cyclic Bézier
tangents, contour guards use dense uncapped samples and segment distances, and
refinement patches use the same bounded vector fitting. Opaque output uses normal
antialiasing with an editable vector backing instead of `crispEdges`.
See [400% SVG render comparisons](artifacts/v0611/comparisons/index.html).

```bash
python -m pip install -e '.[test]'
# Install CMake + Ninja if needed: python -m pip install cmake ninja
cmake -S Native/ChromaticContours -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
python -m pytest -q
```

CairoSVG needs the system Cairo library (e.g. `libcairo2` on Debian). C++ is optional:
`--backend python` works without a compiled library; `--backend native` requires it.
Native ABI remains 2; no existing exports changed.

```bash
# Existing v0.5 invocation
python -m chromatic_vector references/head_shoulder_crop.png precise.svg \
  --quality precise --target-shapes 2000 --merge-mse-budget 45 \
  --shape-report precise.json

# New mode; preserve full input size and reproduce benchmark geometry settings
python -m chromatic_vector references/head_shoulder_crop.png head.svg \
  --mode illustrative --preset detail --max-side 0 --simplify 1.1 --smooth .9 \
  --max-error 1.1 --target-shapes 250 --shape-report head.json --debug-dir debug/head

python -m chromatic_vector references/torso_crop.png torso.svg \
  --mode illustrative --preset draft --max-side 0 --target-shapes 140 \
  --shape-report torso.json --debug-dir debug/torso
```

Explicit new flags override the selected preset:

| Flag | draft | detail | Valid range |
|---|---:|---:|---|
| `--base-colors` | 10 | 16 | 2..256 |
| `--target-shapes` | 140 | 250 | >=0, 0=unlimited |
| `--detail-budget` | 35 | 100 | >=0; maximum accepted overlays |
| `--boundary-preservation` | .82 | .90 | 0..1 |
| `--residual-threshold` | .15 | .08 | 0..1 |
| `--max-shape-levels` | 3 | 3 | 1..3 |
| `--primitive-fit` | auto | auto | auto/strict/off |
| `--refine-passes` | 1 | 2 | 0..5 |

`--edge-protection` keeps its v0.5 penalty semantics (default 2.5) in the legacy
pipeline. Use `--boundary-preservation` for illustrative protection. Old clustering
and merge flags remain available for the legacy mode; illustrative uses its new
palette, merge and refinement controls. It requires shared geometry / shapes export.
Illustrative CLI defaults to full resolution, `--simplify 1.1 --smooth .9
--max-error 1.1`. Legacy quality defaults remain unchanged; explicit geometry flags
override the illustrative defaults. Source RGB is never blurred for metrics.

`target-shapes` is soft and excludes no important boundary merely to meet a count.
Alpha is preserved as exact opacity strata; no raster is embedded. Correction masks
use the existing opaque-paint convention (alpha >=128), lock edited color regions,
and resize with nearest-neighbor when the source is intentionally downscaled.
Layers (`background/base_planes/shadows/highlights/details/accents`) are heuristic,
without semantic object recognition. Groups follow actual draw order.

The Python API is `chromatic_vector.cli.vectorize(..., mode='illustrative',
preset='detail', target_shapes=250, base_colors=16, detail_budget=100, max_side=0,
simplify=1.1, smooth=.9, max_error=1.1)`.
For compatibility its old positional defaults remain unchanged; CLI preset defaults
are resolved before calling the function. Reports contain actual normalized settings.
Timings / peak memory are diagnostic and naturally vary between runs; SVG, shape IDs,
geometry, configuration and quality metrics are deterministic on the same software.

Reproduce all comparisons (several minutes on a CPU):

```bash
python benchmarks/run.py --output artifacts/baseline --modes precise --backends python native
python benchmarks/run.py --output artifacts/v061 --modes draft detail --backends python native
python benchmarks/audit_contours.py
python benchmarks/compare_contours.py
python benchmarks/report_contours.py
python benchmarks/verify_legacy.py
```

Open `artifacts/v061/comparisons/index.html` locally. The directory also contains 100%,
200% and thumbnail PNGs, CSV and JSON. Each case has actual SVG, preview PNG,
shape report and measured memory/runtime. Debug folders include importance, opacity,
base planes, pyramids, residuals, ROI decisions and exact uint32 region maps.
See [v0.6.1 contour report](docs/V061_CONTOUR_FIX_RU.md) and the historical
[v0.6 technical report](docs/V06_TECHNICAL_REPORT_RU.md) for results and limitations,
and `handoff/` for the supplied requirements and original v0.5 archive.

No GPU, Unity, Unipolar-Gambit, commercial vectorization API or downloaded model is used.
