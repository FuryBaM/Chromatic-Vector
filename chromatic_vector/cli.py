"""CLI for the standalone reference vectorizer."""
import argparse
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image
from .palette import segment
from .corrections import apply_color_overlay
from .contours import extract_edges, trace_rings
from .native import extract_native_edges
from .modes import POSTER_GEOMETRY, PRESETS, IllustrativeConfig
from .svg import write_svg
from .shared import SharedBoundaryGeometry
from .shape_graph import build_document, layer_geometry_styles
from .shape_optimizer import merge_similar_shapes
from .agglomerate import agglomerate_to_budget
from .svg_shapes import write_layered_svg
import json


def vectorize(source: str | Path, dest: str | Path, colors=18, min_area=6,
              max_side=1100, simplify=None, smooth=None, transparent_background=False,
              seed=23, debug_dir: str | Path | None = None, backend: str = "auto",
              geometry: str = "shared", simplifier: str = "game",
              refine: int = 0, max_error: float | None = None,
              correction_mask: str | Path | None = None,
              export_mode: str = "shapes", merge_tolerance: float = 10.0,
              merge_max_area: int = 48, adaptive_layers: bool = True,
              shape_report: str | Path | None = None,
              target_shapes: int | None = None, merge_mse_budget: float = 45.0,
              edge_protection: float = 2.5, *, mode: str = "precise", preset: str = "detail",
              base_colors: int | None = None, detail_budget: int | None = None,
              boundary_preservation: float | None = None, residual_threshold: float | None = None,
              max_shape_levels: int | None = None, primitive_fit: str | None = None,
              refine_passes: int | None = None, contour_method: str = "bounded",
              neural_model: str | Path | None = None, neural_optimize_steps: int = 5, neural_merge: bool = False):
    if (neural_model is not None or neural_merge) and (mode!='illustrative' or contour_method!='neural'):
        raise ValueError('Neural options require --mode illustrative --contour-method neural')
    if mode not in ('precise', 'illustrative'):
        raise ValueError('--mode must be precise or illustrative')
    poster = mode == 'illustrative' and preset == 'poster'
    geometry_defaults = POSTER_GEOMETRY if poster else dict(simplify=.8, smooth=.7, max_error=0.)
    simplify = geometry_defaults['simplify'] if simplify is None else simplify
    smooth = geometry_defaults['smooth'] if smooth is None else smooth
    max_error = geometry_defaults['max_error'] if max_error is None else max_error
    if target_shapes is None and not poster:
        target_shapes = 0
    if mode == 'illustrative':
        from .pipeline import vectorize_illustrative
        config = IllustrativeConfig.resolve(preset, base_colors=base_colors,
            target_shapes=target_shapes, detail_budget=detail_budget,
            boundary_preservation=boundary_preservation, residual_threshold=residual_threshold,
            max_shape_levels=max_shape_levels, primitive_fit=primitive_fit,
            refine_passes=refine_passes, seed=seed, contour_method=contour_method, neural_model=str(neural_model) if neural_model else None,
            neural_optimize_steps=neural_optimize_steps,neural_merge=neural_merge)
        if geometry != 'shared' or export_mode != 'shapes':
            raise ValueError('Illustrative mode requires --geometry shared and --export-mode shapes')
        return vectorize_illustrative(source, dest, config=config, backend=backend,
            max_side=max_side, correction_mask=correction_mask,
            transparent_background=transparent_background, shape_report=shape_report,
            debug_dir=debug_dir, simplify=simplify, smooth=smooth,
            max_error=max_error if max_error else .8)
    if contour_method!='bounded':
        raise ValueError('--contour-method requires --mode illustrative')
    started = time.monotonic()
    image = Image.open(source).convert("RGBA")
    w, h = image.size
    if max_side and max(w, h) > max_side:
        scale = max_side / max(w, h)
        image = image.resize((max(1, round(w*scale)), max(1, round(h*scale))), Image.Resampling.LANCZOS)
    rgba = np.array(image)
    rgb = rgba[..., :3]
    # Edge-preserving noise reduction before color clustering.
    rgb = cv2.bilateralFilter(rgb, 5, 22, 8)
    seg = segment(rgb, colors=colors, min_area=min_area, seed=seed, refine=refine)
    correction_stats = {"corrected_pixels": 0, "correction_colors": 0}
    if correction_mask is not None:
        seg, correction_stats = apply_color_overlay(seg, rgb, correction_mask)
        # The manually painted swatches become the reference colors for the
        # region-union stages. Using the original RGB here would silently
        # recolor user corrections back toward the unedited image.
        painted = np.asarray(Image.open(correction_mask).convert('RGBA'))
        active = painted[..., 3] >= 128
        if np.any(active):
            rgb = rgb.copy()
            rgb[active] = painted[..., :3][active]
    seg, merge_stats = merge_similar_shapes(seg, rgb, merge_tolerance, merge_max_area)
    budget_stats = None
    if target_shapes:
        seg, budget_stats = agglomerate_to_budget(
            seg, rgb, target_shapes, max_added_mse=merge_mse_budget,
            edge_protection=edge_protection)
    edges = extract_native_edges(seg.labels) if backend != "python" else None
    if backend == "native" and edges is None:
        raise RuntimeError("Native backend requested but no compiled library was found")
    using_native = edges is not None
    if edges is None:
        edges = extract_edges(seg.labels)
    rings = trace_rings(edges)
    skip = set()
    if transparent_background:
        corners = [int(seg.labels[0, 0]), int(seg.labels[0, -1]),
                   int(seg.labels[-1, 0]), int(seg.labels[-1, -1])]
        # Only drop a corner region when it actually touches >= 3 corners.
        winner = max(set(corners), key=corners.count)
        if corners.count(winner) >= 3:
            skip.add(winner)
    # Transparent pixels should ideally remain transparent, even with differing RGB.
    # MVP assumes an opaque source; alpha mask support is a next milestone.
    h, w = seg.labels.shape
    if geometry not in ("shared", "legacy"):
        raise ValueError("geometry must be shared or legacy")
    if export_mode not in ("shapes", "regions"):
        raise ValueError("export_mode must be shapes or regions")
    document = build_document(seg.labels, seg.region_colors, rings)
    if transparent_background and export_mode == 'shapes' and document.background_id is not None:
        # One "background color" can consist of multiple physically disconnected
        # regions. Treat all identified background shapes as transparent.
        skip.update(document.layers['background'])
    styles = layer_geometry_styles(document, simplify, smooth) if adaptive_layers and export_mode == 'shapes' else None
    shared = (SharedBoundaryGeometry(edges, simplify, smooth, simplifier,
                                     max_error=max_error, region_styles=styles)
              if geometry == "shared" else None)
    shared_paths = shared.paths(rings) if shared else None
    if export_mode == "shapes":
        if shared_paths is None:
            from .svg import _ring_path, _simplify
            shared_paths = {rid: [_ring_path(_simplify(r, simplify), smooth) for r in loops]
                            for rid, loops in rings.items()}
        primitive_counts = write_layered_svg(dest, document, shared_paths, skip)
    else:
        write_svg(dest, w, h, rings, seg.region_colors, skip, simplify, smooth,
                  shared_paths=shared_paths)
        primitive_counts = {"path": len(rings)-len(skip)}
    if shape_report is not None:
        report_path = Path(shape_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            'width': w, 'height': h, 'background_id': document.background_id,
            'target_shapes': target_shapes if target_shapes else None,
            'budget_merge': vars(budget_stats) if budget_stats else None,
            'layers': document.layer_stats(), 'primitives': primitive_counts,
            'shapes': [
                dict(id=sh.region_id, color=list(sh.fill_rgb), area=sh.pixel_area,
                     bbox=list(sh.bbox), layer=sh.layer, primitive=sh.primitive,
                     parent_id=sh.parent_id, children=sh.children,
                     depth=sh.nesting_depth, holes=len(sh.holes),
                     outer_loops=len(sh.outer), contrast=round(sh.contrast, 2))
                for sh in document.shapes.values() if sh.region_id not in skip
            ]}
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    if debug_dir:
        folder = Path(debug_dir)
        folder.mkdir(parents=True, exist_ok=True)
        Image.fromarray(seg.region_colors[seg.labels]).save(folder / "regions.png")
        # Use npy to keep region IDs exact (unlike a PNG palette encoding).
        np.save(folder / "region_ids.npy", seg.labels)
        reconstructed = seg.region_colors[seg.labels]
        error = np.mean(np.abs(reconstructed.astype(np.float32) - rgb.astype(np.float32)), axis=-1)
        # A continuous tonal heatmap, not an indexed region visualization.
        Image.fromarray(np.clip(error * 4, 0, 255).astype(np.uint8)).save(folder / "color_error.png")
    return {
        "width": w, "height": h, "colors": colors,
        "regions": len(seg.sizes), "edges": len(edges), "backend": "native" if using_native else "python",
        "rings": sum(map(len, rings.values())),
        "geometry": geometry,
        "export_mode": export_mode,
        "merged_regions": merge_stats.joined,
        "budget_merge": vars(budget_stats) if budget_stats else None,
        "shape_count": len(document.shapes) - len(skip),
        "merge_color_mae_delta": round(merge_stats.color_mae_delta, 4),
        "layers": document.layer_stats(),
        "primitives": primitive_counts,
        "simplifier": simplifier if shared else "legacy",
        "shared_chains": len(shared.chains) if shared else 0,
        "cubic_segments": shared.curve_count if shared else 0,
        "geometry_corrections": shared.geometry_corrections if shared else 0,
        "geometry_fallbacks": shared.geometry_fallbacks if shared else 0,
        "rgb_mae": round(float(np.mean(np.abs(seg.region_colors[seg.labels].astype(np.float32)-rgb.astype(np.float32)))), 3),
        "refine_passes": refine,
        **correction_stats,
        "seconds": round(time.monotonic()-started, 2),
    }


def main():
    p = argparse.ArgumentParser(description="Convert raster color regions into editable SVG paths")
    p.add_argument("input", type=Path, help="PNG/JPG/WebP input")
    p.add_argument("output", type=Path, help="SVG output")
    p.add_argument('--mode', choices=('precise','illustrative'), default='precise')
    p.add_argument('--preset', choices=tuple(PRESETS), default='detail')
    p.add_argument('--base-colors', type=int)
    p.add_argument('--detail-budget', type=int)
    p.add_argument('--boundary-preservation', type=float, help='Illustrative hard boundary preservation 0..1; legacy --edge-protection is unchanged')
    p.add_argument('--residual-threshold', type=float)
    p.add_argument('--max-shape-levels', type=int)
    p.add_argument('--primitive-fit', choices=('auto','strict','off'))
    p.add_argument('--refine-passes', type=int)
    p.add_argument('--contour-method',choices=('bounded','unipolar-adaptive','unipolar-chaikin','neural'),default='bounded')
    p.add_argument('--neural-model',type=Path)
    p.add_argument('--neural-optimize-steps',type=int,default=5)
    p.add_argument('--neural-merge',action='store_true')
    p.add_argument("--quality", choices=("balanced", "precise"), default="balanced",
                   help="Preset; explicit flags override its choices")
    p.add_argument("--colors", type=int, default=None, help="Palette size (2..256)")
    p.add_argument("--min-area", type=int, default=None, help="Merge tiny color regions (pixels)")
    p.add_argument("--max-side", type=int, default=None, help="Maximum processed image side; 0 disables resize")
    p.add_argument("--simplify", type=float, default=None, help="Shared-boundary simplification in input pixels")
    p.add_argument("--smooth", type=float, default=None, help="Shared Bezier smoothing 0..1")
    p.add_argument("--transparent-background", action="store_true", help="Drop inferred background shapes (structural layer)")
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--debug-dir", type=Path)
    p.add_argument("--backend", choices=("auto", "python", "native"), default="auto")
    p.add_argument("--simplifier", choices=("game", "rdp"), default="game",
                   help="Game-inspired adaptive chain filter or RDP only")
    p.add_argument("--geometry", choices=("shared", "legacy"), default="shared",
                   help="Topologically shared curves (default), or legacy independent rings")
    p.add_argument("--refine", type=int, default=None, help="Edge-aware correction passes (0..5)")
    p.add_argument("--max-error", type=float, default=None,
                   help="Maximum approximate symmetric geometry deviation in pixels; 0 disables guard")
    p.add_argument("--export-mode", choices=("shapes", "regions"), default="shapes",
                   help="Structured SVG layers + primitives (default), or v0.3-style paths")
    p.add_argument("--merge-tolerance", type=float, default=10.0,
                   help="Maximum RGB Euclidean color difference for adjacent shape fusion; 0 disables")
    p.add_argument("--merge-max-area", type=int, default=48,
                   help="At least one shape must have <= this many pixels to merge")
    p.add_argument("--target-shapes", type=int, default=None,
                   help="Desired maximum shapes; 0 disables global consolidation (v0.4 behavior)")
    p.add_argument("--merge-mse-budget", type=float, default=45.0,
                   help="Maximum added RGB squared error per pixel/channel due to shape consolidation")
    p.add_argument("--edge-protection", type=float, default=2.5,
                   help="Penalty for fusing strong source-image edges; 0 disables soft protection")
    p.add_argument("--flat-layer-geometry", action="store_true",
                   help="Disable per-layer shared-curve simplification strength")
    p.add_argument("--shape-report", type=Path,
                   help="Optional editable-shape metadata JSON sidecar")
    p.add_argument("--correction-mask", type=Path,
                   help="Optional same-size transparent RGBA PNG: opaque painted pixels replace local colors")
    args = p.parse_args()
    if (args.neural_model is not None or args.neural_merge) and (args.mode!='illustrative' or args.contour_method!='neural'):
        p.error('Neural options require --mode illustrative --contour-method neural')
    if args.mode!='illustrative' and args.contour_method!='bounded':
        p.error('--contour-method requires --mode illustrative')
    if args.target_shapes is None:
        if args.mode == 'illustrative':
            args.target_shapes = PRESETS[args.preset]['target_shapes']
        else:
            args.target_shapes = 2000
    if args.mode == 'illustrative':
        try:
            IllustrativeConfig.resolve(args.preset, **{k:getattr(args,k) for k in
                ('base_colors','detail_budget','boundary_preservation','residual_threshold',
                 'max_shape_levels','primitive_fit','refine_passes','contour_method','neural_optimize_steps','neural_merge')})
        except ValueError as exc:
            p.error(str(exc))
    presets = {
        "balanced": dict(colors=18, min_area=6, max_side=1100, simplify=0.8,
                         smooth=0.7, refine=0, max_error=0.0),
        "precise": dict(colors=24, min_area=10, max_side=0, simplify=0.6,
                        smooth=0.55, refine=2, max_error=0.8),
    }
    defaults=presets[args.quality].copy()
    if args.mode=='illustrative':
        defaults.update(max_side=0,simplify=1.1,smooth=.9,max_error=1.1)
        if args.preset == 'poster':
            defaults.update(POSTER_GEOMETRY)
    for key, default in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, default)
    if not args.input.exists():
        p.error(f"Input does not exist: {args.input}")
    if args.simplify < 0 or not 0 <= args.smooth <= 1 or args.max_error < 0 or not 0 <= args.refine <= 5 or args.merge_tolerance < 0 or args.merge_max_area < 0 or args.target_shapes < 0 or args.merge_mse_budget < 0 or args.edge_protection < 0:
        p.error("--simplify >=0, --smooth 0..1, --max-error >=0 and --refine 0..5 required")
    if args.mode == 'illustrative' and (args.geometry != 'shared' or args.export_mode != 'shapes'):
        p.error('Illustrative mode requires --geometry shared and --export-mode shapes')
    if args.colors < 2 or args.colors > 256 or args.min_area < 0 or args.max_side < 0:
        p.error('--colors must be 2..256; --min-area and --max-side must be >=0')
    stats = vectorize(args.input, args.output, args.colors, args.min_area,
                      args.max_side, args.simplify, args.smooth,
                      args.transparent_background, args.seed, args.debug_dir, args.backend,
                      args.geometry, args.simplifier, args.refine, args.max_error, args.correction_mask,
                      args.export_mode, args.merge_tolerance, args.merge_max_area,
                      not args.flat_layer_geometry, args.shape_report,
                      args.target_shapes, args.merge_mse_budget, args.edge_protection,
                      mode=args.mode, preset=args.preset, base_colors=args.base_colors,
                      detail_budget=args.detail_budget, boundary_preservation=args.boundary_preservation,
                      residual_threshold=args.residual_threshold, max_shape_levels=args.max_shape_levels,
                      primitive_fit=args.primitive_fit, refine_passes=args.refine_passes, contour_method=args.contour_method,neural_model=args.neural_model,
                      neural_optimize_steps=args.neural_optimize_steps,neural_merge=args.neural_merge)
    print("Chromatic Vector:", stats)
    print("SVG:", args.output.resolve())


if __name__ == "__main__":
    main()
