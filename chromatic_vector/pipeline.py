"""Optional illustrative orchestration. The v0.5 precise pipeline is untouched."""
from collections import Counter
from dataclasses import replace
from html import escape
import json
from pathlib import Path
import resource
import time
import cv2
import numpy as np
from PIL import Image
from .modes import IllustrativeConfig
from . import __version__
from .flatten import flatten_rgba, pyramid
from .enhance import auto_supersample, restore_blacks, supersample
from .base_segmentation import propose, merge_planes
from .palette import poster_contrast
from .contours import extract_edges, trace_rings, signed_area
from .native import extract_native_edges
from .shape_graph import build_document
from .geometry import safe_shared
from .shape_fitter import ellipse_fit, ellipse_attributes, ellipse_path, linear_primitive
from .hierarchy import VectorDocument, VectorShape, classify, containment_parents, layer_and_compact
from .metrics import render_svg, quality_metrics, residual
from .refinement import refine_document


# OKLab distance below which a seam between two supersampled planes is not worth keeping.
FAINT_SEAM_CONTRAST = .04


def vectorize_illustrative(source,dest,*,config,backend='auto',max_side=0,correction_mask=None,
                           transparent_background=False,shape_report=None,debug_dir=None,
                           simplify=.65,smooth=.4,max_error=.8):
    if backend not in ('auto','python','native'): raise ValueError('--backend must be auto, python or native')
    if max_side<0: raise ValueError('--max-side must be >=0')
    for name,value in (('simplify',simplify),('smooth',smooth),('max_error',max_error)):
        if not np.isfinite(value) or value<0 or (name=='smooth' and value>1):
            raise ValueError(f'Invalid {name}')
    started=time.monotonic();timings={};warnings=[]
    last_stamp=started
    def stamp(name):
        nonlocal last_stamp
        now=time.monotonic();timings[name]=now-last_stamp;last_stamp=now
    image=Image.open(source).convert('RGBA');original_size=image.size
    if max_side and max(image.size)>max_side:
        shrink=max_side/max(image.size)
        image=image.resize(tuple(max(1,round(v*shrink)) for v in image.size),Image.Resampling.LANCZOS)
    source_rgba=np.array(image);source_rgba.setflags(write=False)
    display_size=image.size
    # Small opaque posters are traced on an enlarged, edge-restored copy of the source.
    scale=config.supersample
    opaque=bool(np.all(source_rgba[...,3]==255))
    if scale==0:
        # The learned contour model was trained on the source pixel lattice.
        automatic=config.preset=='poster' and config.contour_method!='neural' and opaque and correction_mask is None
        scale=auto_supersample(*display_size) if automatic else 1
    elif scale>1 and not opaque:
        warnings.append('Supersampling needs an opaque source; the original pixel grid was used.')
        scale=1
    if scale>1:
        rgba=np.dstack((supersample(source_rgba[...,:3],scale),
                        np.full((display_size[1]*scale,display_size[0]*scale),255,np.uint8)))
        image=Image.fromarray(rgba)
    else:rgba=source_rgba
    rgba.setflags(write=False)
    reference=rgba.copy();alpha=rgba[...,3];locked=np.zeros(alpha.shape,bool)
    correction_stats=dict(corrected_pixels=0,correction_colors=0)
    if correction_mask is not None:
        painted=Image.open(correction_mask).convert('RGBA')
        if painted.size==original_size and image.size!=original_size:
            painted=painted.resize(image.size,Image.Resampling.NEAREST)
        if painted.size!=image.size: raise ValueError('Correction mask size must match source or working canvas')
        painted=np.array(painted);locked=(painted[...,3]>=128)&(alpha>0)
        reference[...,:3][locked]=painted[...,:3][locked]
        correction_stats=dict(corrected_pixels=int(locked.sum()),correction_colors=len(np.unique(painted[...,:3][locked],axis=0)))
    flat,importance=flatten_rgba(reference,config.preset,scale)
    flat[locked]=reference[...,:3][locked]
    stamp('flatten')
    seg=propose(flat,reference[...,:3],alpha,config,locked,reference[...,:3])
    seg,merge_stats,rag=merge_planes(seg,reference[...,:3],importance,alpha,locked,config,
                                     min_contrast=FAINT_SEAM_CONTRAST if scale>1 else 0.)
    stamp('segmentation_rag')
    if max_error==0:simplify=0.;smooth=0.
    # Working pixels are finer than source pixels; keep the fit tolerance tied to the source.
    geometry_scale=max(1.,scale/2)
    simplify*=geometry_scale;max_error*=geometry_scale
    neural_model=None
    if config.contour_method=='neural':
        from .neural import BezierModel,NeuralContext
        neural_model=BezierModel(config.neural_model)
    def construct(current):
        edges=extract_native_edges(current.labels) if backend!='python' else None
        if backend=='native' and edges is None:raise RuntimeError('Native backend requested but no compiled library was found')
        actual='native' if edges is not None else 'python'
        if edges is None:edges=extract_edges(current.labels)
        edges=edges[np.lexsort(tuple(edges[:,j] for j in (4,3,2,1,0)))]
        rings=trace_rings(edges);legacy=build_document(current.labels,current.region_colors,rings)
        region_alpha=np.zeros(len(current.sizes),np.uint8);region_alpha[current.labels.ravel()]=alpha.ravel()
        region_lock=np.bincount(current.labels.ravel(),weights=locked.ravel(),minlength=len(current.sizes))>0
        has_alpha=bool(np.any(alpha!=255))
        exact_rectangles = config.contour_method == 'bounded' or (config.preset == 'poster' and config.contour_method == 'neural')
        styles={rid:(0.,0.) for rid in legacy.shapes if has_alpha or region_lock[rid] or
            (legacy.shapes[rid].primitive=='rectangle-candidate' and (exact_rectangles or
             min(legacy.shapes[rid].bbox[2]-legacy.shapes[rid].bbox[0],legacy.shapes[rid].bbox[3]-legacy.shapes[rid].bbox[1])>=8))}
        context=NeuralContext(reference[...,:3],current.labels,current.region_colors,neural_model,config.neural_optimize_steps) if neural_model else None
        shared,metrics=safe_shared(edges,simplify,smooth,max_error,styles,method=config.contour_method,
                                   neural_context=context,poster=config.preset=='poster',
                                   poster_scale=geometry_scale)
        return edges,rings,legacy,region_alpha,region_lock,has_alpha,shared,metrics,context,actual
    geometry_state=construct(seg)
    neural_merge_stats=None
    if config.neural_merge:
        from .neural import conservative_merge,preview_partition
        from .metrics import objective
        context=geometry_state[8]
        candidate,neural_merge_stats=conservative_merge(seg,reference[...,:3],alpha,locked,importance,
            context.merge_hints,config.boundary_preservation)
        if neural_merge_stats['merged']:
            candidate_state=construct(candidate)
            before,count_before=preview_partition(seg,alpha,geometry_state[6],geometry_state[1])
            after,count_after=preview_partition(candidate,alpha,candidate_state[6],candidate_state[1])
            before_loss=objective(reference,before,importance,count_before)
            after_loss=objective(reference,after,importance,count_after)
            neural_merge_stats.update(objective_before=before_loss,objective_after=after_loss,
                                      render_accepted=bool(after_loss<before_loss-1e-8))
            if neural_merge_stats['render_accepted']:
                seg=candidate;geometry_state=candidate_state
            else:
                neural_merge_stats['rolled_back_merges']=neural_merge_stats['merged']
                neural_merge_stats['merged']=0;neural_merge_stats['after']=neural_merge_stats['before']
                neural_merge_stats['proposed_added_mse']=neural_merge_stats['added_mse'];neural_merge_stats['added_mse']=0.
    edges,rings,legacy,region_alpha,region_lock,has_alpha,shared,geometry_metrics,neural_context,actual_backend=geometry_state
    paths=shared.paths(rings)
    skip={rid for rid in legacy.shapes if region_alpha[rid]==0}
    if transparent_background and not has_alpha and legacy.background_id is not None:
        skip.add(legacy.background_id)
        warnings.append('Background removal uses a canvas-connected geometric heuristic.')
    parents=containment_parents(legacy)
    boundaries=[]
    for i,chain in enumerate(shared.chains):
        pair=chain.paired_regions
        protected=(len(pair)<2 or region_alpha[pair[0]]!=region_alpha[pair[-1]] or any(region_lock[r] for r in pair))
        if len(pair)==2:
            a,b=pair
            # Re-evaluate final pair protection after merging; all protected roots survive.
            ys=np.clip(np.array(chain.vertices)[:,1],0,len(alpha)-1)
            xs=np.clip(np.array(chain.vertices)[:,0],0,alpha.shape[1]-1)
            strength=float(importance[ys,xs].mean())
            protected=protected or strength>.4
        else: strength=1.
        boundaries.append(dict(uid=f'boundary-{i}',regions=list(pair),protected=bool(protected),importance=strength,
                               junctions=[] if chain.closed else [list(chain.vertices[0]),list(chain.vertices[-1])],closed=chain.closed,
                               fitted_start=chain.cubics[0][0].tolist(),fitted_end=chain.cubics[-1][-1].tolist(),
                               polyline=chain.vertices,cubics=[[p.tolist() for p in c] for c in chain.cubics]))
    # Ellipse substitutions replace BOTH references to one isolated closed boundary.
    ellipse_fits={}
    if not has_alpha and not locked.any() and config.primitive_fit!='off':
        for rid,shape in legacy.shapes.items():
            if rid in skip or shape.holes or len(shape.outer)!=1 or len(shape.adjacent)!=1: continue
            if rid==legacy.background_id: continue
            neighbor=shape.adjacent[0]
            matching=[c for c in shared.chains if c.closed and c.paired_regions==tuple(sorted((rid,neighbor)))]
            if len(matching)!=1 or len(paths[rid])!=1: continue
            x0,y0,x1,y1=shape.bbox
            localmask=seg.labels[y0:y1,x0:x1]==rid
            localring=shape.outer[0]-np.array((x0,y0))
            fit=ellipse_fit(localmask,localring,config.primitive_fit)
            if fit is None: continue
            # Require one pixel clearance from every unrelated shape.
            roi=seg.labels[max(0,y0-2):min(len(alpha),y1+2),max(0,x0-2):min(alpha.shape[1],x1+2)]
            if np.any((roi!=rid)&(roi!=neighbor)): continue
            fit['cx']+=x0;fit['cy']+=y0
            # Locate the identical physical cycle among the neighbor's hole rings.
            for k,ring in enumerate(rings[neighbor]):
                if signed_area(ring)<0 and set(map(tuple,ring))==set(matching[0].vertices):
                    paths[neighbor][k]=ellipse_path(fit,False)
                    paths[rid]=[ellipse_path(fit,True)]
                    ellipse_fits[rid]=fit
                    break
    if neural_context and ellipse_fits:
        for record in neural_context.records:
            if record.get('emitted') and any(r in ellipse_fits for r in record['regions']):
                record['emitted']=False;record['final_reason']='primitive_substitution'
        neural_context.counters['emitted_segments']=sum(r.get('emitted',False) for r in neural_context.records)
        geometry_metrics['neural']=neural_context.report()
    shapes=[]
    for rid,shape in legacy.shapes.items():
        if rid in skip: continue
        layer,level=classify(shape,legacy,config.max_shape_levels)
        if rid in ellipse_fits: primitive='ellipse';geometry=ellipse_attributes(ellipse_fits[rid])
        else: primitive,geometry=linear_primitive(paths[rid],shape,config.primitive_fit)
        parent=parents.get(rid)
        # Supersampled planes average a black-lifted source; export the true tone.
        fill_rgb=restore_blacks(shape.fill_rgb) if scale>1 and not region_lock[rid] else shape.fill_rgb
        shapes.append(VectorShape(f'shape-{rid}',[rid],level,layer,(*fill_rgb,int(region_alpha[rid])),geometry,
                                  primitive,shape.bbox,shape.pixel_area,
                                  containment_parent_uid=f'shape-{parent}' if parent is not None and parent not in skip else None,
                                  draw_order=len(shapes),score=shape.contrast/255,fit_metrics=ellipse_fits.get(rid,{}),
                                  locked=bool(region_lock[rid]),holes=len(shape.holes)))
    geometry_metrics['alpha_exact']=has_alpha
    geometry_metrics['seam_underpaint']=not has_alpha and not transparent_background
    dominant = max(shapes, key=lambda shape: shape.area) if geometry_metrics['seam_underpaint'] and shapes else None
    if dominant is not None:
        geometry_metrics['opaque_backdrop']='#'+''.join(f'{v:02x}' for v in dominant.fill[:3])
    document=VectorDocument(alpha.shape[1],alpha.shape[0],shapes,boundaries,geometry_metrics,
                            display_size=tuple(display_size) if scale>1 else None)
    stamp('geometry_hierarchy_fit')
    # Explicit background removal changes the comparison alpha target consistently.
    expected=reference.copy()
    for rid in skip: expected[...,3][seg.labels==rid]=0
    if scale>1:
        # The supersampled partition already holds the accents that 1x overlays restore.
        rendered,refine_stats,residuals=refine_document(document,expected,flat,importance,seg.labels,locked,
                                                        replace(config,detail_budget=0),paths)
        base_render=rendered
    else:
        base_render=render_svg(document.svg().encode())
        rendered,refine_stats,residuals=refine_document(document,expected,flat,importance,seg.labels,locked,config,paths)
    stamp('render_refinement')
    # Shock filtering already restores the contrast that blur removed from a supersampled source.
    if config.preset == 'poster' and scale == 1:
        base_shapes = [shape for shape in document.shapes if shape.kind == 'base']
        if len({shape.fill[:3] for shape in base_shapes}) > 1:
            for shape in document.shapes:
                if not shape.locked and shape.fill[3] == 255:
                    shape.fill = (*poster_contrast(shape.fill[:3]), shape.fill[3])
            if dominant is not None:
                geometry_metrics['opaque_backdrop'] = '#'+''.join(f'{v:02x}' for v in dominant.fill[:3])
            rendered = render_svg(document.svg().encode())
            base_render = (render_svg(replace(document, shapes=base_shapes, clips={}).svg().encode())
                           if refine_stats['accepted_patches'] else rendered)
    if config.preset == 'poster' and scale > 1:
        original_svg = document.svg(display=True).encode()
        original_display = render_svg(original_svg)
        packed = layer_and_compact(document, padding=max_error+2*scale)
        for candidate in (packed, replace(document, compact=True)):
            candidate_svg = candidate.svg(display=True).encode()
            if (len(candidate_svg) < len(original_svg)
                    and np.array_equal(render_svg(candidate_svg), original_display)
                    and np.array_equal(render_svg(candidate.svg().encode()), rendered)):
                geometry_metrics['export_compaction'] = dict(shapes_before=len(document.shapes),
                                                              bytes_before=len(original_svg))
                document = candidate
                break
        stamp('export_compaction')
    if config.stacked and geometry_metrics['seam_underpaint'] and not locked.any():
        from .stacking import stack_poster
        holes_before = sum(shape.holes for shape in document.shapes)
        document, saved, rendered = stack_poster(document, seg.labels, scale, max_error, rendered)
        geometry_metrics['stacked_segments_removed'] = saved
        geometry_metrics['stacked_holes_filled'] = holes_before - sum(shape.holes for shape in document.shapes)
        stamp('stacked_export')
    document.write(dest)
    stats=document.stats()
    if config.target_shapes and stats['shape_count']>config.target_shapes:
        warnings.append(f'Soft target {config.target_shapes} exceeded: {stats["shape_count"]} shapes; protected boundaries and distortion budget take priority.')
    if neural_context and not neural_context.counters['emitted_segments']:
        warnings.append('No learned curve survived the guards; classical geometry was retained.')
    if scale>1:
        # Report fidelity against the untouched source on its own pixel grid.
        def displayed(raster):
            return cv2.resize(raster,tuple(display_size),interpolation=cv2.INTER_AREA)
        target=source_rgba.copy();target[...,3]=displayed(expected[...,3])
        before=quality_metrics(target,displayed(base_render));after=quality_metrics(target,displayed(rendered))
    else:
        before=quality_metrics(expected,base_render);after=quality_metrics(expected,rendered)
    if after['weighted_edge_loss']>.15:
        warnings.append('Measured source-gradient edge loss exceeds 15%; review thin details and prefer detail or precise where needed.')
    stamp('export_metrics')
    report=dict(shape_report_schema_version=2,version=__version__,mode='illustrative',preset=config.preset,
                config={**config.to_dict(), 'max_side':max_side, 'simplify':simplify, 'smooth':smooth, 'max_error':max_error},source=str(source),source_dimensions=list(original_size),
                width=document.width,height=document.height,supersample=scale,display_dimensions=list(display_size),backend=actual_backend,native_backend_version=2 if actual_backend=='native' else None,
                **stats,**correction_stats,stages_timing=timings,seconds=time.monotonic()-started,
                peak_memory_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                svg_bytes=Path(dest).stat().st_size,
                cubic_segments=sum(shape.geometry.count('C ') for shape in document.shapes)
                               + sum(path.count('C ') for path in document.clips.values()),
                merge=merge_stats,boundary_protected_count=sum(b['protected'] for b in boundaries),
                residual_proposals=refine_stats['proposals'],accepted_patches=refine_stats['accepted_patches'],rejected_patches=refine_stats['rejected_patches'],
                refinement=refine_stats,metrics_before=before,metrics_after=after,geometry=geometry_metrics,
                fallback_count=geometry_metrics['intersection_fallbacks']+geometry_metrics['guard_fallbacks'],
                warnings=warnings,heuristic_layers=True,neural=neural_context.report() if neural_context else None,neural_merge=neural_merge_stats,shapes=[s.metadata() for s in document.shapes],boundaries=boundaries)
    if shape_report:
        path=Path(shape_report);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    if debug_dir:
        folder=Path(debug_dir);folder.mkdir(parents=True,exist_ok=True)
        Image.fromarray(source_rgba).save(folder/'source.png')
        if scale>1:Image.fromarray(rgba).save(folder/'enhanced.png')
        Image.fromarray(np.dstack((flat,alpha))).save(folder/'flattened.png')
        Image.fromarray(alpha).save(folder/'alpha_mask.png')
        Image.fromarray(np.uint8(np.clip(importance*255,0,255))).save(folder/'edge_importance.png')
        Image.fromarray(seg.region_colors[seg.labels]).save(folder/'segmentation_base.png')
        Image.fromarray(base_render).save(folder/'primary_shapes.png')
        Image.fromarray(rendered).save(folder/'preview.png')
        for i,err in enumerate(residuals): Image.fromarray(np.uint8(np.clip(err*600,0,255))).save(folder/f'residual_{i}.png')
        Image.fromarray(np.uint8(np.clip(residual(expected,rendered,importance)*600,0,255))).save(folder/'residual_final.png')
        overlay=rendered.copy();outline=np.zeros(alpha.shape,bool)
        outline[:,1:]|=seg.labels[:,1:]!=seg.labels[:,:-1];outline[1:]|=seg.labels[1:]!=seg.labels[:-1]
        overlay[outline]=(255,40,100,255);Image.fromarray(overlay).save(folder/'outline_overlay.png')
        for scale,small in pyramid(np.dstack((flat,alpha))): Image.fromarray(small).save(folder/f'pyramid_{scale:g}.png')
        np.save(folder/'region_ids.npy',seg.labels.astype(np.uint32))
        (folder/'initial_rag.json').write_text(json.dumps([dict(regions=list(pair),**st) for pair,st in rag.items()],indent=2))
        (folder/'candidate_rois.json').write_text(json.dumps(refine_stats['patch_log'],indent=2))
        (folder/'metrics.json').write_text(json.dumps({k:v for k,v in report.items() if k not in ('shapes','boundaries')},indent=2,allow_nan=False))
        document.write(folder/'output.svg')
        document.write(folder/'vector_contours.svg',contours=True)
    return {k:v for k,v in report.items() if k not in ('shapes','boundaries','refinement')}
