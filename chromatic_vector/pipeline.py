"""Optional illustrative orchestration. The v0.5 precise pipeline is untouched."""
from collections import Counter
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
from .base_segmentation import propose, merge_planes
from .contours import extract_edges, trace_rings, signed_area
from .native import extract_native_edges
from .shape_graph import build_document
from .geometry import safe_shared
from .shape_fitter import ellipse_fit, ellipse_attributes, ellipse_path, linear_primitive
from .hierarchy import VectorDocument, VectorShape, classify, containment_parents
from .metrics import render_svg, quality_metrics, residual
from .refinement import refine_document


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
        scale=max_side/max(image.size)
        image=image.resize(tuple(max(1,round(v*scale)) for v in image.size),Image.Resampling.LANCZOS)
    rgba=np.array(image);rgba.setflags(write=False)
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
    flat,importance=flatten_rgba(reference,config.preset)
    flat[locked]=reference[...,:3][locked]
    stamp('flatten')
    seg=propose(flat,reference[...,:3],alpha,config,locked,reference[...,:3])
    seg,merge_stats,rag=merge_planes(seg,reference[...,:3],importance,alpha,locked,config)
    stamp('segmentation_rag')
    if max_error==0:simplify=0.;smooth=0.
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
        styles={rid:(0.,0.) for rid in legacy.shapes if has_alpha or region_lock[rid] or
            (legacy.shapes[rid].primitive=='rectangle-candidate' and (config.contour_method=='bounded' or
             min(legacy.shapes[rid].bbox[2]-legacy.shapes[rid].bbox[0],legacy.shapes[rid].bbox[3]-legacy.shapes[rid].bbox[1])>=8))}
        context=NeuralContext(reference[...,:3],current.labels,current.region_colors,neural_model,config.neural_optimize_steps) if neural_model else None
        shared,metrics=safe_shared(edges,simplify,smooth,max_error,styles,method=config.contour_method,
                                   neural_context=context,poster=config.preset=='poster')
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
        shapes.append(VectorShape(f'shape-{rid}',[rid],level,layer,(*shape.fill_rgb,int(region_alpha[rid])),geometry,
                                  primitive,shape.bbox,shape.pixel_area,
                                  containment_parent_uid=f'shape-{parent}' if parent is not None and parent not in skip else None,
                                  draw_order=len(shapes),score=shape.contrast/255,fit_metrics=ellipse_fits.get(rid,{}),
                                  locked=bool(region_lock[rid]),holes=len(shape.holes)))
    geometry_metrics['alpha_exact']=has_alpha
    geometry_metrics['seam_underpaint']=not has_alpha and not transparent_background
    if geometry_metrics['seam_underpaint'] and shapes:
        dominant=max(shapes,key=lambda s:s.area)
        geometry_metrics['opaque_backdrop']='#'+''.join(f'{v:02x}' for v in dominant.fill[:3])
    document=VectorDocument(alpha.shape[1],alpha.shape[0],shapes,boundaries,geometry_metrics)
    stamp('geometry_hierarchy_fit')
    base_render=render_svg(document.svg().encode())
    # Explicit background removal changes the comparison alpha target consistently.
    expected=reference.copy()
    for rid in skip: expected[...,3][seg.labels==rid]=0
    rendered,refine_stats,residuals=refine_document(document,expected,flat,importance,seg.labels,locked,config,paths)
    stamp('render_refinement')
    document.write(dest)
    stats=document.stats()
    if config.target_shapes and stats['shape_count']>config.target_shapes:
        warnings.append(f'Soft target {config.target_shapes} exceeded: {stats["shape_count"]} shapes; protected boundaries and distortion budget take priority.')
    if neural_context and not neural_context.counters['emitted_segments']:
        warnings.append('No learned curve survived the guards; classical geometry was retained.')
    before=quality_metrics(expected,base_render);after=quality_metrics(expected,rendered)
    if after['weighted_edge_loss']>.15:
        warnings.append('Measured source-gradient edge loss exceeds 15%; review thin details and prefer detail or precise where needed.')
    stamp('export_metrics')
    report=dict(shape_report_schema_version=2,version=__version__,mode='illustrative',preset=config.preset,
                config={**config.to_dict(), 'max_side':max_side, 'simplify':simplify, 'smooth':smooth, 'max_error':max_error},source=str(source),source_dimensions=list(original_size),
                width=document.width,height=document.height,backend=actual_backend,native_backend_version=2 if actual_backend=='native' else None,
                **stats,**correction_stats,stages_timing=timings,seconds=time.monotonic()-started,
                peak_memory_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                svg_bytes=Path(dest).stat().st_size,cubic_segments=Path(dest).read_text().count('C '),merge=merge_stats,boundary_protected_count=sum(b['protected'] for b in boundaries),
                residual_proposals=refine_stats['proposals'],accepted_patches=refine_stats['accepted_patches'],rejected_patches=refine_stats['rejected_patches'],
                refinement=refine_stats,metrics_before=before,metrics_after=after,geometry=geometry_metrics,
                fallback_count=geometry_metrics['intersection_fallbacks']+geometry_metrics['guard_fallbacks'],
                warnings=warnings,heuristic_layers=True,neural=neural_context.report() if neural_context else None,neural_merge=neural_merge_stats,shapes=[s.metadata() for s in document.shapes],boundaries=boundaries)
    if shape_report:
        path=Path(shape_report);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    if debug_dir:
        folder=Path(debug_dir);folder.mkdir(parents=True,exist_ok=True)
        Image.fromarray(rgba).save(folder/'source.png')
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
