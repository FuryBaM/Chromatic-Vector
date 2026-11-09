"""Render-based multi-scale residual proposals with clipped, bounded overlays."""
from html import escape
import cv2
import numpy as np
from .palette import quantize, connected_regions
from .contours import extract_edges, trace_rings
from .shared import SharedBoundaryGeometry
from .hierarchy import VectorShape
from .metrics import render_svg, residual, objective


def mask_path(mask, x=0, y=0, method="bounded"):
    edges=extract_edges(mask.astype(np.int32))
    rings=trace_rings(edges)
    if 1 not in rings: return ''
    # Smooth vector patches as well as the base: adding raw staircase details
    # would reintroduce the same defect after a successful base fit.
    from .geometry import safe_shared
    if x or y:
        edges=edges.copy();edges[:,(1,3)]+=x;edges[:,(2,4)]+=y
        rings=trace_rings(edges)
    graph,_=safe_shared(edges,simplify=.8,smooth=.7,max_error=.8,method=method)
    return ' '.join(graph.paths(rings)[1])


def refine_document(document, source, flat, importance, labels, locked, config, base_paths):
    rendered=render_svg(document.svg().encode())
    initial_objective=objective(source,rendered,importance,len(document.shapes))
    log=[]; accepted=0; residual_images=[residual(source,rendered,importance)]
    base_by_region={s.source_region_ids[0]:s for s in document.shapes if s.kind=='base'}
    if config.detail_budget==0 or config.refine_passes==0:
        return rendered,dict(proposals=0,accepted_patches=0,rejected_patches=0,patch_log=[],
                             objective_before=initial_objective,objective_after=initial_objective),residual_images
    # Fine segmentation is a proposal source only; the rendered objective decides.
    colors=min(48,max(config.base_colors+8,config.base_colors*2),source.shape[0]*source.shape[1])
    if colors<2: return rendered,dict(proposals=0,accepted_patches=0,rejected_patches=0,patch_log=[],objective_before=initial_objective,objective_after=initial_objective),residual_images
    ids=quantize(flat,colors,config.seed)
    nbase=int(labels.max())+1
    combos=ids.astype(np.int64)*nbase+labels
    _,dense=np.unique(combos,return_inverse=True)
    fine=connected_regions(dense.reshape(labels.shape).astype(np.int32),source[...,:3])
    slices=__import__('scipy').ndimage.find_objects(fine.labels+1)
    used=set()
    for pass_index in range(config.refine_passes):
        error=residual(source,rendered,importance)
        high=(error>=config.residual_threshold)&(source[...,3]==255)&~locked
        high=cv2.morphologyEx(high.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))>0
        candidates=[]
        for rid,sl in enumerate(slices):
            if sl is None or rid in used or fine.sizes[rid]<5: continue
            local=fine.labels[sl]==rid
            eligible=local & high[sl]
            if not eligible.any(): continue
            score=float((error[sl]*eligible).sum())*(1+min(1,float(fine.sizes[rid])/100))
            candidates.append((-score,rid,sl))
        candidates.sort(key=lambda c:(c[0],c[1]))
        pass_shapes=list(document.shapes); pass_render=rendered.copy();pass_log_start=len(log)
        before_pass=objective(source,rendered,importance,len(document.shapes))
        for _,rid,sl in candidates[:max(20,(config.detail_budget-accepted)*3)]:
            if accepted>=config.detail_budget: break
            used.add(rid)
            y0,y1=sl[0].start,sl[0].stop;x0,x1=sl[1].start,sl[1].stop
            mask=(fine.labels[sl]==rid)&~locked[sl]&(source[sl][...,3]==255)
            ys,xs=np.nonzero(mask)
            if len(xs)<5: continue
            parent_id=int(labels[y0+ys[0],x0+xs[0]])
            parent=base_by_region.get(parent_id)
            if parent is None or parent.locked: continue
            # All proposal pixels belong to one base region by construction.
            color=tuple(map(int,np.rint(source[sl][...,:3][mask].mean(axis=0))))
            d=mask_path(mask,x0,y0,method='bounded' if config.contour_method=='neural' else config.contour_method)
            if not d: continue
            document.clips.setdefault(parent.uid,' '.join(base_paths[parent_id]))
            level=('primary','secondary','detail')[config.max_shape_levels-1]
            layer='highlights' if np.dot(np.array(color)-np.array(parent.fill[:3]),(.2126,.7152,.0722))>10 else 'shadows'
            shape=VectorShape(f'patch-{pass_index}-{rid}',[parent_id],level,layer,(*color,255),
                              'fill-rule="evenodd" d="'+escape(d,quote=True)+'"','path',
                              (x0,y0,x1,y1),len(xs),composition_parent_uid=parent.uid,
                              draw_order=len(document.shapes),clipped_by_uid=parent.uid,
                              score=float(error[sl][mask].mean()),kind='overlay')
            bounds=(max(0,x0-4),max(0,y0-4),min(document.width,x1+4),min(document.height,y1+4))
            bx,by,ex,ey=bounds; roi=np.s_[by:ey,bx:ex]
            before=rendered[roi].copy()
            before_value=float(residual(source[roi],before,importance[roi]).sum())
            document.shapes.append(shape)
            after=render_svg(document.svg(bounds).encode())
            after_value=float(residual(source[roi],after,importance[roi]).sum())
            gain=(before_value-after_value)/(document.width*document.height)-.000003
            entry=dict(pass_index=pass_index,uid=shape.uid,bbox=list(bounds),source_region_id=rid,
                       objective_delta=-gain,before=before_value,after=after_value,
                       accepted=bool(gain>1e-8),reason='objective_improved' if gain>1e-8 else 'complexity_or_residual')
            if gain>1e-8:
                accepted+=1
                rendered[roi]=after
            else: document.shapes.pop()
            log.append(entry)
        rendered=render_svg(document.svg().encode())
        after_pass=objective(source,rendered,importance,len(document.shapes))
        if after_pass>=before_pass-1e-9:
            document.shapes=pass_shapes;rendered=pass_render
            for entry in log[pass_log_start:]:
                if entry['accepted']:
                    accepted-=1;entry['accepted']=False;entry['reason']='global_objective_rollback'
            break
        residual_images.append(residual(source,rendered,importance))
        if accepted>=config.detail_budget: break
    # Remove clip definitions for rejected proposals.
    active_clips={s.clipped_by_uid for s in document.shapes if s.clipped_by_uid}
    document.clips={k:v for k,v in document.clips.items() if k in active_clips}
    return rendered,dict(proposals=len(log),accepted_patches=accepted,rejected_patches=len(log)-accepted,
                         patch_log=log,objective_before=initial_objective,
                         objective_after=objective(source,rendered,importance,len(document.shapes))),residual_images
