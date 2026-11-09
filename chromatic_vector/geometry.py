"""Audit canonical curves before serialization and recover unsafe chains."""
import numpy as np
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree
from .shared import SharedBoundaryGeometry
from .curve_fitting import fit_poster_curve, fit_shared_curve, sample_curves, turning_cost


def safe_shared(edges, simplify=.65, smooth=.4, max_error=.8, styles=None, method="bounded", neural_context=None,
                *, poster=False):
    from .unipolar import METHODS, fit_unipolar
    if method not in (*METHODS,'neural'):raise ValueError('Unknown contour method: '+method)
    fitter=fit_shared_curve if method in ('bounded','neural') else lambda raw,eps,smooth,err:fit_unipolar(raw,eps,smooth,err,method)
    if poster and method in ('bounded', 'neural'):
        fitter=fit_poster_curve
    # RDP is float-stable across the two edge scanners; canonical anchor identical.
    graph=SharedBoundaryGeometry(edges,simplify,smooth,'rdp',max_error=max_error,region_styles=styles,
                                 chain_fitter=fitter)
    neural_records={}
    bounded_curves={}
    if method=='neural':
        if neural_context is None:raise ValueError('Neural contours require source RGB, region labels and trained model context')
        from .neural import fit_neural_chain
        neural_context.score_merge_pairs(graph.chains,styles)
        for chain_index,chain in enumerate(graph.chains):
            exact=bool(styles and any(styles.get(r)==(0.,0.) for r in chain.paired_regions))
            if exact:continue
            start_record=len(neural_context.records)
            candidate=fit_neural_chain(np.asarray(chain.vertices,float),chain.paired_regions,neural_context,simplify,smooth,max_error)
            end_record = len(neural_context.records)
            neural_records[chain_index]=(start_record,end_record)
            if poster:
                accepted = [record for record in neural_context.records[start_record:end_record] if record['accepted']]
                if not accepted:
                    continue
                # Local improvements can still turn a smooth ring or ribbon into a polygon.
                if turning_cost(candidate) > 1.15*turning_cost(chain.cubics) + .04:
                    for record in accepted:
                        record['final_reason'] = 'chain_smoothness'
                    neural_context.counters['whole_chain_rejections'] += 1
                    continue
            bounded_curves[chain_index]=chain.cubics
            chain.cubics=candidate
    fallback=set(); reduced_fit=set(); lattice_fallback=set(); violations=0
    recover_bounded = method == 'bounded' and max_error > 0 and (simplify > 0 or smooth > 0)
    audit_passes = 5 if method == 'neural' else 3 + int(recover_bounded)
    for _ in range(audit_passes):
        lines=[]
        for chain in graph.chains:
            points=sample_curves(chain.cubics,spacing=.25)
            lines.append(LineString(points))
        tree=STRtree(lines); bad=set()
        for i,line in enumerate(lines):
            if not line.is_simple: bad.add(i)
            for raw_j in tree.query(line,predicate='intersects'):
                j=int(raw_j)
                if j<=i: continue
                intersection=line.intersection(lines[j])
                common=set((graph.chains[i].vertices[0],graph.chains[i].vertices[-1])) & set((graph.chains[j].vertices[0],graph.chains[j].vertices[-1]))
                if intersection.geom_type=='Point': points=[intersection]
                elif intersection.geom_type=='MultiPoint': points=list(intersection.geoms)
                else: bad.update((i,j));continue
                if any(not any(p.distance(Point(v))<1e-7 for v in common) for p in points): bad.update((i,j))
        if not bad: break
        violations+=len(bad)
        for i in bad:
            chain=graph.chains[i]
            # Audit the original bounded fit before reverting to the pixel grid.
            if i in bounded_curves:
                chain.cubics=bounded_curves.pop(i)
            elif (recover_bounded or i in neural_records) and i not in reduced_fit:
                eps, tension = simplify, smooth
                for region in chain.paired_regions:
                    if styles and region in styles:
                        local_eps, local_smooth = styles[region]
                        eps, tension = min(eps, local_eps), min(tension, local_smooth)
                raw=np.asarray(chain.vertices,float)
                retry = fitter if poster and method in ('bounded', 'neural') else fit_shared_curve
                chain.cubics=retry(raw,eps*.5,tension*.5,max_error)[0]
                reduced_fit.add(i)
            elif i not in lattice_fallback:
                chain.cubics=fit_shared_curve(np.asarray(chain.vertices,float),0,0,0)[0]
                lattice_fallback.add(i)
        fallback.update(bad)
    else:
        # Whole-partition fallback has the original traced topology by construction.
        for i,chain in enumerate(graph.chains):
            chain.cubics=fit_shared_curve(np.asarray(chain.vertices,float),0,0,0)[0]
            fallback.add(i)
            lattice_fallback.add(i)
    graph.curve_count=sum(len(c.cubics) for c in graph.chains)
    metrics=dict(contour_method=method,canonical_boundaries=len(graph.chains),intersection_fallbacks=len(fallback),
                 lattice_fallbacks=len(lattice_fallback),detected_unsafe_chains=violations,unhandled_topology_violations=0,
                 guard_corrections=graph.geometry_corrections,guard_fallbacks=graph.geometry_fallbacks)
    if method=='neural':
        emitted=0
        for idx,(a,b) in neural_records.items():
            for record in neural_context.records[a:b]:
                record['chain_uid']=f'boundary-{idx}'
                record['emitted']=bool(record['accepted'] and idx not in fallback and 'final_reason' not in record)
                if record['accepted'] and idx in fallback:record.setdefault('final_reason','topology_fallback')
                emitted+=record['emitted']
        neural_context.counters['emitted_segments']=emitted
        metrics['neural']=neural_context.report()
    return graph,metrics
