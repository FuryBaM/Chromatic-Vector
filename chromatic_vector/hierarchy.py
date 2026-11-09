"""Explicit editable partition/overlay document; layer labels are heuristic."""
from collections import Counter
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.strtree import STRtree

from . import __version__

LAYERS=('background','base_planes','shadows','highlights','details','accents')

@dataclass
class VectorShape:
    uid: str
    source_region_ids: list[int]
    level: str
    layer: str
    fill: tuple[int,int,int,int]
    geometry: str
    primitive: str
    bbox: tuple[float,float,float,float]
    area: int
    containment_parent_uid: str | None = None
    composition_parent_uid: str | None = None
    draw_order: int = 0
    clipped_by_uid: str | None = None
    score: float = 0.
    kind: str = 'base'
    fit_metrics: dict = field(default_factory=dict)
    locked: bool = False
    holes: int = 0

    def metadata(self):
        return {key:value for key,value in vars(self).items() if key!='geometry'}

@dataclass
class VectorDocument:
    width: int
    height: int
    shapes: list[VectorShape]
    boundaries: list[dict]
    geometry_metrics: dict
    clips: dict[str,str] = field(default_factory=dict)

    def svg(self, bounds=None):
        fmt=lambda x: f'{x:.6f}'.rstrip('0').rstrip('.') or '0'
        vx,vy,vw,vh = (0,0,self.width,self.height) if bounds is None else (bounds[0],bounds[1],bounds[2]-bounds[0],bounds[3]-bounds[1])
        lines=['<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" width="{vw}" height="{vh}" viewBox="{vx} {vy} {vw} {vh}">',
            f'<title>Chromatic Vector v{__version__} — editable geometric layers</title>']
        if self.clips:
            lines.append('<defs>')
            for uid,geometry in self.clips.items():
                lines.append(f'<clipPath id="clip-{uid}" clipPathUnits="userSpaceOnUse"><path fill-rule="evenodd" clip-rule="evenodd" d="{escape(geometry,quote=True)}"/></clipPath>')
            lines.append('</defs>')
        if self.geometry_metrics.get('opaque_backdrop'):
            lines.append(f'<rect data-role="opaque-canvas" x="0" y="0" width="{self.width}" height="{self.height}" fill="{self.geometry_metrics["opaque_backdrop"]}"/>')
        if self.geometry_metrics.get('seam_underpaint'):
            # Opaque-only vector backing. The fills remain canonical; a narrow
            # under-stroke covers antialias seams without disabling antialiasing.
            lines.append('<g data-role="seam-underpaint" stroke-width="0.6" stroke-linejoin="round" stroke-linecap="round" aria-hidden="true">')
            for shape in self.shapes:
                if shape.kind!='base':continue
                if bounds is not None:
                    x0,y0,x1,y1=shape.bbox
                    if x1+2<bounds[0] or x0-2>bounds[2] or y1+2<bounds[1] or y0-2>bounds[3]:continue
                color='#'+''.join(f'{c:02x}' for c in shape.fill[:3])
                lines.append(f'<use href="#{shape.uid}" stroke="{color}"/>')
            lines.append('</g>')
        # Groups are contiguous draw-order runs. Never regroup overlapping shapes.
        previous=None; run=0
        for shape in sorted(self.shapes,key=lambda s:s.draw_order):
            if bounds is not None:
                x0,y0,x1,y1=shape.bbox
                if x1+2<bounds[0] or x0-2>bounds[2] or y1+2<bounds[1] or y0-2>bounds[3]: continue
            group=(shape.kind,shape.layer)
            if group!=previous:
                if previous is not None: lines.append('</g>')
                lines.append(f'<g id="layer-{shape.layer}-{run}" data-layer="{shape.layer}" data-heuristic="true" inkscape:groupmode="layer" inkscape:label="{shape.layer}">')
                previous=group; run+=1
            color='#'+''.join(f'{c:02x}' for c in shape.fill[:3])
            attrs=(('shape-rendering="crispEdges" ' if (self.geometry_metrics.get('alpha_exact') or shape.locked) and shape.kind=='base' else 'shape-rendering="geometricPrecision" ')+f'id="{shape.uid}" data-shape-id="{shape.uid}" data-layer="{shape.layer}" data-level="{shape.level}" '
                   f'data-parent="{shape.containment_parent_uid or "none"}" data-composition-parent="{shape.composition_parent_uid or "none"}" '
                   f'data-draw-order="{shape.draw_order}" data-kind="{shape.kind}" fill="{color}" fill-opacity="{fmt(shape.fill[3]/255)}"')
            if shape.clipped_by_uid: attrs+=f' clip-path="url(#clip-{shape.clipped_by_uid})"'
            lines.append(f'<{shape.primitive} {attrs} {shape.geometry}/>')
        if previous is not None: lines.append('</g>')
        lines.append('</svg>')
        return '\n'.join(lines)

    def write(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(self.svg(),encoding='utf-8')

    def stats(self):
        return dict(shape_count=len(self.shapes),layers=dict(Counter(s.layer for s in self.shapes)),
                    primitives=dict(Counter(s.primitive for s in self.shapes)),
                    shape_levels=dict(Counter(s.level for s in self.shapes)),
                    helpers=dict(opaque_canvas_rect=int(bool(self.geometry_metrics.get('opaque_backdrop'))),
                                 seam_underpaint_uses=sum(s.kind=='base' for s in self.shapes) if self.geometry_metrics.get('seam_underpaint') else 0))


def containment_parents(legacy):
    # A containing region is established by an actual hole, not graph distance.
    holes=[]; owners=[]
    for rid,shape in legacy.shapes.items():
        for ring in shape.holes:
            poly=Polygon(ring)
            if poly.is_valid and poly.area>0:
                holes.append(poly);owners.append(rid)
    if not holes: return {}
    tree=STRtree(holes); result={}
    for rid,shape in legacy.shapes.items():
        if not shape.outer: continue
        poly=Polygon(shape.outer[0])
        if not poly.is_valid: continue
        point=poly.representative_point()
        candidates=[int(j) for j in tree.query(point,predicate='within') if owners[int(j)]!=rid and holes[int(j)].covers(poly)]
        if candidates:
            j=min(candidates,key=lambda k:(holes[k].area,owners[k]))
            result[rid]=owners[j]
    return result


def classify(shape, legacy, max_levels):
    area=shape.pixel_area; total=legacy.width*legacy.height
    if shape.layer=='background': layer='background'; level='primary'
    elif area>=max(40,total*.006): layer='base_planes';level='primary'
    else:
        level='secondary' if area>=max(12,total*.0005) else 'detail'
        ref=legacy.shapes.get(shape.parent_id)
        if ref is None and shape.adjacent:
            ref=max((legacy.shapes[j] for j in shape.adjacent),key=lambda s:s.pixel_area)
        lum=lambda c: float(np.dot(c,(.2126,.7152,.0722)))
        delta=lum(shape.fill_rgb)-lum(ref.fill_rgb) if ref else 0
        if level=='detail': layer='accents' if shape.contrast>75 else 'details'
        else: layer='shadows' if delta < -12 else 'highlights' if delta>12 else 'base_planes'
    level=('primary' if max_levels==1 else 'secondary' if max_levels==2 and level=='detail' else level)
    return layer,level
