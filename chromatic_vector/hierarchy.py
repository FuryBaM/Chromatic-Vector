"""Explicit editable partition/overlay document; layer labels are heuristic."""
from collections import Counter
from dataclasses import dataclass, field, replace
import heapq
from html import escape
from pathlib import Path
import re
import numpy as np
from shapely.geometry import Polygon, Point, box
from shapely.strtree import STRtree

from . import __version__

LAYERS=('background','base_planes','shadows','highlights','details','accents')
PATH_PREFIX = 'fill-rule="evenodd" d="'
PATH_DELIMITERS = re.compile(r' *([MLQCAZ]) *')


def compact_path(data):
    return PATH_DELIMITERS.sub(r'\1', data).replace(' -', '-')

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
    display_size: tuple[int,int] | None = None
    compact: bool = False

    def svg(self, bounds=None, *, contours=False, display=False):
        fmt=lambda x: f'{x:.6f}'.rstrip('0').rstrip('.') or '0'
        vx,vy,vw,vh = (0,0,self.width,self.height) if bounds is None else (bounds[0],bounds[1],bounds[2]-bounds[0],bounds[3]-bounds[1])
        # A supersampled document keeps working coordinates and is displayed at the source size.
        width,height = self.display_size if display and bounds is None and self.display_size else (vw,vh)
        root_rendering = ' shape-rendering="geometricPrecision"' if self.compact else ''
        lines=['<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" width="{width}" height="{height}" viewBox="{vx} {vy} {vw} {vh}"{root_rendering}>',
            f'<title>Chromatic Vector v{__version__} — editable geometric layers</title>']
        if self.clips:
            lines.append('<defs>')
            for uid,geometry in self.clips.items():
                if self.compact:
                    geometry = compact_path(geometry)
                lines.append(f'<clipPath id="clip-{uid}" clipPathUnits="userSpaceOnUse"><path fill-rule="evenodd" clip-rule="evenodd" d="{escape(geometry,quote=True)}"/></clipPath>')
            lines.append('</defs>')
        if self.geometry_metrics.get('opaque_backdrop'):
            backdrop = f'<rect data-role="opaque-canvas" x="0" y="0" width="{self.width}" height="{self.height}" fill="{self.geometry_metrics["opaque_backdrop"]}"/>'
            if self.compact:
                backdrop = f'<g id="layer-canvas" inkscape:groupmode="layer" inkscape:label="background">{backdrop}</g>'
            lines.append(backdrop)
        if self.geometry_metrics.get('seam_underpaint') and not contours:
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
            if contours:
                color = '#' + ''.join(f'{round(c * .45):02x}' for c in shape.fill[:3])
            crisp = (self.geometry_metrics.get('alpha_exact') or shape.locked) and shape.kind == 'base'
            rendering = 'shape-rendering="crispEdges" ' if crisp else '' if self.compact else 'shape-rendering="geometricPrecision" '
            opacity = '' if self.compact and shape.fill[3] == 255 else f' fill-opacity="{fmt(shape.fill[3]/255)}"'
            attrs=(rendering+f'id="{shape.uid}" data-shape-id="{shape.uid}" data-layer="{shape.layer}" data-level="{shape.level}" '
                   f'data-parent="{shape.containment_parent_uid or "none"}" data-composition-parent="{shape.composition_parent_uid or "none"}" '
                   f'data-draw-order="{shape.draw_order}" data-kind="{shape.kind}" fill="{color}"{opacity}')
            if shape.clipped_by_uid: attrs+=f' clip-path="url(#clip-{shape.clipped_by_uid})"'
            if contours:
                width = .65 if shape.level == 'primary' else .42
                opacity = .42 if shape.kind == 'overlay' else .78
                attrs += (f' stroke="#59dedb" stroke-width="{width}" stroke-opacity="{opacity}"'
                          ' stroke-linejoin="round"')
            geometry = shape.geometry
            if self.compact and geometry.startswith(PATH_PREFIX):
                geometry = PATH_PREFIX + compact_path(geometry[len(PATH_PREFIX):-1]) + '"'
            lines.append(f'<{shape.primitive} {attrs} {geometry}/>')
        if previous is not None: lines.append('</g>')
        lines.append('</svg>')
        return '\n'.join(lines)

    def write(self,path,*,contours=False):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(self.svg(contours=contours,display=True),encoding='utf-8')

    def stats(self):
        return dict(shape_count=len(self.shapes),layers=dict(Counter(s.layer for s in self.shapes)),
                    primitives=dict(Counter(s.primitive for s in self.shapes)),
                    shape_levels=dict(Counter(s.level for s in self.shapes)),
                    helpers=dict(opaque_canvas_rect=int(bool(self.geometry_metrics.get('opaque_backdrop'))),
                                 seam_underpaint_uses=sum(s.kind=='base' for s in self.shapes) if self.geometry_metrics.get('seam_underpaint') else 0))


def pack_equal_fills(sequence, dependencies, keys, shapes):
    groups = []
    positions = [0] * len(shapes)
    compatible = {}
    for index in sequence:
        key = keys[index]
        target = compatible.get(key, -1)
        if target <= max((positions[other] for other in dependencies[index]), default=-1):
            target = len(groups)
            groups.append([])
            if key is not None:
                compatible[key] = target
        groups[target].append(shapes[index])
        positions[index] = target
    return groups


def layer_and_compact(document, padding):
    """Schedule independent shapes by layer and combine disjoint equal fills."""
    shapes = sorted(document.shapes, key=lambda shape: shape.draw_order)
    if not shapes:
        return replace(document, compact=True)
    protected = {uid for shape in shapes for uid in
                 (shape.containment_parent_uid, shape.composition_parent_uid, shape.clipped_by_uid) if uid}
    envelopes = [box(x0-padding, y0-padding, x1+padding, y1+padding)
                 for x0, y0, x1, y1 in (shape.bbox for shape in shapes)]
    earlier, later = STRtree(envelopes).query(envelopes, predicate='intersects')
    followers = [[] for _ in shapes]
    predecessors = [[] for _ in shapes]
    for first, second in zip(earlier, later):
        if first < second:
            followers[first].append(second)
            predecessors[second].append(first)
    waiting = [len(items) for items in predecessors]

    order = {layer: index for index, layer in enumerate(LAYERS)}
    levels = {'primary': 0, 'secondary': 1, 'detail': 2}
    pending = []
    keys = []
    for shape in shapes:
        can_combine = (shape.kind == 'base' and shape.fill[3] == 255 and not shape.locked
                       and shape.uid not in protected and not shape.clipped_by_uid and not shape.fit_metrics
                       and shape.geometry.startswith(PATH_PREFIX))
        paint_stage = shape.layer if shape.layer == 'background' or shape.level == 'primary' else 'details'
        keys.append((paint_stage, shape.fill, shape.containment_parent_uid) if can_combine else None)

    def release(index):
        shape = shapes[index]
        heapq.heappush(pending, (shape.kind != 'base', shape.layer != 'background',
                                 levels[shape.level], order[shape.layer], index))

    for index, count in enumerate(waiting):
        if count == 0:
            release(index)
    scheduled = []
    while pending:
        *_, index = heapq.heappop(pending)
        scheduled.append(index)
        for following in followers[index]:
            waiting[following] -= 1
            if waiting[following] == 0:
                release(following)

    # Try moving independent paths earlier and later in the paint order.
    forward = pack_equal_fills(scheduled, predecessors, keys, shapes)
    backward = pack_equal_fills(reversed(scheduled), followers, keys, shapes)
    groups = forward if len(forward) <= len(backward) else reversed(backward)

    packed = []
    for parts in groups:
        shape = parts[0]
        if len(parts) > 1:
            x0, y0, x1, y1 = zip(*(part.bbox for part in parts))
            data = ' '.join(part.geometry[len(PATH_PREFIX):-1] for part in parts)
            shape = replace(max(parts, key=lambda part: part.area), geometry=PATH_PREFIX + data + '"',
                            bbox=(min(x0), min(y0), max(x1), max(y1)),
                            source_region_ids=[rid for part in parts for rid in part.source_region_ids],
                            area=sum(part.area for part in parts), holes=sum(part.holes for part in parts))
        packed.append(replace(shape, draw_order=len(packed)))
    return replace(document, shapes=packed, compact=True)


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
