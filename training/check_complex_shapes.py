"""Compare exported SVG geometry on independent, hand-authored vector fixtures."""
import argparse
import json
from io import BytesIO
from pathlib import Path
import sys

import cairosvg
import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chromatic_vector.cli import vectorize
from chromatic_vector.metrics import render_svg


BACKGROUND = '#151922'
GOLD = '#dfb36d'
BLUE = '#428ab2'
RED = '#ba5267'
FIXTURES = {
    'nested_apertures': f'''
        <path fill="{GOLD}" fill-rule="evenodd" d="M 19 56 C 19 28 53 12 97 15 C 135 17 155 44 150 69 C 143 100 114 114 72 110 C 35 106 17 86 19 56 Z
            M 36 56 C 36 43 51 34 69 37 C 87 39 89 65 71 73 C 53 81 36 71 36 56 Z
            M 96 56 C 101 44 122 42 130 55 C 138 68 124 83 110 82 C 100 81 92 69 96 56 Z"/>
        <ellipse fill="{BLUE}" cx="59" cy="55" rx="8" ry="7"/>
        <path fill="{RED}" d="M 104 66 Q 111 49 124 58 Q 127 72 113 76 Z"/>
    ''',
    'thin_straps': f'''
        <path fill="{GOLD}" d="M 11 31 C 38 3 66 12 96 37 C 119 57 139 72 164 60 L 162 64 C 136 77 115 60 93 41 C 65 17 40 8 13 35 Z"/>
        <path fill="{BLUE}" d="M 10 45 C 38 16 66 23 94 48 C 120 73 142 85 167 70 L 166 74 C 142 90 117 77 92 52 C 65 28 40 21 12 50 Z"/>
        <path fill="{RED}" d="M 13 94 C 47 72 67 67 86 78 C 108 91 127 107 163 95 L 164 100 C 127 113 106 95 84 83 C 65 72 47 78 15 100 Z"/>
    ''',
    't_junctions': f'''
        <path fill="{GOLD}" d="M 12 21 L 154 20 L 159 59 C 131 50 108 51 83 65 C 63 79 42 79 16 67 Z"/>
        <path fill="{BLUE}" d="M 16 67 C 42 79 63 79 83 65 Q 73 93 80 110 L 24 109 Z"/>
        <path fill="{RED}" d="M 83 65 C 108 51 131 50 159 59 L 152 103 L 80 110 Q 73 93 83 65 Z"/>
    ''',
    'sharp_notches': f'''
        <path fill="{GOLD}" fill-rule="evenodd" d="M 11 75 L 53 25 L 65 37 L 94 10 L 151 43 L 132 58 L 160 79 L 120 108 L 101 94 L 84 112 L 43 91 L 41 104 Z
            M 62 59 L 75 67 L 65 84 L 53 74 Z M 104 39 L 118 49 L 102 63 L 95 50 Z"/>
        <path fill="{BLUE}" d="M 118 111 L 164 80 L 164 92 L 139 112 Z"/>
        <path fill="{RED}" d="M 33 37 L 47 19 L 51 23 L 36 41 Z"/>
    ''',
}


def rgb(hex_color):
    return [int(hex_color[i:i+2], 16) for i in (1, 3, 5)]


def topology(image):
    """Count robust components and holes per source fill, ignoring specks under 4 px."""
    colors = np.asarray([rgb(color) for color in (BACKGROUND, GOLD, BLUE, RED)])
    delta = image[..., None, :3].astype(float) - colors
    distance = (delta**2).sum(axis=-1)
    ids = distance.argmin(axis=-1)
    # Antialias blends between two source fills must not count as a third object.
    ids[distance.min(axis=-1) > 42**2] = 0
    signatures = []
    for index in range(1, len(colors)):
        mask = (ids == index).astype(np.uint8)
        _, _, components, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
        _, _, gaps, _ = cv2.connectedComponentsWithStats(1-mask, connectivity=8)
        x, y, w, h, area = gaps[1:].T
        enclosed = (x > 0) & (y > 0) & (x+w < mask.shape[1]) & (y+h < mask.shape[0])
        signatures.append([int((components[1:, 4] >= 4).sum()), int((enclosed & (area >= 4)).sum())])
    return signatures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    modes = [('bounded', {}), ('original', {'contour_method': 'neural'}),
             ('candidate', {'contour_method': 'neural', 'neural_model': args.model})]
    canvas = Image.new('RGB', (4*352, len(FIXTURES)*260), '#202428')
    draw = ImageDraw.Draw(canvas)
    summary = {}
    for row, (name, drawing) in enumerate(FIXTURES.items()):
        vector = f'<svg xmlns="http://www.w3.org/2000/svg" width="176" height="120"><rect width="176" height="120" fill="{BACKGROUND}"/>{drawing}</svg>'
        truth_svg = args.output / f'{name}-truth.svg'
        truth_svg.write_text(vector, encoding='utf-8')
        source = args.output / f'{name}-source.png'
        high = Image.open(BytesIO(cairosvg.svg2png(bytestring=vector.encode(), scale=4))).convert('RGB')
        high.resize((176, 120), Image.Resampling.BOX).save(source)
        truth = topology(np.asarray(Image.open(source)))
        summary[name] = {'truth': truth}
        draw.text((5, row*260+4), f'{name} - vector truth', fill='white')
        canvas.paste(high.resize((352, 240)), (0, row*260+20))
        for col, (method, options) in enumerate(modes, 1):
            svg = args.output / f'{name}-{method}.svg'
            # Compare contour fitters on the source lattice the learned model was trained for.
            stats = vectorize(source, svg, mode='illustrative', preset='poster', base_colors=4,
                              target_shapes=0, detail_budget=0, max_side=0, supersample=1, **options)
            rendered = render_svg(svg)
            signature = topology(rendered)
            geometry = stats['geometry']
            summary[name][method] = dict(signature=signature, matches_truth=signature == truth,
                                         alpha_exact=bool(np.all(rendered[..., 3] == 255)),
                                         lattice_fallbacks=geometry['lattice_fallbacks'],
                                         unresolved_intersections=geometry['unhandled_topology_violations'],
                                         emitted_segments=(stats['neural'] or {}).get('counters', {}).get('emitted_segments', 0),
                                         **stats['metrics_after'])
            image = Image.open(BytesIO(cairosvg.svg2png(url=str(svg), scale=2))).convert('RGB')
            draw.text((col*352+5, row*260+4), method, fill='white')
            canvas.paste(image, (col*352, row*260+20))
        print(name, [(key, summary[name][key]['matches_truth']) for key, _ in modes], flush=True)
    canvas.save(args.output / 'comparison.png')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
