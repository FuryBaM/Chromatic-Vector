"""Independent learned predictions as real SVGs, without post-fit optimization."""
from pathlib import Path
import sys,json
from io import BytesIO
import cv2,numpy as np
from PIL import Image,ImageDraw
import cairosvg
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from chromatic_vector.neural import BezierModel,_render_controls
from chromatic_vector.neural_data import make_dataset


def svg(h,ca,cb):
    color=lambda c:'#'+''.join(f'{int(v):02x}' for v in np.clip(np.rint(c*255),0,255))
    a,b,c,d=map(float,h)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="-.25 -.75 1.5 1.5" width="64" height="64"><rect x="-.25" y="-.75" width="1.5" height="1.5" fill="{color(cb)}"/><path fill="{color(ca)}" d="M -.25 0 L 0 0 C {a} {b} {c} {d} 1 0 L 1.25 0 L 1.25 .75 L -.25 .75 Z"/></svg>'''


def main():
    out=ROOT/'artifacts/neural/gallery';out.mkdir(parents=True,exist_ok=True)
    model=BezierModel();p,c,h,m,l=make_dataset(64,558871)
    indices=np.flatnonzero(l[:,2])[:8];rows=[]
    sheet=Image.new('RGB',(256*3,288*len(indices)),'#eee');draw=ImageDraw.Draw(sheet)
    for row,i in enumerate(indices):
        pred,corner,merge=model.predict(p[i],c[i])
        pixels=p[i,:3].transpose(1,2,0)
        design=np.stack((m[i].ravel(),1-m[i].ravel()),axis=1)
        colors=np.linalg.lstsq(design,pixels.reshape(-1,3),rcond=None)[0]
        original=np.clip(np.rint(pixels*255),0,255).astype(np.uint8)
        Image.fromarray(original).save(out/f'{row}_input.png')
        images=[Image.fromarray(original).resize((256,256),Image.Resampling.NEAREST)]
        for label,controls in [('truth',h[i]),('learned',pred)]:
            data=svg(controls,colors[0],colors[1]);(out/f'{row}_{label}.svg').write_text(data)
            im=Image.open(BytesIO(cairosvg.svg2png(bytestring=data.encode(),scale=4))).convert('RGB');images.append(im)
        for col,(image,title) in enumerate(zip(images,['Raster input 400%','Vector truth 400%','CNN prediction 400%'])):
            sheet.paste(image,(col*256,row*288+28));draw.text((col*256+6,row*288+9),title,fill='black')
        rows.append(dict(index=int(i),truth=h[i].tolist(),predicted=pred.tolist(),corner_probability=corner,merge_probability=merge,
                         mask_mae=float(np.abs(_render_controls(pred)-m[i]).mean()),optimized=False))
    sheet.save(out/'gallery.png')
    (out/'gallery.json').write_text(json.dumps(dict(seed=558871,model_sha256=model.sha256,examples=rows),indent=2))

if __name__=='__main__':main()
