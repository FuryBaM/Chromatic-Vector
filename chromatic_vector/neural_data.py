"""Independent synthetic vector/raster pairs for the experimental Bézier learner.

Cubic endpoints are (0,0),(1,0). Patches cover [-.25,1.25] x [-.75,.75].
Controls are normalized; input channels are RGB, noisy region mask, boundary map.
"""
import cv2
import numpy as np

SIZE=64
X=np.linspace(-.25,1.25,SIZE,dtype=np.float32)
Y=np.linspace(-.75,.75,SIZE,dtype=np.float32)


def curve_samples(handles,count=257):
    h=np.asarray(handles,np.float32).reshape(-1,4)
    t=np.linspace(0,1,count,dtype=np.float32)[None,:];u=1-t
    x=3*u*u*t*h[:,0:1]+3*u*t*t*h[:,2:3]+t**3
    y=3*u*u*t*h[:,1:2]+3*u*t*t*h[:,3:4]
    return x,y


def _complex_neighborhood(rng, yy, rgb, region, ca, cb):
    """Put third regions near a tracked edge, including unfit junctions."""
    xx, vertical = np.meshgrid(X, Y)
    side = rng.choice((-1, 1))
    distance = (vertical - yy[None]) * side
    kind = rng.integers(5)
    if kind == 0:
        # Aperture and a nearby color island on one side of the edge.
        cx, width = rng.uniform(.12, .88), rng.uniform(.12, .30)
        height = rng.uniform(.10, .24)
        ellipse = ((xx-cx)/width)**2 + ((distance-height-.04)/height)**2
        third = ellipse < 1
        third &= ellipse > rng.uniform(.08, .42)
    elif kind == 1:
        # Narrow ribbon bounded by a second, distinct edge.
        width = rng.uniform(.045, .20) + .04*np.sin(2*np.pi*xx + rng.uniform(0, 2*np.pi))
        third = distance > width
    elif kind == 2:
        # A neighboring concavity, close to the central edge without crossing it.
        top = rng.uniform(.04, .12) + .7*(xx-rng.uniform(.1, .9))**2
        third = (distance > top) & (distance < top + rng.uniform(.10, .26))
    elif kind == 3:
        # A third color interrupts the tracked pair at a T junction or crossing.
        tip = rng.uniform(.2, .8)
        front = rng.uniform(-.13, -.04) + rng.uniform(.25, .9)*np.abs(xx-tip)
        third = distance > front
    else:
        # A junction near an endpoint still leaves the middle of the span fit-able.
        tip = rng.choice((0., 1.)) + rng.uniform(-.025, .025)
        front = rng.uniform(-.06, -.02) + rng.uniform(.65, 1.1)*np.abs(xx-tip)
        third = distance > front

    if rng.random() < .65:
        # Actual crop windows often include unrelated colors on both sides.
        opposite = rng.uniform(.09, .42) + .06*np.sin(2*np.pi*xx + rng.uniform(0, 2*np.pi))
        third |= distance < -opposite

    # Match the runtime's unknown-region value and leave the boundary channel
    # describing only the chain under fit, even when another edge is visible.
    color = rng.uniform(.02, .90, 3) if rng.random() < .6 else (ca+cb)/2
    coverage = cv2.GaussianBlur(third.astype(np.float32), (3, 3), .65)[..., None]
    rgb = rgb*(1-coverage) + color*coverage
    region = np.where(third, .5, region).astype(np.float32)
    return rgb, region, kind == 3


def make_dataset(count,seed,*,complex_shapes=False):
    rng=np.random.default_rng(seed)
    patches=np.empty((count,5,SIZE,SIZE),np.float32)
    handles=np.empty((count,4),np.float32)
    masks=np.empty((count,SIZE,SIZE),np.float32)
    labels=np.empty((count,3),np.float32)
    contexts=np.empty((count,8),np.float32)
    for i in range(count):
        h=np.array([rng.uniform(.15,.45),rng.uniform(-.65,.65),rng.uniform(.55,.85),rng.uniform(-.65,.65)],np.float32)
        kind=rng.choice(4,p=[.55,.15,.15,.15]) # curve, corner, artificial seam, true small object
        x,y=curve_samples(h)
        yy=np.interp(X,x[0],y[0]).astype(np.float32)
        if kind==1:
            peak=rng.uniform(-.5,.5);center=rng.uniform(.3,.7)
            yy=np.interp(X,[0,center,1],[0,peak,0]).astype(np.float32)
        # Analytic soft coverage approximates supersampled rasterization.
        truth=1/(1+np.exp(np.clip(-(Y[:,None]-yy[None,:])/.012,-50,50)))
        ca=rng.uniform(.02,.88,3).astype(np.float32)
        cb=rng.uniform(.02,.88,3).astype(np.float32)
        if kind==2:cb=np.clip(ca+rng.normal(0,.008,3),0,1).astype(np.float32)
        elif np.linalg.norm(ca-cb)<.18:cb=1-ca
        rgb=truth[...,None]*ca+(1-truth[...,None])*cb
        # Gradients, sensor noise and unrelated narrow accents prevent a clean-mask oracle.
        rgb+=rng.normal(0,rng.uniform(.002,.025),rgb.shape)
        rgb+=np.linspace(-1,1,SIZE)[None,:,None]*rng.uniform(-.04,.04)
        sensor=int(rng.integers(16,97))
        noisy=cv2.resize(cv2.resize(truth,(sensor,sensor),interpolation=cv2.INTER_AREA),(SIZE,SIZE),interpolation=cv2.INTER_NEAREST)
        noisy=(noisy>.5).astype(np.float32)
        if rng.random()<.3:
            jitter=rng.random((SIZE,SIZE))<.003
            noisy[jitter]=1-noisy[jitter]
        if kind==3 or (kind==2 and rng.random()<.7):
            # Both true small objects and artificial partitions are represented.
            xx,yy=np.meshgrid(np.arange(SIZE),np.arange(SIZE))
            cx,cy=rng.uniform(22,42,2);rx,ry=rng.uniform(2,10,2)
            box=bool(rng.random()<.5)
            noisy=(((abs(xx-cx)<rx)&(abs(yy-cy)<ry)) if box else (((xx-cx)/rx)**2+((yy-cy)/ry)**2<1)).astype(np.float32)
            if kind==3:
                rgb=noisy[...,None]*ca+(1-noisy[...,None])*cb+rng.normal(0,.012,rgb.shape)
        else:box=False
        edge=cv2.morphologyEx(noisy,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))
        edge=cv2.GaussianBlur(edge,(3,3),.65)
        region=noisy if rng.random()<.5 else 1-noisy
        junction = False
        if complex_shapes and kind == 0:
            rgb, region, junction = _complex_neighborhood(rng, yy, rgb, region, ca, cb)
        patches[i]=np.concatenate((np.clip(rgb,0,1).transpose(2,0,1),region[None],edge[None]),axis=0)
        handles[i]=h;masks[i]=truth;labels[i]=[kind==1 or (kind==3 and box) or junction,kind==2,kind==0 and not junction]
        # Source/context statistics; last two scalars encode relative source sampling.
        contexts[i]=[*np.clip(rgb,0,1).mean(axis=(0,1)),float(np.clip(rgb,0,1).std()),
                     float(np.linalg.norm(ca-cb)),float(region.mean()),sensor/96.,1.]
    return patches,contexts,handles,masks,labels
