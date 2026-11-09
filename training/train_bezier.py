"""CPU training on synthetic vectors. No photo fixtures enter training/validation.

python training/train_bezier.py --train 14000 --epochs 12
Exports an ONNX model for the existing OpenCV CPU runtime and independent test metrics.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from chromatic_vector.neural_data import make_dataset,SIZE,X,Y


class PatchBezier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv2d(5,12,5,2,2),nn.ReLU(),nn.Conv2d(12,24,3,2,1),nn.ReLU(),
                                   nn.Conv2d(24,32,3,2,1),nn.ReLU(),nn.Flatten())
        self.head=nn.Sequential(nn.Linear(32*8*8+8,128),nn.ReLU(),nn.Linear(128,6))
    def forward(self,patch,context):
        return self.head(torch.cat((self.encoder(patch),context),dim=1))


def decode(z):
    return torch.stack((.15+.30*z[:,0].sigmoid(),.80*z[:,1].tanh(),
                        .55+.30*z[:,2].sigmoid(),.80*z[:,3].tanh()),dim=1)


def raster(handles):
    t=torch.linspace(0,1,129,device=handles.device)[None];u=1-t
    x=3*u*u*t*handles[:,0:1]+3*u*t*t*handles[:,2:3]+t**3
    y=3*u*u*t*handles[:,1:2]+3*u*t*t*handles[:,3:4]
    query=torch.tensor(X,device=handles.device)[None].expand(len(handles),-1).contiguous().clamp(0,1)
    indices=torch.searchsorted(x.detach().contiguous(),query,right=True).clamp(1,128)
    x0=x.gather(1,indices-1);x1=x.gather(1,indices)
    y0=y.gather(1,indices-1);y1=y.gather(1,indices)
    yy=y0+(query-x0)/(x1-x0).clamp_min(1e-6)*(y1-y0)
    grid=torch.tensor(Y,device=handles.device)[None,:,None]
    return torch.sigmoid((grid-yy[:,None,:])/.018)


def tensor_dataset(data):return TensorDataset(*[torch.from_numpy(a) for a in data])


def evaluate(model,data):
    model.eval();totals=[]
    with torch.no_grad():
        for p,c,h,m,l in DataLoader(tensor_dataset(data),batch_size=256):
            z=model(p,c);pred=decode(z);smooth=l[:,2]>.5
            error=(raster(pred)-m).abs().mean(dim=(1,2))
            for i in range(len(p)):
                totals.append([float(error[i]),bool(smooth[i]),float(z[i,4].sigmoid()),float(z[i,5].sigmoid()),float(l[i,0]),float(l[i,1])])
    a=np.asarray(totals)
    return dict(smooth_raster_mae=float(a[a[:,1]>0,0].mean()),
                corner_accuracy=float(((a[:,2]>.5)==(a[:,4]>.5)).mean()),
                merge_accuracy=float(((a[:,3]>.5)==(a[:,5]>.5)).mean()))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train',type=int,default=14000);ap.add_argument('--epochs',type=int,default=12)
    ap.add_argument('--output',type=Path,default=ROOT/'models/bezier_patch_v1');args=ap.parse_args()
    torch.set_num_threads(3);torch.manual_seed(23);np.random.seed(23)
    started=time.monotonic();args.output.mkdir(parents=True,exist_ok=True)
    train=make_dataset(args.train,23);val=make_dataset(1600,1234);test=make_dataset(1600,198761)
    loader=DataLoader(tensor_dataset(train),batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(23))
    model=PatchBezier();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
    history=[];best=float('inf');best_state=None
    for epoch in range(args.epochs):
        model.train();losses=[]
        for p,c,h,m,l in loader:
            z=model(p,c);pred=decode(z);valid=l[:,2]
            render=(raster(pred)-m).abs().mean(dim=(1,2))
            controls=(pred-h).square().mean(dim=1)
            loss=((render+.08*controls)*valid).sum()/valid.sum().clamp_min(1)
            loss+=.12*nn.functional.binary_cross_entropy_with_logits(z[:,4:],l[:,:2])
            # Mild control regularization discourages unnecessarily bent handles.
            loss+=.0002*pred[:,[1,3]].square().mean()
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2);opt.step();losses.append(float(loss.detach()))
        metrics=evaluate(model,val);row=dict(epoch=epoch+1,loss=float(np.mean(losses)),**metrics,seconds=time.monotonic()-started)
        history.append(row);print(json.dumps(row),flush=True)
        if metrics['smooth_raster_mae']<best:
            best=metrics['smooth_raster_mae'];best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state);model.eval()
    torch.save({'state_dict':model.state_dict(),'architecture':'PatchBezier-v1'},args.output/'checkpoint.pt')
    torch.onnx.export(model,(torch.zeros(1,5,64,64),torch.zeros(1,8)),str(args.output/'model.onnx'),
        input_names=['patch','context'],output_names=['prediction'],dynamic_axes={'patch':{0:'batch'},'context':{0:'batch'},'prediction':{0:'batch'}},
        opset_version=17,dynamo=False)
    metrics=evaluate(model,test)
    mask=test[3];valid=test[4][:,2]>.5
    straight=1/(1+np.exp(np.clip(-Y[:,None]/.018,-50,50)))
    metrics['straight_raster_mae']=float(np.abs(mask[valid]-straight).mean())
    # Real exported runtime parity, independent of PyTorch execution.
    net=cv2.dnn.readNetFromONNX(str(args.output/'model.onnx'));p,c=test[:2]
    net.setInput(p[:16],'patch');net.setInput(c[:16],'context');actual=net.forward()
    with torch.no_grad():expected=model(torch.from_numpy(p[:16]),torch.from_numpy(c[:16])).numpy()
    parity=float(np.abs(actual-expected).max());assert parity<1e-4,parity
    meta=dict(schema=1,architecture='PatchBezier-v1',patch_size=64,input_channels=['R','G','B','region_A','raw_boundary'],
              train_count=args.train,validation_count=1600,test_count=1600,train_seed=23,validation_seed=1234,test_seed=198761,
              synthetic_only=True,generator_version=2,photo_fixtures_used_in_training=False,outputs=['p1x','p1y','p2x','p2y','corner_logit','merge_logit'],
              test_metrics=metrics,opencv_pytorch_max_error=parity,history=history,seconds=time.monotonic()-started,
              sha256=hashlib.sha256((args.output/'model.onnx').read_bytes()).hexdigest())
    (args.output/'metadata.json').write_text(json.dumps(meta,indent=2,allow_nan=False));print(json.dumps(meta['test_metrics']),flush=True)

if __name__=='__main__':main()
