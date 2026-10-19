"""Train a small CNN-GRU to predict 5s ego-relative trajectories from comma1M.
Requires processed samples.csv from prepare_comma1m.py.
"""
import argparse,csv,json,random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset

class DriveSet(Dataset):
    def __init__(self,rows):self.rows=rows
    def __len__(self):return len(self.rows)
    def __getitem__(self,idx):
        r=self.rows[idx]
        imgs=[]
        for p in json.loads(r['history']):
            with Image.open(p) as img:
                imgs.append(np.asarray(img.convert('RGB'),dtype=np.uint8))
        x=torch.from_numpy(np.stack(imgs).copy()).permute(0,3,1,2).float().div_(255)
        return x,torch.tensor([float(r['speed'])/30],dtype=torch.float32),torch.tensor(json.loads(r['trajectory']),dtype=torch.float32)

class CNNGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv2d(3,24,5,stride=2,padding=2),nn.ReLU(),nn.Conv2d(24,40,3,stride=2,padding=1),nn.ReLU(),nn.Conv2d(40,64,3,stride=2,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((3,6)),nn.Flatten(),nn.Linear(64*3*6,128),nn.ReLU())
        self.gru=nn.GRU(128,128,batch_first=True)
        self.head=nn.Sequential(nn.Linear(129,128),nn.ReLU(),nn.Linear(128,40))
    def forward(self,x,v):
        b,t,c,h,w=x.shape
        z=self.encoder(x.reshape(b*t,c,h,w)).reshape(b,t,-1)
        z,_=self.gru(z)
        return self.head(torch.cat((z[:,-1],v),dim=1)).reshape(b,20,2)

def metrics(pred,true):
    dist=np.linalg.norm(pred-true,axis=-1)
    return float(dist.mean()),float(dist[:,-1].mean())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='data/processed/comma1m_v1/samples.csv')
    ap.add_argument('--epochs',type=int,default=10)
    ap.add_argument('--batch-size',type=int,default=16)
    ap.add_argument('--lr',type=float,default=3e-4)
    ap.add_argument('--output',default='checkpoints/comma1m_cnn_gru_v1')
    args=ap.parse_args()
    random.seed(27); np.random.seed(27); torch.manual_seed(27)
    with open(args.data,newline='') as f:rows=list(csv.DictReader(f))
    segments=sorted(set(r['segment'] for r in rows))
    if len(segments)<3:raise RuntimeError(f'Need 3+ complete segments; found {len(segments)}')
    random.shuffle(segments)
    nval=max(1,round(len(segments)*0.22))
    valid_ids=set(segments[:nval])
    train=[r for r in rows if r['segment'] not in valid_ids]
    valid=[r for r in rows if r['segment'] in valid_ids]
    if not train or not valid:raise RuntimeError('Empty train or validation split')
    print(f'Train {len(train)} examples from {len(set(r["segment"] for r in train))} segments; val {len(valid)} from {len(valid_ids)} segments',flush=True)
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    (out/'split.json').write_text(json.dumps({'train':sorted(set(r['segment'] for r in train)),'validation':sorted(valid_ids)},indent=2))
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print('Device:',device,flush=True)
    tl=DataLoader(DriveSet(train),batch_size=args.batch_size,shuffle=True,num_workers=0)
    vl=DataLoader(DriveSet(valid),batch_size=args.batch_size,shuffle=False,num_workers=0)
    model=CNNGRU().to(device)
    optim=torch.optim.AdamW(model.parameters(),lr=args.lr)
    best=float('inf')
    with (out/'history.csv').open('w',newline='') as hf:
        writer=csv.writer(hf);writer.writerow(['epoch','train_loss','val_loss','ADE_m','FDE_m','constant_velocity_ADE_m','constant_velocity_FDE_m'])
        for epoch in range(1,args.epochs+1):
            model.train();losses=[]
            for imgs,speed,target in tl:
                imgs,speed,target=imgs.to(device),speed.to(device),target.to(device)
                pred=model(imgs,speed)
                loss=nn.functional.smooth_l1_loss(pred/30,target/30)
                optim.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optim.step();losses.append(float(loss.detach().cpu()))
            model.eval();v_losses=[];ps=[];ts=[];vs=[]
            with torch.no_grad():
                for imgs,speed,target in vl:
                    pred=model(imgs.to(device),speed.to(device)).cpu()
                    v_losses.append(float(nn.functional.smooth_l1_loss(pred/30,target/30)))
                    ps.append(pred.numpy());ts.append(target.numpy());vs.append(speed.numpy()*30)
            pred=np.concatenate(ps);true=np.concatenate(ts);speed=np.concatenate(vs)[:,0]
            ade,fde=metrics(pred,true)
            baseline=np.stack([speed[:,None]*np.arange(1,21)[None,:]*.25,np.zeros((len(speed),20))],axis=-1)
            bade,bfde=metrics(baseline,true)
            vlmean=float(np.mean(v_losses))
            writer.writerow([epoch,np.mean(losses),vlmean,ade,fde,bade,bfde]);hf.flush()
            torch.save({'model_state_dict':model.state_dict(),'epoch':epoch,'config':{'type':'cnn_gru_trajectory','history_frames':4,'future_waypoints':20}},out/'last.pt')
            if vlmean<best:
                best=vlmean
                torch.save({'model_state_dict':model.state_dict(),'epoch':epoch,'config':{'type':'cnn_gru_trajectory','history_frames':4,'future_waypoints':20}},out/'best.pt')
            print(f'Epoch {epoch:02d} train={np.mean(losses):.4f} val={vlmean:.4f} ADE={ade:.2f}m FDE={fde:.2f}m | constant-velocity ADE={bade:.2f}m FDE={bfde:.2f}m',flush=True)
    print('Saved:',out)
if __name__=='__main__':main()
