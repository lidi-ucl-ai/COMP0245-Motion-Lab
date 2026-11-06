import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from jacobian_model import JacobianNet, RawJacobianNet

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--raw-angles-from', type=Path, metavar='LJC_CHECKPOINT',
                   help='Refit the raw-angle ablation with navigation and control from the full LJC checkpoint')
    a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(a.seed);np.random.seed(a.seed)
    assert torch.cuda.is_available();device='cuda';d=np.load('data/jacobian_supervision.npz')
    q=torch.from_numpy(d['train_q']).to(device);j=torch.from_numpy(d['train_j']).to(device)
    vq=torch.from_numpy(d['validation_q']).to(device);vj=torch.from_numpy(d['validation_j']).to(device)
    network = RawJacobianNet if a.raw_angles_from else JacobianNet
    net=network().to(device);opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-5)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs,eta_min=1e-5)
    if a.raw_angles_from:
        full = torch.load(a.raw_angles_from, map_location='cpu', weights_only=False)
        nav, control = full['base_memory'], full['control']
        out = Path(f'runs/jac_rawangle_s{a.seed}')
    else:
        nav = torch.load(f'checkpoints/navigation_s{a.seed}/best.pt', map_location='cpu', weights_only=False)['base_memory']
        control = json.loads(Path('configs/control.json').read_text())
        out = Path(f'checkpoints/jacobian_s{a.seed}')
    out.mkdir(parents=True, exist_ok=True)
    start=time.perf_counter();history=[];best=float('inf');weight=torch.tensor([1.,1.,2.],device=device).reshape(1,3,1)/ (4/3)
    for epoch in range(a.epochs):
        net.train();perm=torch.randperm(len(q),device=device);total=0.
        for ids in perm.split(2048):
            pred=net(q[ids]);loss=(((pred-j[ids])**2)*weight).mean()
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step();total+=loss.item()*len(ids)
        net.eval()
        with torch.no_grad():err=net(vq)-vj;val=float((err.square()*weight).mean());maxerr=float(err.abs().max())
        row=dict(epoch=epoch+1,train_mse=total/len(q),val_mse=val,val_max_abs=maxerr,elapsed_s=time.perf_counter()-start);history.append(row)
        if val<best:
            best=val
            torch.save(dict(family='learned_reversible_jacobian',config={'width':256},state_dict={k:v.cpu() for k,v in net.state_dict().items()},
                base_memory=nav,control=control,seed=a.seed,
                best_epoch=epoch+1,val_mse=val,data_sha256=hashlib.sha256(Path('data/jacobian_supervision.npz').read_bytes()).hexdigest()),out/'best.pt')
        sched.step()
        if epoch%10==0 or epoch==a.epochs-1:print(json.dumps(row),flush=True)
    torch.cuda.synchronize()
    summary=dict(seed=a.seed,device=torch.cuda.get_device_name(),torch=torch.__version__,epochs=a.epochs,batch_size=2048,
        optimizer='AdamW lr1e-3 wd1e-5 cosine to1e-5 clip5',best_val_rmse_m_per_rad=best**.5,fit_seconds=time.perf_counter()-start,
        parameter_count=sum(p.numel() for p in net.parameters()),history=history)
    (out/'training.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='history'}),flush=True)
if __name__=='__main__':main()
