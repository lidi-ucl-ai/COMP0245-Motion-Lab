from pathlib import Path
import argparse,hashlib,torch
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True);a=p.parse_args()
d=torch.load(a.source,map_location='cpu',weights_only=False);g=d['actor']['gravity'];g={**g,'verified_source_checkpoint_sha256':hashlib.sha256(Path(a.source).read_bytes()).hexdigest()};out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);torch.save(g,out);print('VERIFIED_GRAVITY_EXTRACTED',out,flush=True)
