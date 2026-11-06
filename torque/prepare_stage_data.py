from pathlib import Path
import concurrent.futures,json,time,numpy as np,torch
NAV=None
def prepare(p):
 global NAV
 from .optimized_policy import PhaseOnly,stage_features
 if NAV is None:NAV=torch.load('runs/frozen_navigation.pt',map_location='cpu',weights_only=False)['base_memory']
 dest=Path('data/stage_original')/p.name
 if dest.exists():return json.loads(dest.with_suffix('.json').read_text())
 meta=json.loads(p.with_suffix('.json').read_text());phase=PhaseOnly(NAV)
 with np.load(p) as d:
  obs=d['obs'];initial=obs[0];xs=[];stages=[]
  for o in obs:
   state=phase.advance(o);xs.append(stage_features(o,initial,state,phase.reference_time));stages.append(state)
  dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,x=np.asarray(xs,dtype='float32'),y=d['y'][:,4:11],reference=d['reference'],error=np.linalg.norm(obs[:,18:21]-obs[:,33:36],axis=1).astype('float32'),phase=np.asarray(stages,dtype='int8'))
 dest.with_suffix('.json').write_text(json.dumps(meta,indent=2));return meta
if __name__=='__main__':
 start=time.perf_counter();paths=sorted(Path('data/original_train').glob('*.npz'))
 with concurrent.futures.ProcessPoolExecutor(max_workers=12) as pool:
  rows=list(pool.map(prepare,paths))
 Path('data/stage_original.summary.json').write_text(json.dumps(dict(episodes=len(rows),wall_s=time.perf_counter()-start),indent=2));print('STAGE_DATA_READY',len(rows),flush=True)
