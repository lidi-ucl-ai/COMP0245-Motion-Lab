from pathlib import Path
import argparse,json,concurrent.futures,hashlib,time,numpy as np
ENV=None
def relabel(task):
 global ENV
 folder,path=task
 from env import RobotEnv
 from .markov_teacher import MarkovTeacher
 from .physical_teacher import align_missing_inertias
 from .learned_gravity import body_gravity
 if ENV is None:ENV=RobotEnv();align_missing_inertias(ENV)
 dest=Path('data/markov_physical')/(folder+'_'+path.name)
 if dest.exists():return json.loads(dest.with_suffix('.json').read_text())
 meta=json.loads(path.with_suffix('.json').read_text());assert 'train' in meta['scenario']['split']
 with np.load(path) as d:
  obs=d['obs'][::2];initial=d['obs'][0]
  if folder=='original_train':
   with np.load(Path('data/stage_original')/path.name) as stage:xs=stage['x'][::2]
  else:xs=d['x'][::2]
 horizon=float(xs[0,69])*15;teacher=MarkovTeacher(ENV,initial,horizon);rows={k:[] for k in ['y','gravity','velocity','response','local']}
 for o in obs:
  _,aux=teacher.label(o)
  for key,val in [('y',aux['raw']),('gravity',aux['gravity']),('velocity',aux['velocity']),('response',aux['error_response']),('local',aux['local_valid'])]:rows[key].append(np.copy(val))
 dest.parent.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(dest,x=np.column_stack([xs,body_gravity(obs)]).astype('float32'),obs=obs,**{k:np.asarray(v,dtype='float32') for k,v in rows.items()},error=np.linalg.norm(obs[:,18:21]-obs[:,33:36],axis=1).astype('float32'))
 r=dict(source=str(path),scenario=meta['scenario'],samples=len(obs),teacher='observed-state physical-inertia-aligned, kd30, pre-slew raw torques',horizon=horizon,local_rows=int(np.sum(rows['local'])))
 dest.with_suffix('.json').write_text(json.dumps(r,indent=2));return r
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=8);a=p.parse_args();start=time.perf_counter()
 tasks=[(folder,path) for folder in ['original_train','mixed_arm','hold_mix','hold_lowmix','hold_pure'] for path in sorted(Path('data',folder).glob('*.npz'))]
 with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
  rows=[]
  for r in pool.map(relabel,tasks):
   rows.append(r)
   if len(rows)%32==0:print('RELABEL_PROGRESS',len(rows),len(tasks),flush=True)
 result=dict(episodes=len(rows),samples=sum(r['samples'] for r in rows),local_rows=sum(r['local_rows'] for r in rows),wall_s=time.perf_counter()-start,teacher_sources={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ['markov_teacher.py','physical_teacher.py']},scope='only old/new training observations; zero validation/test targets')
 Path('data/markov_physical.summary.json').write_text(json.dumps(result,indent=2));print('MARKOV_RELABEL_DONE',json.dumps(result),flush=True)
