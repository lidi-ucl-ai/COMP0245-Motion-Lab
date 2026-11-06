"""Use the unchanged strict evaluator with a neural-specialist loader and GC."""
from pathlib import Path
import argparse,json,time,csv,gc,concurrent.futures,numpy as np

def run(task):
 gc.collect()
 from .domain_policy import DomainPolicy
 from . import evaluate_optimized
 DomainPolicy.last_selection=[];DomainPolicy.last_active_post={}
 result=evaluate_optimized.run(task, policy_class=DomainPolicy)
 if DomainPolicy.last_selection:
  result['domain_selection']=DomainPolicy.last_selection.copy();result['active_output_constraints']=DomainPolicy.last_active_post.copy();result['policy_family']='learned_tree_selected_frozen_neural_specialists'
  (Path(task[2])/(result['id']+'.json')).write_text(json.dumps(result,indent=2))
 gc.collect();return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--scenarios',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=8);a=p.parse_args()
 with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
  rows=[]
  for r in pool.map(run,[(s,a.checkpoint,a.out) for s in json.loads(Path(a.scenarios).read_text())]):rows.append(r);print(json.dumps(dict(id=r['id'],success=r['success'],done=len(rows))),flush=True)
 summary=dict(checkpoint=a.checkpoint,n=len(rows),successes=sum(r['success'] for r in rows),mean_capped_seconds=float(np.mean([r['sim_seconds'] for r in rows])),mean_error_m=float(np.mean([r['final_error_m'] for r in rows])),mean_base_error_m=float(np.mean([r['base_goal_error_m'] for r in rows])),exceptions=sum(r['exception'] is not None for r in rows),model_type='frozen neural specialist bank with supervised domain selection tree')
 (Path(a.out)/'summary.json').write_text(json.dumps(summary,indent=2));print('DOMAIN_EVAL_DONE',json.dumps(summary),flush=True)
