"""Episode-level expert data and DAgger labels; observation is saved BEFORE stepping."""
import argparse,concurrent.futures,json,os,time
from pathlib import Path
import numpy as np

def episode(task):
    scenario,out,policy_path,beta=task
    from env import RobotEnv
    from expert import Expert
    path=Path(out)/f"{scenario['id']}.npz"
    if path.exists():
        return json.loads(path.with_suffix('.json').read_text())
    e=RobotEnv();obs=e.reset(scenario);ex=Expert(e)
    policy=None
    if policy_path:
        from models import Policy
        policy=Policy(policy_path,env=e);policy.reset()
    obs_list=[];acts=[];executed=[];phases=[];qdes=[];errs=[];times=[];reaching=[];directions=[];wheel_targets=[]
    start=time.perf_counter();rng=np.random.default_rng(scenario['seed']+17)
    while True:
        action,aux=ex.label()  # Updates expert memory on actual visited states, once per step.
        student=policy.act(obs) if policy is not None else action
        mixed=action if policy is None or rng.random()<beta else student
        obs_list.append(obs);acts.append(action);phases.append(aux['phase']);qdes.append(aux['q_des']);times.append(e.k*e.dt);reaching.append(aux['reach_active']);directions.append(aux['direction']);wheel_targets.append(aux['wheel_target'])
        obs,info=e.step(mixed);executed.append(info['applied']);errs.append(info['error'])
        if info['done']:break
    record=dict(scenario=scenario,steps=e.k,expert_queries=ex.queries,success=info['success'],reason=info['reason'],
                final_error=info['error'],min_error=min(errs),sim_seconds=e.k*e.dt,wall_seconds=time.perf_counter()-start,
                policy=policy_path,beta=beta,reference_duration_s=aux['reference_duration_s'],alignment='obs_t and expert_action_t before action execution; error is post-step')
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp.npz')
    np.savez_compressed(temp,obs=np.asarray(obs_list,dtype='float32'),action=np.asarray(acts,dtype='float32'),
        executed_action=np.asarray(executed,dtype='float32'),phase=np.asarray(phases,dtype='int8'),reach_active=np.asarray(reaching,dtype=bool),direction=np.asarray(directions,dtype='int8'),wheel_target=np.asarray(wheel_targets,dtype='float32'),
        q_des=np.asarray(qdes,dtype='float32'),error=np.asarray(errs,dtype='float32'),time=np.asarray(times,dtype='float32'))
    temp.replace(path)
    path.with_suffix('.json').write_text(json.dumps(record,indent=2))
    e.close();return record

def main():
    p=argparse.ArgumentParser();p.add_argument('--scenarios',required=True);p.add_argument('--out',required=True)
    p.add_argument('--limit',type=int);p.add_argument('--offset',type=int,default=0);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--policy');p.add_argument('--beta',type=float,default=0.)
    a=p.parse_args();rows=json.loads(Path(a.scenarios).read_text())[a.offset:];rows=rows[:a.limit] if a.limit else rows
    if a.policy:assert all(r['split'] in ['train','extra_train'] for r in rows),'DAgger cannot query validation/test'
    tasks=[(r,a.out,a.policy,a.beta) for r in rows]
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(a.workers,4)) as pool:
        for r in pool.map(episode,tasks):
            print(json.dumps({k:r[k] for k in ['scenario','steps','success','reason','final_error','wall_seconds']}),flush=True)

if __name__=='__main__':main()
