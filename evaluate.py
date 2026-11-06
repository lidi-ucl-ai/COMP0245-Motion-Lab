"""Run the bundled report scenarios with frozen checkpoints and write JSON results."""
import argparse
import collections
import concurrent.futures
import json
from pathlib import Path

import numpy as np

from policies import MODELS, ROOT, load_policy, reset_for_policy
from env import RobotEnv


def rollout(task):
    key, scenario, cap = task
    e = RobotEnv()
    try:
        obs = reset_for_policy(key, e, scenario)
        policy = load_policy(key, e, obs)
        info = {}
        for _ in range(cap or int(np.ceil(scenario['duration'] / e.dt)) + 1):
            action = np.asarray(policy.act(obs))
            if action.shape != (11,) or not np.isfinite(action).all():
                raise ValueError(f'{key} returned an invalid action')
            obs, info = e.step(action)
            if info['done']:
                break
        return dict(model=key, id=scenario['id'], group=scenario.get('group'),
                    steps=e.k, simulation_seconds=e.k * e.dt, success=bool(info['success']),
                    reason=info['reason'] if info['done'] else 'smoke_cap',
                    final_error_m=float(info['error']), sha256=MODELS[key].get('sha256'))
    finally:
        e.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, default='torque')
    parser.add_argument('--suite', choices=('matched', 'ablation', 'robustness', 'extreme', 'concurrent', 'high-reach'))
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--smoke', action='store_true', help='Run 120 physics steps for every entry.')
    args = parser.parse_args()
    entry = MODELS[args.model]
    default_suites = {'raic_integrated': 'matched', 'ljc_parallel': 'concurrent',
                      'ljc_sequential': 'concurrent', 'parallel_expert': 'concurrent', 'ljc_pre_high': 'high-reach'}
    suite = args.suite or default_suites.get(args.model, 'ablation' if entry['group'] == 'Ablations' else 'matched')
    if args.smoke:
        suite = 'matched'
    if suite == 'extreme' and args.model != 'jacobian':
        parser.error('The report extreme suite is for LJC.')
    filename = {'matched': 'test256.json', 'ablation': 'ablation128.json', 'robustness': 'robustness80.json',
                'extreme': 'jac_extreme52.json',
                'concurrent': 'concurrent_test100.json', 'high-reach': 'high_reach128.json'}[suite]
    scenes = json.loads((ROOT / 'configs' / filename).read_text())
    jobs = [(key, scenes[0], 120) for key in MODELS] if args.smoke else [(args.model, s, None) for s in scenes[:args.limit]]
    if args.workers == 1:
        rows = list(map(rollout, jobs))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(rollout, jobs))
    for row in rows:
        print(json.dumps(row), flush=True)
    if not args.smoke:
        scored = [r for r in rows if r['group'] != 'known_extreme_diagnostic']
        counts = collections.Counter(r['group'] for r in scored)
        successes = collections.Counter(r['group'] for r in scored if r['success'])
        summary = dict(suite=suite, total=len(scored), successful=sum(successes.values()),
                       groups={group: [successes[group], count] for group, count in counts.items()})
        if suite == 'extreme':
            summary['diagnostics'] = [r for r in rows if r['group'] == 'known_extreme_diagnostic']
        print(json.dumps(summary, indent=2))
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(dict(summary=summary, rows=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
