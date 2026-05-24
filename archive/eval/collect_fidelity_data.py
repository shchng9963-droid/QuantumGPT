"""Fast data collection for fidelity predictor.

Runs in background: nohup python eval/collect_fidelity_data.py &
Produces eval/fidelity_data.json when done.
"""
import sys, time, json
import numpy as np
sys.path.insert(0, '/home/wangshuchang/quantumgpt')

from backends.synthetic_drift import (
    SyntheticDriftBackend, STABLE, LINEAR_DECAY, SUDDEN_DEGRADATION, DIURNAL_CYCLE
)
from bench.circuits import ghz, qft, bernstein_vazirani, vqe_ansatz, qaoa_maxcut
from eval.fidelity_predictor import extract_circuit_features, extract_backend_features

CIRCUITS = {
    'ghz_3': lambda: ghz(3),
    'ghz_5': lambda: ghz(5),
    'qft_4': lambda: qft(4),
    'bv_5': lambda: bernstein_vazirani('10110'),
    'vqe_4': lambda: vqe_ansatz(4),
    'qaoa_4': lambda: qaoa_maxcut(4),
}

DRIFT_CONDITIONS = [
    ('stable', STABLE, [0.0]),
    ('linear', LINEAR_DECAY, [0, 5, 10, 15, 20]),
    ('sudden', SUDDEN_DEGRADATION, [0, 3, 5, 7, 10]),
    ('diurnal', DIURNAL_CYCLE, [0, 6, 12, 18, 24]),
]

BACKENDS = ['FakeBrisbane', 'FakeSherbrooke']

print('Extracting circuit features...')
cf_cache = {}
for cname, builder in CIRCUITS.items():
    circ, _ = builder()
    cf_cache[cname] = extract_circuit_features(circ, cname)
    print(f'  {cname}: depth={cf_cache[cname].depth}, 1q={cf_cache[cname].n_1q_gates}, 2q={cf_cache[cname].n_2q_gates}')

total = len(CIRCUITS) * len(BACKENDS) * sum(len(c[2]) for c in DRIFT_CONDITIONS)
print(f'\nCollecting {total} samples (est. {total*0.45:.0f}s)...')

samples = []
t0 = time.time()
count = 0

for bname in BACKENDS:
    for dname, profile, tpoints in DRIFT_CONDITIONS:
        backend = SyntheticDriftBackend(bname, profile)
        for t in tpoints:
            backend.set_time(t)
            bf = extract_backend_features(backend)
            for cname, builder in CIRCUITS.items():
                circ, _ = builder()
                cf = cf_cache[cname]
                try:
                    result = backend.run(circ, shots=2048)
                    fidelity = result.fidelity if result.fidelity else 0.0
                except Exception as e:
                    print(f'  ERROR: {cname}/{bname}/t={t}: {e}')
                    fidelity = 0.0
                samples.append({
                    'circuit_features': cf.to_array().tolist(),
                    'backend_features': bf.to_array().tolist(),
                    'fidelity': fidelity,
                    'metadata': {'circuit': cname, 'backend': bname, 'drift': dname, 'time': t},
                })
                count += 1
                if count % 12 == 0:
                    elapsed = time.time() - t0
                    eta = elapsed / count * (total - count)
                    print(f'  [{count}/{total}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s | last fidelity={fidelity:.4f}')

elapsed = time.time() - t0
print(f'\nDone: {len(samples)} samples in {elapsed:.1f}s')

with open('/home/wangshuchang/quantumgpt/eval/fidelity_data.json', 'w') as f:
    json.dump(samples, f)
print('Saved to eval/fidelity_data.json')

fids = [s['fidelity'] for s in samples]
print(f'Fidelity stats: min={min(fids):.4f}, max={max(fids):.4f}, mean={np.mean(fids):.4f}, std={np.std(fids):.4f}')

# Signal completion
with open('/home/wangshuchang/quantumgpt/eval/.collect_done', 'w') as f:
    f.write(f'done at {time.time()}\n{len(samples)} samples\n')
