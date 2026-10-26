"""Read local data and check abstract task logic; NOT a simulator rollout."""
from pathlib import Path
import hashlib
import itertools
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / 'third_party/mikasa/mikasa_robo_suite/vla/memory_envs'
result = {'scope': 'Dataset inspection + abstract semantic checks, not simulator execution',
          'data': [], 'source_sha256': {}}
for name in ['batteries_checker_hard_vla', 'batteries_checker_easy_vla',
             'shell_game_shuffle_color_lamp_touch_vla', 'gather_and_recall_vla', 'timed_transfer_vla']:
    p = ENV / (name + '.py')
    result['source_sha256'][str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
for p in sorted((ROOT / 'data/audit').glob('batteries*')):
    d = pd.read_parquet(p / 'data/chunk-000/file-000.parquet')
    m = json.loads((p / 'source_rlds_metadata.json').read_text())
    lengths = d.groupby('episode_index', sort=True).size().tolist()
    length_order_matches = lengths == m['episode_lengths']
    length_multiset_matches = sorted(lengths) == sorted(m['episode_lengths'])
    sample = ROOT / 'data/samples' / (p.name + '-episode-000000.npz')
    with np.load(sample, allow_pickle=False) as a:
        shapes = {k: list(a[k].shape) for k in a.files}
    result['data'].append({'task': m['env_id'], 'episodes': len(lengths), 'frames': len(d),
        'mean_length': float(np.mean(lengths)), 'columns': list(d.columns),
        'metadata_success_once': sum(bool(x) for x in m['success_once']),
        'length_metadata_order_matches': length_order_matches,
        'length_metadata_multiset_matches': length_multiset_matches,
        'first_parquet_length': lengths[0], 'first_source_metadata_length': m['episode_lengths'][0],
        'sample_shapes': shapes})

# From evaluate(): checked flags are monotone; a first confirmation adds W_i
# to found_working_mask. Once all active indices are checked, found == W.
# Enumerate a superset of configured working counts, including zero, N=3/6.
checks = 0
for n in (3, 6):
    for working in itertools.product((False, True), repeat=n):
        checked = [False] * n
        found = [False] * n
        for i in range(n):
            newly_checked = not checked[i]
            checked[i] = True
            found[i] = found[i] or (working[i] and newly_checked)
        assert all(checked) and sum(found) == sum(working)
        checks += 1
result['abstract_battery_check'] = {'working_assignments': checks,
    'all_pass': True, 'assumption': 'Every insert-return-confirm event succeeds; all batteries returned home'}

# Two legal two-swap histories have identical visited-slot/color checklists
# but different answers to a red cue. This proves binding, NOT probabilistic
# joint-belief necessity (a deterministic permutation tracker solves it).
def slots_after(pairs):
    at_slot = list(range(3))
    for a, b in pairs:
        at_slot[a], at_slot[b] = at_slot[b], at_slot[a]
    return {color: at_slot.index(color) for color in range(3)}
a, b = slots_after([(0, 1), (1, 2)]), slots_after([(1, 2), (0, 1)])
assert a[0] != b[0]
result['shell_binding_counterexample'] = {'history_A': [[0,1],[1,2]],
    'history_B': [[1,2],[0,1]], 'color_to_slot_A': a, 'color_to_slot_B': b,
    'meaning': 'Binding required; deterministic tracker still sufficient with unambiguous observations'}
out = ROOT / 'runs/joint-task-suitability-2026-10-26.json'
out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
