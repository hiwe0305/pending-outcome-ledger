"""Abstract necessity-gate check for the Sort-by-Outcome extension; NOT a simulator rollout.

Per battery: hidden quality W, tester box with hidden interior. An insertion seats with
probability 1-eps; otherwise the battery falls into a covered spill tray. The lamp is lit
iff a seated battery works, and the policy sees the lamp only with probability v. After
the attempt the gripper reaches in: finger width reveals held (seated) vs empty (spilled).
Routing: working -> home slot, broken -> reject bin. Success requires every battery routed
by its true quality within the step budget.
"""
from pathlib import Path
import json
import random

ROOT = Path(__file__).resolve().parents[1]
COST = {'test': 1.0, 'retrieve': 1.5, 'route': 0.5}
INF = 10**6


def gate_witness(eps, p_w):
    """Same marginals, different joint, different action after the same grasp evidence."""
    # After one attempt with lamp observed OFF; states (L, Q-evidence).
    joint = {('box', 'NW'): (1 - eps) * (1 - p_w), ('spill', 'U'): eps}
    z = sum(joint.values())
    joint = {k: v / z for k, v in joint.items()}
    pl = {l: sum(v for (a, _), v in joint.items() if a == l) for l in ['box', 'spill']}
    pq = {q: sum(v for (_, b), v in joint.items() if b == q) for q in ['NW', 'U']}
    product = {(l, q): pl[l] * pq[q] for l in pl for q in pq}

    def q_given_box(b):  # grasp says held => L=box
        m = {q: b.get(('box', q), 0.0) for q in ['NW', 'U']}
        s = sum(m.values())
        return {q: round(v / s, 4) for q, v in m.items()}
    keyed = lambda b: {f'{l},{q}': round(v, 4) for (l, q), v in b.items()}
    return {'joint': keyed(joint), 'product_of_marginals': keyed(product),
            'marginals_equal': True, 'P(Q|held) joint': q_given_box(joint),
            'P(Q|held) product': q_given_box(product)}


def p_working_after_offs(k, eps, p_w):
    """Independent-marginal filter: k lamp-off attempts, grasp evidence NOT propagated to Q."""
    return p_w * eps**k / (p_w * eps**k + 1 - p_w)


def run_battery(w, policy, eps, v, p_w, rng, knob):
    """Returns (cost, tests, routed_correctly).

    knob = tau for 'independent' (route reject once P(broken) >= tau);
    knob = (k_spill, k_missed) for 'joint': after that many retests of an untested battery
    (spilled, or seated with the lamp cue missed) route by prior majority.
    """
    cost, tests, offs, retests = 0.0, 0, 0, 0
    guess = p_w >= 0.5  # prior-majority guess: True = route home
    while True:
        cost += COST['test']
        tests += 1
        seated = rng.random() > eps
        lamp = (seated and w) if rng.random() < v else None
        if not seated:
            cost += COST['retrieve']  # every policy must fetch it from the spill tray
        end = cost + COST['route']
        if policy == 'oracle':
            return end, tests, True
        if policy == 'checklist':  # remembers "done", not the outcome; cue gone at routing
            return end, tests, w  # always routes home
        if lamp is True:
            return end, tests, w
        if policy == 'joint':
            if seated and lamp is False:  # off + held => tested broken
                return end, tests, not w
            limit = knob[0] if not seated else knob[1]  # spilled vs missed cue
            if retests >= limit:  # still untested: gamble on prior
                return end, tests, guess == w
            retests += 1
            continue
        if policy == 'independent':
            if lamp is False:
                offs += 1
                if 1 - p_working_after_offs(offs, eps, p_w) >= knob:
                    return end, tests, not w
            continue
        if policy == 'user':
            action = user_policy(seated, lamp, offs)
            if action != 'retest':
                return end, tests, (action == 'home') == w
            offs += lamp is False
            continue
        raise ValueError(policy)


def user_policy(held, lamp, previous_offs):
    """Optional shortcut policy to stress-test the gate.

    held: bool from finger width; lamp: True/False/None (missed); previous_offs: int.
    Return 'home', 'reject' or 'retest'; return None to skip this policy.
    """
    # TODO(user): viết một shortcut policy bạn nghĩ có thể thắng mà không cần joint belief.
    return None


def evaluate(policy, knob=None, n=6, eps=0.2, v=0.9, p_w=0.5, budget=13.0, episodes=20000, seed=17):
    rng = random.Random(seed)
    wins = correct = tests = 0
    for _ in range(episodes):
        total, ok = 0.0, True
        for _ in range(n):
            c, t, good = run_battery(rng.random() < p_w, policy, eps, v, p_w, rng, knob)
            total += c
            tests += t
            ok &= good
        correct += ok
        wins += ok and total <= budget
    return {'sr': wins / episodes, 'routing_acc_no_budget': correct / episodes,
            'tests_per_battery': round(tests / episodes / n, 3)}


KNOBS = {'independent': [0.5, 0.9, 0.99, 0.999],
         'joint': [(s, m) for s in [0, 1, 2, INF] for m in [0, 1, INF]]}


def best(policy, **kw):
    """Best knob per cell: generous to every baseline."""
    rows = [dict(evaluate(policy, k, **kw), knob=k) for k in KNOBS.get(policy, [None])]
    return max(rows, key=lambda r: r['sr'])


# ---------------------------------------------------------------------------
# v2: memory representations under one shared observation model (H3, H5).
#
# Every memory is scored as the Bayes-optimal reader of exactly the information
# it keeps, so differences come from what is remembered, not from rule tuning.
# Per attempt, latent draws: seated (1-eps), cue occluded (occ), policy-call
# frame lands in the cue window (hit), KEM short-term anchors cover the window
# (cover), reading flipped (noise). Lamp lit iff seated and working.
#   persistent lamp: lit while seated, so hit = 1 (Hard-like tester box)
#   transient lamp: brief flash, a policy-call frame catches it only w.p. hit
# Deferred routing: test every battery first, sort at the end (MIKASA Checklist
# definition: "report the result set"), so memory must survive the whole episode.
# KEM-like is generous: unbounded buffer and correct identity binding.
# ---------------------------------------------------------------------------
SCENARIOS = {
    'S1-simple': dict(lamp='persistent', deferred=False, eps=0.2, occ=0.1, hit=1.0, cover=0.5, noise=0.0),
    'S2-transient': dict(lamp='transient', deferred=False, eps=0.2, occ=0.1, hit=0.5, cover=0.5, noise=0.0),
    'S3-compound': dict(lamp='transient', deferred=True, eps=0.2, occ=0.2, hit=0.5, cover=0.5, noise=0.1),
}
MEMORIES = ['oracle', 'eem+kem', 'eem', 'bkey-count', 'bkey-1bit', 'kem', 'independent', 'checklist']
TAUS = [0.5, 0.8, 0.9, 0.95, 0.99, 0.999]
# Retest an unsure battery with probability rho: lets a tie at p=0.5 be split
# (retest some, gamble others) so tight budgets don't punish a better-informed memory.
RHOS = [1.0, 0.75, 0.5, 0.25]
KNOBS_V2 = [(t, r) for t in TAUS for r in RHOS if t > 0.5 or r == 1.0]
BUDGETS = [12.0, 14.0, 17.0]
K_MAX = 5


def features(rep, a, latest=True):
    """What memory `rep` keeps from attempt a = (seated, reading, hit, cover)."""
    seated, reading, hit, cover = a
    frame = reading if hit else None  # cue seen at a policy-call frame
    if rep in ('eem', 'bkey-count'):
        return frame, seated
    if rep == 'eem+kem':  # belief fed by intra-chunk capture
        return reading, seated
    if rep == 'bkey-1bit':
        return 'W' if frame is True else 'NW' if frame is False and seated else 'U'
    if rep == 'kem':  # stores on-events; off is a non-event, seen only via anchors
        event = reading is True
        return (event, reading is False and cover, seated) if latest else event
    if rep == 'independent':  # lamp evidence only, grasp result not linked to Q
        return frame
    raise ValueError(rep)


def likelihoods(sc):
    """P(feature | W) for every (rep, latest) by exact enumeration of attempt latents."""
    table = {}
    for seated, ps in [(True, 1 - sc['eps']), (False, sc['eps'])]:
        for occ, po in [(True, sc['occ']), (False, 1 - sc['occ'])]:
            for hit, ph in [(True, sc['hit']), (False, 1 - sc['hit'])]:
                for cover, pc in [(True, sc['cover']), (False, 1 - sc['cover'])]:
                    for flip, pf in [(True, sc['noise']), (False, 1 - sc['noise'])]:
                        p = ps * po * ph * pc * pf
                        for w in (True, False):
                            reading = None if occ else ((seated and w) != flip)
                            a = (seated, reading, hit, cover)
                            for rep in MEMORIES[1:-1]:
                                for latest in (True, False):
                                    key = (rep, latest, features(rep, a, latest))
                                    table.setdefault(key, [0.0, 0.0])[0 if w else 1] += p
    return table


def posterior(lik, rep, attempts, p_w):
    """P(W | what rep keeps of this battery's attempts)."""
    if rep == 'bkey-1bit':
        attempts = attempts[-1:]  # one overwritten flag
    l1, l0 = p_w, 1 - p_w
    for i, a in enumerate(attempts):
        f1, f0 = lik[(rep, i == len(attempts) - 1 or rep != 'kem', features(rep, a, i == len(attempts) - 1))]
        l1, l0 = l1 * f1, l0 * f0
    return l1 / (l1 + l0)


def final_summary(rep, attempts):
    """What survives until deferred routing for memories that lose information."""
    if rep == 'kem':
        return any(a[1] is True for a in attempts)
    if rep == 'bkey-1bit':
        return features(rep, attempts[-1])
    return None


def stratum(latents):
    if any(not s for s, o, f in latents):
        return 'spill'
    if any(o for s, o, f in latents):
        return 'occluded'
    if any(f for s, o, f in latents):
        return 'noisy'
    return 'clean'


def run_v2(rep, knob, sc, lik, n=6, p_w=0.5, episodes=10000, seed=17, table=None):
    """Returns per-episode (cost, all_correct), per-stratum counts and summary counts."""
    tau, rho = knob
    rng = random.Random(seed)  # environment stream, identical across memories
    coin = random.Random(seed + 1000)  # policy randomization only
    episodes_out, strata, counts, tests = [], {}, {}, 0
    for _ in range(episodes):
        cost, ok = 0.0, True
        for _ in range(n):
            w = rng.random() < p_w
            attempts, latents = [], []
            while True:
                seated = rng.random() >= sc['eps']
                occ = rng.random() < sc['occ']
                hit = rng.random() < sc['hit']
                cover = rng.random() < sc['cover']
                flip = rng.random() < sc['noise']
                reading = None if occ else ((seated and w) != flip)
                attempts.append((seated, reading, hit, cover))
                latents.append((seated, occ, flip))
                cost += COST['test'] + (0 if seated else COST['retrieve'])
                if rep in ('oracle', 'checklist'):
                    break
                p = posterior(lik, rep, attempts, p_w)
                if max(p, 1 - p) >= tau or len(attempts) >= K_MAX or coin.random() >= rho:
                    break
            tests += len(attempts)
            cost += COST['route']
            if rep == 'oracle':
                home = w
            elif rep == 'checklist':
                home = p_w >= 0.5
            elif sc['deferred'] and rep in ('kem', 'bkey-1bit'):
                s = final_summary(rep, attempts)
                c = counts.setdefault(s, [0, 0])
                c[0 if w else 1] += 1
                c1, c0 = (table or {}).get(s, (1, 1))
                home = c1 >= c0
            else:
                home = p >= 0.5
            good = home == w
            ok &= good
            st = strata.setdefault(stratum(latents), [0, 0])
            st[0] += good
            st[1] += 1
        episodes_out.append((cost, ok))
    return episodes_out, strata, counts, tests / episodes / n


def evaluate_v2(rep, knob, sc, lik, **kw):
    table = None
    if sc['deferred'] and rep in ('kem', 'bkey-1bit'):  # fit the reader on a separate run
        _, _, table, _ = run_v2(rep, knob, sc, lik, seed=29, **kw)
    eps_out, strata, _, tpb = run_v2(rep, knob, sc, lik, table=table, **kw)
    m = len(eps_out)
    return {'sr': {b: sum(ok and c <= b for c, ok in eps_out) / m for b in BUDGETS},
            'acc_no_budget': sum(ok for _, ok in eps_out) / m, 'tests_per_battery': round(tpb, 3),
            'battery_acc_by_stratum': {k: round(v[0] / v[1], 4) for k, v in sorted(strata.items())},
            'stratum_share': {k: round(v[1] / sum(x[1] for x in strata.values()), 4)
                              for k, v in sorted(strata.items())}}


def main_v2():
    out = {}
    for name, sc in SCENARIOS.items():
        lik = likelihoods(sc)
        rows = {rep: {k: evaluate_v2(rep, k, sc, lik)
                      for k in (KNOBS_V2 if rep not in ('oracle', 'checklist') else [(1.0, 1.0)])}
                for rep in MEMORIES}
        ref = lambda rep: rows[rep].get((0.95, 1.0), rows[rep].get((1.0, 1.0)))
        best = {rep: {b: max(((r['sr'][b], k) for k, r in rows[rep].items())) for b in BUDGETS}
                for rep in MEMORIES}
        out[name] = {'scenario': sc, 'best_sr_by_budget': best,
                     'frontier_rho_1': {rep: {k[0]: {m: r[m] for m in ('acc_no_budget', 'tests_per_battery')}
                                              for k, r in rows[rep].items() if k[1] == 1.0} for rep in MEMORIES},
                     'strata_at_tau_0.95': {rep: ref(rep)['battery_acc_by_stratum'] for rep in MEMORIES},
                     'stratum_share': ref('eem')['stratum_share']}
    return out


def main():
    out = {'scope': 'Abstract POMDP sanity check of the proposed extension, not simulator execution',
           'cost': COST, 'witness': gate_witness(0.2, 0.5), 'sweep': [], 'frontier': []}
    policies = ['oracle', 'joint', 'independent', 'checklist']
    if user_policy(True, False, 0) is not None:
        policies.append('user')
    for eps in [0.0, 0.1, 0.2, 0.3]:
        for budget in [11.0, 13.0, 16.0]:
            row = {'eps': eps, 'budget': budget}
            for p in policies:
                r = best(p, eps=eps, budget=budget)
                row[p] = r['sr'] if r['knob'] is None else (r['sr'], r['knob'])
            out['sweep'].append(row)
    # Accuracy/test-count frontier at eps=0.2, no budget pressure.
    for p in ['joint', 'independent']:
        for k in KNOBS[p]:
            out['frontier'].append(dict(evaluate(p, k, budget=INF), policy=p, knob=k))
    (ROOT / 'runs/necessity-gate-sorting.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out['witness'], indent=1))
    for r in out['sweep'] + out['frontier']:
        print(r)


if __name__ == '__main__':
    import sys
    if '--v2-only' not in sys.argv:
        main()
    v2 = main_v2()
    (ROOT / 'runs/necessity-gate-memories.json').write_text(json.dumps(v2, indent=1, default=str))
    for name, r in v2.items():
        print('==', name, 'stratum share', r['stratum_share'])
        for rep in MEMORIES:
            print(f"  {rep:12s} best SR {r['best_sr_by_budget'][rep]}  strata@0.95 {r['strata_at_tau_0.95'][rep]}")
