"""POL-V evidence heads on synthetic events: the outcome lives only in v1's first time block, a "place" pattern 3x
larger is shared by v1 and v2 (what a re-encounter at the same spot sees)."""
import json

import numpy as np
import pytest
import torch
from sklearn.metrics import normalized_mutual_info_score, roc_auc_score

from pol.evidence import EvidenceHeads, ece, fit_evidence, item_id, sample_labelled

D, NT, NTOK, DF, Z, NPLACE = 32, 4, 8, 16, 3, 4
KW = dict(E=5, d_tok=D, d_feat=DF, n_time=NT, n_tok=NTOK, d_ctx=32)


def make_items(n, seed, occlude_frac=0.0, absent=None):
    rng = np.random.default_rng(seed)
    g = np.random.default_rng(123)  # patterns fixed across splits
    out_pat, place_pat = g.normal(size=(Z, NTOK, D)), g.normal(size=(NPLACE, NTOK, D))
    items = []
    for i in range(n):
        z = int(rng.integers(Z)) if absent is None else int(rng.choice([c for c in range(Z) if c != absent]))
        place, occ = int(rng.integers(NPLACE)), bool(rng.random() < occlude_frac)
        v1 = rng.normal(size=(NT, NTOK, D)) + 3 * place_pat[place]
        if not occ:
            v1[0] += out_pat[z]
        early = rng.normal(size=(2, 2, DF))
        early[:, :, 0] += 3.0 * occ  # the occluder is visible from the first post frames
        items.append({"task": "SealedBin-3", "family": "sealed_bin", "seed": i, "t": 0, "kind": "R+", "outcome": z,
                      "place": place, "occluded": occ, "pooled": rng.normal(size=(2, DF)), "act": rng.normal(size=(16, 7)),
                      "v1": v1, "v2": rng.normal(size=(NT, NTOK, D)) + 3 * place_pat[place], "w2": rng.normal(size=16),
                      "pooled2": rng.normal(size=(2, DF)), "has_v2": bool(rng.random() < 0.7), "early": early})
    return items


def batch(items):
    b = {k: torch.as_tensor(np.stack([it[k] for it in items]), dtype=torch.float32)
         for k in ("pooled", "act", "v1", "v2", "w2", "pooled2", "early")}
    b.update(has_v2=torch.as_tensor([it["has_v2"] for it in items]), task="SealedBin-3",
             label=torch.as_tensor([it["outcome"] for it in items]))
    return b


def train(items, steps=150, seed=0, val=None):
    torch.manual_seed(seed)
    lab = sample_labelled(items, 32, seed)
    m = EvidenceHeads({"SealedBin-3": Z}, **KW)
    return fit_evidence(m, {"SealedBin-3": batch(lab)}, {"SealedBin-3": batch(items)}, steps=steps, lr=1e-3,
                        val=val), lab


@pytest.fixture(scope="module")
def trained():
    items = make_items(400, 0, occlude_frac=0.5)
    return train(items, val=[batch(sample_labelled(make_items(200, 2, 0.5), 16, 1))])


def test_sample_labelled_is_stratified_and_reproducible():
    items = make_items(300, 0)
    items[0]["outcome"] = None
    items[1]["outcome"] = 3  # stuck: held out
    a, b = sample_labelled(items, 32, 7), sample_labelled(items, 32, 7)
    assert [item_id(x) for x in a] == [item_id(x) for x in b] and len(a) == 32
    assert {x["outcome"] for x in a} == {0, 1, 2}
    rare = [it for it in items if it["outcome"] != 2] + [it for it in items if it["outcome"] == 2][:1]
    assert sum(x["outcome"] == 2 for x in sample_labelled(rare, 32, 0)) == 1  # >= 1 per class present


def test_outcome_not_place_with_32_labels(trained):
    m, _ = trained
    test = [it for it in make_items(300, 1)]
    b = batch(test)
    with torch.no_grad():
        p = m.q_v1("SealedBin-3", b["pooled"], b["act"], b["v1"]).numpy()
        ll = m.loglik_v1("SealedBin-3", b["pooled"], b["act"], b["v1"])
    y = b["label"].numpy()
    assert roc_auc_score(y, p, multi_class="ovr") >= 0.9
    assert normalized_mutual_info_score([it["place"] for it in test], p.argmax(1)) < 0.1
    # loglik is the posterior / prior ratio, ensemble-averaged in probability space
    assert torch.allclose(ll, torch.as_tensor(p).log() - m.pi(1, "SealedBin-3").log(), atol=1e-5)


def test_omega_separates_visible_from_occluded(trained):
    m, _ = trained
    test = make_items(300, 3, occlude_frac=0.5)
    b = batch(test)
    with torch.no_grad():
        om = m.omega("SealedBin-3", b["pooled"], b["act"], b["early"]).numpy()
    assert roc_auc_score([not it["occluded"] for it in test], om) >= 0.9


def test_omega_is_causal():
    m = EvidenceHeads({"SealedBin-3": Z}, **KW).eval()
    b = batch(make_items(8, 0))
    late = torch.cat([b["early"], torch.randn(8, 6, 2, DF)], 1)  # frames after the first 2 must not matter
    with torch.no_grad():
        assert torch.equal(m.omega("SealedBin-3", b["pooled"], b["act"], b["early"]),
                           m.omega("SealedBin-3", b["pooled"], b["act"], late))


def test_calibrated_ece(trained):
    m, _ = trained
    b = batch(make_items(400, 4, occlude_frac=0.5))  # the val distribution: half the events show no outcome
    with torch.no_grad():
        p = m.q_v1("SealedBin-3", b["pooled"], b["act"], b["v1"])
    assert ece(p, b["label"]) <= 0.1


def test_absent_class_stays_finite():
    items = make_items(200, 5, absent=2)
    m, _ = train(items, steps=40)
    b = batch(items[:50])
    with torch.no_grad():
        for x in (m.loglik_v1("SealedBin-3", b["pooled"], b["act"], b["v1"]),
                  m.loglik_v2("SealedBin-3", b["v2"], b["w2"], b["pooled2"]),
                  m.prior_logits("SealedBin-3", b["pooled"], b["act"]),
                  m.omega("SealedBin-3", b["pooled"], b["act"], b["early"])):
            assert torch.isfinite(x).all()
        assert (m.q_v1("SealedBin-3", b["pooled"], b["act"], b["v1"])[:, 2] > 0).all()
    assert (m.pi(1, "SealedBin-3") > 0).all()


def test_polv_variant_smoke(tmp_path, monkeypatch):
    from pol.assets import load_assets
    from pol.train_stage_a import main
    from test_stage_a import write_dataset

    monkeypatch.setenv("WANDB_MODE", "disabled")
    data, feats = write_dataset(tmp_path / "ds")
    out = tmp_path / "POLV"
    main(["--variant", "POLV", "--data", data, "--features", feats, "--tasks", "SealedBin-3", "--out", str(out),
          "--steps", "50"])
    prov = json.loads((out / "assets.json").read_text())
    assert prov["label_seed"] == 0 and prov["labelled_ids"]["train"]
    heads = load_assets(out, feats)["evidence_heads"]
    assert isinstance(heads, EvidenceHeads)
