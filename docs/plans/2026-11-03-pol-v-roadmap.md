# POL-V: kiểm tra hiện trạng, kiến trúc mới, kế hoạch sửa (LaTeX → code) và roadmap thực nghiệm

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Chuyển POL từ "giả thuyết outcome học không nhãn" sang **POL-V**: ledger posterior weak-supervised có
hiệu chuẩn, observability đo bằng information gain kỳ vọng, và verification chọn theo VOI (nhìn lại hay thăm dò
vật lý), đủ bằng chứng cho Q1.

**Architecture:** Giữ event FSM, features, U-Suite, Stage B. Thay outcome model EM bằng các head likelihood
per-task (ensemble, ≤32 nhãn/task); ledger thành bộ lọc Bayes có data-association đúng và eviction theo giá trị;
thêm planner VOI nhỏ ngoài π0 chọn primitive kiểm chứng, đưa vào π0 như một intent token.

**Tech Stack:** như spec gốc (PyTorch, openpi `215abfb`, ManiSkill3 b15 + MIKASA `3d5a21a`, scikit-learn cho probe).

**Spec:** [`docs/specs/2026-10-26-pol-design.md`](../specs/2026-10-26-pol-design.md) (giữ §3–5, §6.1, §6.5–6.6, §7–9);
phần thay thế nằm ở §2 của tài liệu này. Bối cảnh kết quả: [`docs/HANDOFF.md`](../HANDOFF.md).

---

## 1. Kiểm tra: cái đang có có khớp với research question không?

Research question (mới): *Khi outcome của hành động của chính robot bị che lúc xảy ra và không lộ ra khi quay lại,
một memory giữ posterior theo từng event, biết event đó có quan sát được không, và chọn cách kiểm chứng theo giá
trị thông tin, có sửa được lỗi im lặng tốt hơn các memory lưu cái đã thấy ở cùng dung lượng không?*

| # | Chỗ | Hiện trạng (đã đọc) | Khớp? | Hệ quả |
|---|---|---|---|---|
| 1 | Abstract `main.tex:48`, Related `:143`, Conclusion `:504` | "hypotheses learned without labels", "label-free" | **Không** | Claim đã bị round 1 bác bỏ (6 biến thể ≈ chance) |
| 2 | Abstract `main.tex:43-44` | Câu chuyện "fingers close on the battery or on nothing" = suy luận từ v1 ∧ v2 | Khớp câu chuyện | Nhưng gate không đo nó (dòng 3) |
| 3 | `src/pol/labels.py:18` `GATE_TARGET` | BatteriesSort `({0},{1})`: working vs broken, **spill bị loại** | **Không** | Nhãn chỉ có trong v1 (đèn chớp) ⇒ gate đo "nhớ cue", không đo cập nhật muộn (H4) |
| 4 | `src/pol/ledger.py:90-104`, `main.tex:67` | Evict entry có entropy thấp nhất (resolved đi trước) | **Không** | Verdict "hỏng" bị đẩy ra trước lúc phân loại; FIFO có thể thắng POL |
| 5 | `src/pol/ledger.py:78-82` | `logp += w * ll` cho mọi entry trong ρ | **Không** | Một v2 được đếm cho nhiều entry (không phải Bayes) |
| 6 | `src/pol/outcome_model.py`, spec §6.2 | EM + InfoNCE + IIC, không nhãn | Thay thế | Định lý bất khả thi (§2.6) |
| 7 | ω (spec §6.3, `main.tex` §method) | Bất đồng dự đoán của các z học | Thay thế | Vô nghĩa khi z ≠ outcome; đo ra 0.44–0.52 |
| 8 | `src/jem/envs/expert.py:57-84`, `expert_bins.py:55` | Expert **đã** dùng VOI (`value_of_test`, `choose_verify`) với observation model cho sẵn | **Khớp, và quan trọng** | VOI không mới với dự án; phần mới là *học* observation model và chạy VOI trong policy. Expert = cận trên oracle-VOI |
| 9 | `main.tex:493` Limitations | "Verification is imitated, not planned" | Sẽ hết đúng | POL-V giải quyết đúng limitation này |
| 10 | BatteriesSort occluder | Sửa ở `eb03bb7` | Khớp | Dữ liệu cũ vẫn lộ; phải thu lại |
| 11 | SealedBin | Probe full-label 0.72, 9 positive val | Yếu | Cần cue mạnh hơn hoặc chỉ dùng cho ω/VOI |
| 12 | Baseline "frozen-after-v1" | `configs/pol/variants/POL.yaml` có `resolve: false` (memory-level) | Có một nửa | Cần thêm bản Stage A cho gate |

Kết luận: event FSM, features, U-Suite, Stage B design, baselines §7, W&B, provenance **giữ nguyên**. Outcome
model, ω, ledger update/eviction, gate target, và claim trong LaTeX **phải đổi**.

---

## 2. Kiến trúc POL-V

```
frames ─► SigLIP (frozen) ─► pooled / PCA tokens
proprio ─► gripper FSM ─► event j (G+ / R+ / R0, x_ee, task:kind)
                              │
          ┌───────────────────┴────────────────────┐
          ▼                                        ▼
  [A] evidence heads (per task, ensemble E=5)   [B] observability head
      ℓ1_j(z) = log p(v1_j | z, c_j)                ω_j = E_v1[KL(q(z|v1,c) ‖ p(z|c))]
      ℓ2_r(z) = log p(v2_r | z, c_r)                (expected info gain of the view)
          │                                        │
          └──────────────► [C] Ledger (K slots) ◄──┘
                 entry = {key, log-posterior, ω, H, age, consumed, summary}
                 association: one re-encounter → softmax over entries ∪ {null}
                 eviction: lowest keep-value (consumed / resolved-and-acted / oldest)
                              │
                              ▼
                 [D] VOI planner (outside π0)
                 for each pending j, each check e ∈ {look, probe}:
                   VOI(j,e) = E_o[max_d E[U(d,z)|o]] − max_d E[U(d,z)] − cost(e)
                 → intent ∈ {none, look@j, probe@j}
                              │
                              ▼
                 [E] π0 + memory: K ledger tokens + 1 intent token (Stage B)
```

### 2.1 Outcome space (thay cho M cụm học)
Tập outcome rời rạc **do task định nghĩa** (tên lớp biết trước, không cần sim log lúc chạy):
BatteriesSort `{seated_working, seated_broken, spill}`; SealedBin `{in, hidden_out, visible_out}`;
GatherInject như SealedBin. `stuck` (held-out) không có trong train. Entry dùng 8 chỗ softmax có sẵn (pad).

### 2.2 [A] Evidence heads — weak supervision có hiệu chuẩn
- Input v1: token PCA cửa sổ post `[4,32,256]` + pooled pre `[2,1152]`; input v2: như v1 tại re-encounter + `w2[16]`.
- Head: encoder Transformer 2 lớp hiện có (tái dùng `view_embed`, bỏ scene residual) → linear → `|Z|` logits.
  Huấn luyện discriminative `q_k(z | v_k, c)`; likelihood dùng trong ledger là **tỉ số hậu nghiệm/tiền nghiệm**
  `ℓ_k(z) = log q_k(z|v_k,c) − log π_k(z)` (π_k = tỉ lệ lớp lúc train head k) để cộng được hai nguồn bằng chứng.
- Nhãn: ≤ 32 event/task (cố định seed lấy mẫu, phân tầng theo lớp). Phần còn lại không nhãn dùng cho
  consistency (augment v1: dropout khối thời gian/token) — tuỳ chọn, ablation.
- Ensemble E=5 (seed khác), temperature scaling trên 16 nhãn val riêng ⇒ ECE.
- **Tại sao hợp lý với định lý §2.6:** nhãn neo z vào outcome; đồng thuận đa view không thể.

### 2.3 [B] Observability ω = expected information gain của view
- Định nghĩa: `ω_j = E_{v1 ~ p(v1|c_j)} KL(q(z|v1,c_j) ‖ p(z|c_j))`, ước lượng bằng head hồi quy nhỏ
  `ω̂(c_j, a_j, mask_j)` học trên train từ **KL thực hiện được** `KL(q(z|v1_j,c_j) ‖ p(z|c_j))` của head [A].
  Ở đây `p(z|c)` là prior head (MLP trên c, a; distilled từ q như cũ).
- Đầu vào `ω̂` **không có v1 của chính event** (chỉ context + 2 frame đầu cửa sổ post, nơi tấm che/tay đã hiện ra)
  ⇒ ω phân biệt "không thể thấy" (KL kỳ vọng ≈ 0) với "thấy nhưng mơ hồ".
- Bắt buộc so với baseline **occluder-classifier** (probe "tấm che hạ / tay che" trên cùng input). Nếu ω không
  thắng, trong paper ω = classifier đó và đóng góp chuyển sang cách VOI dùng nó (§2.5).

### 2.4 [C] Ledger — bộ lọc Bayes đúng
- Ghi: `logp_j = log p(z|c_j) + ℓ1_j(z)` (chuẩn hoá).
- Association (sửa dòng 5): với re-encounter r, `π(j|r) ∝ g_j · exp(−d²/2ρ² + β cos)` trên các entry trong ρ
  **cộng một lựa chọn null** (trọng số `π_0`), chuẩn hoá thành phân phối; cập nhật
  `logp_j ← log[(1−π(j|r))·p_j + π(j|r)·p_j·exp(ℓ2_r)/Z_j]` (mixture, tổng bằng chứng của một v2 = 1).
- Eviction (sửa dòng 4): `keep_value_j = 0` nếu `consumed`; nếu không, `= 1 − max_d E[U(d,z)]` thường hoá
  (entry càng có thể đổi quyết định càng giữ). Hoà → cũ nhất. `consumed` được set khi decision dùng entry đó
  (BatteriesSort: lúc pin được đặt vào slot/bin; SealedBin: lúc bấm xác nhận).
- Entry 50 số giữ layout; thêm cờ `consumed` vào ô `age` dạng dấu (âm = consumed) để không đổi ENTRY_DIM.

### 2.5 [D] VOI planner (đóng góp chính)
- Quyết định d và utility U(d, z) theo task (từ reward sim, viết trong `configs/pol/tasks/*.yaml`):
  BatteriesSort d ∈ {home, bin}: U=1 nếu đúng (`working→home`, `broken/spill→bin`... spill cần retrieve);
  SealedBin d ∈ {confirm, open_lid}.
- Phép kiểm e và mô hình quan sát `p(o | z, e)`:
  - `look`: quay camera/tay lại vùng event — `p(o|z)` = head [A] view-1 tại vị trí đó, độ tin cậy = ω̂ hiện tại.
  - `probe`: re-test (BatteriesSort) / mở nắp (SealedBin) — `p(o|z)` = head [A] học trên các event re-test.
  - cost(e) = số bước trung bình của primitive (đo từ demo) × λ_cost (λ_cost chốt trên val, trước Stage B).
- `VOI(j,e)` tính đóng với |Z| ≤ 4 và o rời rạc (argmax-bin 4 mức của q). Chọn `argmax VOI` nếu > 0, ngược lại none.
- Oracle-VOI = expert hiện tại (observation model thật) ⇒ cận trên; khoảng cách POL-V → oracle là chỉ số chính.

### 2.6 Mệnh đề bất khả thi (đóng góp lý thuyết)
Cho z* outcome, v1, v2 view, s nuisance chung (vị trí, vật). Nếu outcome **unrevealed** theo Definition 1,
`I(z*; v2) ≤ ε`. Với mọi encoder f1, f2 và z = argmax đồng thuận (InfoNCE/IIC/CCA giữa f1(v1), f2(v2)):
`I(z; z*) ≤ I(f2(v2); z*) ≤ ε` khi z là hàm của phần chia sẻ (đẳng thức tối ưu của các objective này — block
identifiability, Daunhawer et al. 2023). Hệ quả: mọi objective đồng thuận hai view bỏ qua outcome unrevealed; 6 run
round 1 là bằng chứng thực nghiệm. Viết như Proposition 2 (thay Proposition 1 "deferring dominates committing"
bằng remark).

### 2.7 [E] π0 + memory
Giữ spec §6.5. Thêm 1 token intent `MLP(onehot(none/look/probe) ⊕ PE(x_target))` vào khối memory. Primitive
kiểm chứng trong demo đã có (expert re-test, open_lid, sweep_behind), nên nhãn intent lấy từ log của expert lúc
train Stage B (log là evaluator-only; intent lúc eval do planner [D] sinh, không từ log).

---

## 3. Kế hoạch sửa LaTeX (`latex-paper/eem-paper/main.tex`)

| Vị trí | Sửa |
|---|---|
| Title | Giữ "Not Yet Seen", phụ đề: "Pending-Outcome Memory and Value-of-Information Verification for VLA Policies" |
| Abstract `:40-58` | Bỏ "learned without labels"; thêm: few-label calibrated posteriors, observability = expected information gain, VOI chọn look/probe; câu về mệnh đề bất khả thi |
| Intro `:61-106` | Contributions: (1) Definition 1 + U-Suite; (2) mệnh đề: đồng thuận đa view không học được outcome unrevealed; (3) POL-V; (4) thực nghiệm |
| Related `:107-155` | Thêm đoạn POMDP/belief-space (BEACON, HAVE, QMDP); active perception (CoMe-VLA, ActiveVLA); failure detection ít nhãn (Foresight, FAIL-Detect). Bỏ "label-free" `:143`. **Chỉ trích sau khi đọc full text** |
| §Pending Action Outcomes `:156-210` | Prop. 1 → Remark; thêm Proposition 2 (§2.6) + chứng minh ngắn |
| §Method `:211-295` | Viết lại theo §2.1–2.5, hình kiến trúc mới (thêm khối VOI), eviction và association mới |
| §U-Suite `:296-333` | BatteriesSort: tấm che trùm đèn; outcome 3 lớp; SealedBin cue mới (nếu làm) |
| §Experiments `:334-484` | H1 giữ; H2 giữ; H3 → ω thắng occluder-classifier; H4 → v1∧v2 > frozen-after-v1; **H6 mới**: VOI Pareto (success × re-check) trội imitated/always/never/AGM; thêm bảng "6 objective không giám sát" |
| Limitations `:485-498` | Bỏ câu "verification is imitated"; thêm: cần ≤32 nhãn/task; U(d,z) cho sẵn theo task |
| `results.tex` keys | Thêm `voi.*`, `prop.*`, `diag.*.ece`; README key list |

Làm LaTeX **sau** mỗi cổng thực nghiệm tương ứng (không viết kết quả trước khi có số).

---

## Global Constraints
- Không đưa `oracle_info`, `working_mask`, sim log vào input của bất kỳ variant nào ngoài Oracle; nhãn chỉ dùng ở
  Stage A cho head [A] (≤ 32/task, ghi seed lấy mẫu vào provenance) và cho các baseline có nhãn (dùng **cùng** tập nhãn).
- Mọi ngưỡng (λ_cost, η, số nhãn, E, ρ, π_0) chốt trên val **trước** Stage B; ghi vào `configs/pol/*.yaml`.
- Mọi variant: cùng feature, cùng Stage B, memory `[K,50]` (+1 intent token cho mọi variant có planner, kể cả AGM+VOI).
- Test CPU: `.venv-cpu/bin/python -m pytest tests/pol -q` phải xanh trước mỗi commit; local tối đa 2 process sim.

## Review Focus
1. Event không có re-encounter (thả vào bin): entry vẫn phải được dùng ở decision và đánh dấu `consumed`.
2. Re-encounter trong ρ của hai entry cùng lúc: tổng trọng số cập nhật = 1 − π_null, không đếm trùng.
3. VOI khi mọi entry đã resolved: planner trả `none`, không bao giờ chọn kiểm.
4. Lớp vắng trong 32 nhãn (spill hiếm): head phải vẫn cho xác suất > 0 (label smoothing / prior), không NaN.
5. ω̂ trên event mà tấm che hạ **sau** 2 frame đầu: không được dùng frame sau (causality).

---

## 4. Tasks code — Phase 0 và 1 (chi tiết); Phase 2–4 có plan riêng khi tới cổng

### Task 1: Nhãn quyết định 3 lớp và probe v1 / v2 / v1∧v2 (cổng của toàn bộ hướng)

**Files:** Modify `src/pol/labels.py:18-29`; Create `scripts/pol/probe_views.py`; Test `tests/pol/test_labels.py`.

**Interfaces — Produces:**
- `DECISION_TARGET = {"batteries_sort": ({0, 2}, {1}), "sealed_bin": ({0, 2}, {1}), "gather_inject": ({0}, {1, 2, 3})}`
  (positive = outcome làm đổi quyết định và bị ẩn; spill/visible_out **giữ** làm negative).
- `decision_target(family, outcome) -> bool | None`. `GATE_TARGET` giữ nguyên tên làm alias tới `DECISION_TARGET`.
- `scripts/pol/probe_views.py --data --features --tasks --labels-per-task 32 --out runs/<round>/probe_views.json`:
  per task, AUROC 3 probe (v1 only, v2 only, v1⊕v2) cho decision target + AUROC 3 lớp macro.

- [ ] Test `test_decision_target_keeps_spill_as_negative`: `decision_target("batteries_sort", 2) is False`,
  `decision_target("batteries_sort", 1) is True`, `decision_target("sealed_bin", 2) is False`.
- [ ] Run → FAIL (`decision_target` chưa có). Implement. Run → PASS; toàn suite xanh.
- [ ] Chạy `probe_views.py` trên round 1 (CPU, remote) cho SealedBin, GatherInject.
  **Cổng P0** (ghi vào HANDOFF): SealedBin `v1⊕v2 − max(v1, v2) ≥ 0.05` hoặc ghi rõ task không cần cập nhật muộn.
  BatteriesSort đợi dữ liệu round 2 (Task 2).
- [ ] Commit `labels: decision target keeps visible failures as negatives; view probes`.

### Task 2: Thu lại BatteriesSort (round 2, remote)

**Files:** Modify `scripts/pol/run_round.sh` (thêm `PCA_FROM` để dùng lại `data/features_round1/pca.npz`);
Modify `docs/HANDOFF.md`.
- [ ] `PCA_FROM` set ⇒ `precompute_features.py` nhận `--pca $PCA_FROM` thay vì `--fit-pca`. Kiểm bằng dry-run in lệnh.
- [ ] Remote: `git pull`; `check_envs.py BatteriesSort-3-VLA-v0` → `[]`; `mv data/pol/BatteriesSort-3 data/pol_leaky/`.
- [ ] `ROUND=round2 SEED_BASE=20000 TASKS=BatteriesSort-3 PCA_FROM=data/features_round1/pca.npz setsid nohup bash scripts/pol/run_round.sh …`
- [ ] **Cổng P1:** probe planb-check B (visible → shutter-down, broken vs working) ≤ 0.60;
  `probe_views.py` BatteriesSort: v1 (broken vs spill) ≤ 0.65, v2 (working vs broken) ≤ 0.65, v1⊕v2 decision ≥ 0.85.
  Trượt ⇒ dừng hướng POL-V, chuyển paper benchmark (Plan D).

### Task 3: Ledger — association có null và eviction theo keep-value

**Files:** Modify `src/pol/ledger.py:49-104`; Test `tests/pol/test_ledger.py`.

**Interfaces — Produces:**
- `Ledger(K, M, eta=0.3, omega0=0.0, rho=0.06, beta=0.0, eviction="value", p_null=0.1)`;
  `association(x, assoc) -> np.ndarray[n]` (giờ là **phân phối** trên entries, tổng = 1 − p_null·[có entry trong ρ]);
  `resolve(weights, loglik)` = cập nhật mixture §2.4; `mark_consumed(i)`; `Entry.consumed: bool = False`;
  `evict_index()` với `"value"` (mặc định mới), `"entropy"` và `"fifo"` giữ cho ablation.
- [ ] Tests: `test_one_reencounter_splits_evidence_between_two_close_entries` (hai entry cùng chỗ, ll quyết định:
  tổng dịch chuyển log-odds của hai entry ≤ một lần cập nhật đầy đủ); `test_consumed_entry_is_evicted_first`;
  `test_resolved_unconsumed_entry_outlives_pending_low_value` ; `test_nothing_in_rho_changes_nothing` (giữ).
- [ ] FAIL → implement → PASS; toàn suite xanh. Commit `ledger: mixture association with a null option, value eviction`.

### Task 4: Evidence heads weak-supervised + prior + ω̂

**Files:** Create `src/pol/evidence.py`; Modify `src/pol/train_stage_a.py` (variant `POLV`); Create
`configs/pol/variants/POLV.yaml`; Test `tests/pol/test_evidence.py`.

**Interfaces — Produces:**
- `class EvidenceHeads(nn.Module)(n_classes: dict[str,int], E=5, d_tok=256)`:
  `loglik_v1(task, pooled, act, v1) -> [B,Z]`, `loglik_v2(task, v2, w2, pooled2) -> [B,Z]`,
  `prior_logits(task, pooled, act) -> [B,Z]`, `omega(task, pooled, act, early_frames) -> [B]`, `calibrate(val_batch)`.
- `sample_labelled(items, n_per_task, seed) -> list` (phân tầng, ít nhất 1/lớp nếu có).
- Train: CE cho head v1/v2 trên tập nhãn (ensemble E), prior distilled từ q, ω̂ hồi quy KL thực hiện được (MSE).
- [ ] Tests: synthetic với outcome chỉ trong v1 khối 0 và nuisance "place" lớn hơn 3×: với 32 nhãn, AUROC ≥ 0.9 và
  `NMI(argmax, place) < 0.1`; ω̂ cao hơn trên event "visible" so với "occluded" (synthetic mask) — AUROC ≥ 0.9;
  ECE sau `calibrate` ≤ 0.1; lớp vắng trong nhãn không cho NaN (Review Focus 4).
- [ ] FAIL → implement → PASS; commit `stage A: few-label evidence heads, prior, expected-information-gain omega`.

### Task 5: Gates mới

**Files:** Modify `src/pol/gates.py` (`THRESHOLDS`, `compute_scores`, `decide`); Test `tests/pol/test_gates.py`.
- Thêm scorer: `frozen_v1` (chỉ ℓ1), `occluder_clf` (probe visible/occluded), `agm_same_labels` (verify head cùng 32 nhãn).
- Ngưỡng mới: G1 `pol ≥ 0.80`, `pol_shuffled ≤ 0.60`; **G4** `pol − frozen_v1 ≥ 0.05` trên unrevealed; G2 `omega ≥ 0.85`
  và `omega − occluder_clf ≥ 0.0` (báo cả hai); ECE ≤ 0.10. Exit code: 0 pass, 2 fail G1/G4, 3 fail G2 only.
- [ ] Tests: synthetic scores đi qua từng nhánh pass/fail mới. Commit `gates: decision target, frozen-after-v1, occluder baseline`.

**Cổng P2 (Stage A, remote, round 2 BS + round 1 SB/GI):** G1 và G4 qua ⇒ Phase 2. G1 qua, G4 trượt ⇒ paper RA-L
dạng B+D (không VOI). G1 trượt ⇒ Plan D.

---

## 5. Roadmap thực nghiệm (Phase 2–5 có plan riêng khi qua cổng trước)

| Phase | Nội dung | Máy | Thời gian ước | Cổng ra |
|---|---|---|---|---|
| P0 | Task 1: nhãn 3 lớp, probe SB/GI | CPU remote | 0.5 ngày | SB cần v1∧v2? |
| P1 | Task 2: thu lại BS, probe | rtx3060 | 1 ngày | probe shutter-down ≤ 0.60; v1⊕v2 ≥ 0.85 |
| P0′ | SealedBin cue mạnh hơn (rung nắp rõ/2 camera) + thu lại SB | rtx3060 | 2–3 ngày | probe full-label ≥ 0.85, hoặc SB chỉ dùng cho ω/VOI |
| P2 | Task 3–5: ledger, heads, gates; Stage A | laptop CPU + rtx3060 | 1–1.5 tuần | G1 ≥ 0.80, G4 ≥ +0.05, ECE ≤ 0.10, G2 báo |
| P2b | Mệnh đề §2.6: chứng minh + synthetic + bảng 6 run | CPU | song song P2 | chứng minh đứng, synthetic khớp |
| P3 | VOI planner offline: replay ledger trên demo, mô phỏng quyết định với U(d,z), so oracle-expert | CPU + sim | 1.5–2 tuần | Pareto trội imitated/always/never/AGM+VOI; stratum unobservable chọn probe ≥ 80 %; ablation ω≡1 tăng re-check thừa ≥ 30 % |
| P4 | Dữ liệu full (BS-3/6, SB-3/5, GI-3/5 theo spec §5.1), features | thuê CPU + GPU | 3–4 ngày | đủ episode; probes giữ |
| P5 | Stage B: π0 + K tokens + intent; POL-V, B0, Markov+proprio, frozen-after-v1, AGM-style(cùng nhãn)+VOI, µVLA, CheckVLA+FIFO, Oracle × 3 seed | 2× H100 thuê | ~2 tuần | val flow loss ổn định |
| P6 | Eval 50 ep/task/seed; CMA; xáo token; capacity sweep K; held-out failure | GPU + CPU | 1 tuần | SR unrevealed ≥ +10 pp so baseline mạnh nhất (CI không chứa 0); revealed ±3 pp; xáo token −10 pp |
| P7 | RoboMME-Counting (spec §13) | GPU | 1 tuần | không làm hại (±3 pp) |
| P8 | Robot thật: 1 task × 3 variant × 20 trial | lab | 2–3 tuần | đúng chiều sim (bắt buộc cho T-RO) |
| P9 | `report.py` → `results.tex`; viết lại LaTeX theo §3 | CPU | 1 tuần | không còn `[?]` |

**Đích theo cổng đạt được:** P0–P2 → RA-L (B+D); P0–P6 → RA-L mạnh / T-RO; P0–P8 → T-RO / IJRR; trượt P1 → Plan D.

**Việc cần làm trước khi viết Related Work:** đọc full text BEACON (2609.22730), HAVE (2509.00271), Foresight
(2606.23085), RoboMemArena (2605.10921), Present but Not Remembered (2607.03372), AGM (2608.29537); xác nhận
từng cái đúng như tóm tắt của phản biện Fable 5.1 (chỉ mới kiểm ở mức abstract).
