# POL: thiết kế hệ thống thực nghiệm (design doc)

Ngày: 2026-10-26. Paper đích: [`latex-paper/eem-paper/main.tex`](../../../../latex-paper/eem-paper/main.tex)
("Not Yet Seen: Pending-Outcome Memory for VLA Policies"). Tài liệu này là đặc tả đủ để một người
xây dựng (hoặc session sau) viết code và chạy **toàn bộ** thực nghiệm mà paper cần, trên GPU thuê.

Trạng thái: **bản thiết kế, chưa có code POL.** Code `src/jem/` hiện tại là của EEM (OFT, schema viết tay);
phần tái dùng được ghi rõ ở §11.

---

## 1. Mục đích

Chứng minh, trên simulation, claim chính của paper:

> Khi outcome của hành động bị che lúc xảy ra **và** không lộ ra khi robot quay lại, một memory giữ
> posterior của outcome (pending), biết outcome đó có quan sát được hay không, và cập nhật muộn khi gặp
> lại, giúp π0 sửa lỗi im lặng tốt hơn các memory "lưu cái đã thấy", ở **cùng dung lượng memory**.

Mọi con số trong các bảng `\res{...}` của paper phải sinh ra được từ pipeline này, có provenance.

## 2. Mục tiêu, không-mục-tiêu, phản-mục-tiêu

**Mục tiêu**
- G1. Pipeline đầy đủ: setup → sinh dữ liệu → kiểm benchmark → Stage A → gate → Stage B → eval → báo cáo → điền `results.tex`.
- G2. Mọi variant (POL + 11 baseline + oracle) dùng chung feature, dữ liệu, Stage B, số byte memory.
- G3. Log real-time lên W&B cho mọi stage; chạy được offline (`WANDB_MODE=offline`) rồi sync sau.
- G4. Có điểm dừng sớm rẻ (gate N1/G1/G2) trước khi tốn GPU cho π0.

**Không-mục-tiêu**
- Không train lại PaliGemma (đóng băng). Không LoRA (openpi PyTorch chưa hỗ trợ). Không π0.5/π0-FAST.
- Không làm gate VOI tường minh (paper đã ghi là limitation).
- Không port RMBench/RoboTwin (hai tay).
- Robot thật: chỉ đảm bảo policy server dùng được cho client robot; không viết driver robot.

**Phản-mục-tiêu** (làm được nhưng là thất bại)
- POL thắng vì được thêm nhãn sim hoặc thêm byte memory mà baseline không có.
- Kết quả chỉ trên task mà chính mình chọn các knob sau khi nhìn kết quả (mọi tỉ lệ inject, ngưỡng, K mặc định phải chốt **trước** khi chạy Stage B).
- Dùng `oracle_info`, `working_mask`, ID object của sim làm input cho bất kỳ variant nào ngoài Oracle.

## 3. Ràng buộc đã biết

| Ràng buộc | Hệ quả thiết kế |
|---|---|
| openpi PyTorch: Python ≥3.11, torch 2.7.1, transformers 4.53.2 + patch; không LoRA/mixed precision/FSDP/EMA | Freeze PaliGemma, train action expert (≈300M) full bf16 |
| MIKASA/ManiSkill: Python 3.10, torch 2.2.1, numpy 1.23.5, mani_skill 3.0.0b15 | **Hai venv tách biệt**, giao tiếp qua websocket theo giao thức `openpi-client` |
| Camera MIKASA 128×128, 2 cam (top + wrist), proprio 7D (xyz, rpy, gripper), action 7D ee-delta chuẩn hóa [-1,1] | π0 resize-with-pad lên 224; pad state/action lên 32D |
| Episode dài 700–1500 bước | Không BPTT qua π0; memory tính ngoài policy (Stage A) |
| Máy dev: RTX 3050 4GB, 14GB RAM | Test được env, collector, Stage A, ledger local; π0 chỉ chạy trên server |
| GPU thuê: 1–2 × H100/A100 80GB | Ngân sách ~450–600 GPU-h (§10) |

## 4. Benchmark: U-Suite (đã chốt)

Thiết kế cặp đôi **cùng scene, chỉ khác khả năng quan sát outcome**, để control "lỗi lộ ra" là sạch.

| Task | Gốc | Lỗi inject (train) | Lỗi held-out | Loại | Vai trò |
|---|---|---|---|---|---|
| `BatteriesSort-{3,6}` | MIKASA BatteriesChecker (env đã có: `jem/envs/batteries_sort.py`), cấu hình **Flash + phân loại trì hoãn** (test hết rồi mới phân loại) | spill vật lý vào máng sau hood (ε=0.2), shutter che lamp (0.15), tester đọc sai (0.05) | **dính pin khi nhả** (0.05) | unrevealed | chính |
| `SealedBin-{3,5}` | MIKASA GatherAndRecall-{3,5} (scene sẵn), disc → **thùng có nắp mờ (chỉ hình ảnh) + tay nắm mở nắp**, cube cùng màu xám | mỗi lần thả: văng ngang 0.1 (thấy được); văng ra sau 0.1 (rơi dưới mái che, không camera nào thấy); dấu hiệu nắp rung P=0.9 nếu văng ra sau, 0.1 nếu thả bình thường (mập mờ); tấm chắn che dấu hiệu P=0.2 (không quan sát được). *Sửa 2026-10-26:* để quyết định kiểm thùng phụ thuộc lịch sử | **nhả thất bại lúc tay che camera wrist** (0.05) | unrevealed | chính |
| `GatherInject-{3,5}` | GatherAndRecall-{3,5} nguyên bản (disc hở) | tuột khi mang (0.15), lệch khi đặt (0.1) | dính khi nhả (0.05) | **revealed** | control: POL phải ≈ Markov |

**Vì sao mỗi task unrevealed (sửa sau builder check):**
- BatteriesSort: outcome của event insert là z ∈ {seated-working, seated-broken, spill}. Khi gặp lại (rút pin), proprio cho biết seated hay spill, **nhưng lamp Flash đã tắt** nên tốt hay hỏng không lộ ra lúc gặp lại, cũng không lộ ra lúc phân loại. Variant *Hard* (lamp sáng suốt khi pin còn cắm) **không dùng**, vì nó lộ outcome lúc gặp lại.
- SealedBin: cube văng ra sau thùng không thấy được ở cả cửa sổ nhả lẫn lúc gặp lại; thùng kín nên nhìn thùng không biết bên trong có mấy cube. Hành vi kiểm chứng trong demo: **trước khi bấm xác nhận**, expert trượt nắp để nhìn bên trong (nếu còn event pending) hoặc quét tay ra sau thùng. Như vậy outcome treo qua nhiều chunk, và hành vi verify có mặt trong dữ liệu train.
- Test revealed và baseline Markov dùng **ảnh + proprio** (độ mở gripper) lúc gặp lại, để công bằng với POL (v2 của POL có chuỗi w_t).
- Ngân sách bước: P95 độ dài episode của expert × 1.15, đo trên 200 episode pilot, **chốt trước Stage B**. Logic gốc của GatherAndRecall (`cubes_on_disc` sticky, kích flash, điều kiện bấm nút) phải viết lại cho cả SealedBin và GatherInject, vì lỗi có thể làm cube rời đích.
- Nắp thùng SealedBin là actor kinematic trượt khi gripper đẩy tay nắm quá ngưỡng (như cơ chế button). Hood và tấm chắn sau thùng là occluder hình ảnh (không va chạm), giống hood hiện có. Pin spill rơi vào máng mở phía sau hood; lấy lại bằng `try_pick_from_pose` như collector hiện có.
| `RoboMME-Counting` (BinFill, PickXTimes) | RoboMME gốc | không inject | — | revealed | control "không làm hại"; P2 (tùy chọn) |
| `BatteriesChecker-3` (MIKASA gốc) | MIKASA | không | — | — | negative control (outcome không đổi hành động) |

Ghi chú:
- GatherAndRecall có thêm phần "lamp flash → bấm nút màu". **Giữ nguyên** phần đó ở cả SealedBin và GatherInject (cùng nhiễu memory cho mọi variant), success = điều kiện gốc ∧ mọi khối ở đích.
- Inject **vật lý**: giảm `friction`/lực kẹp (tuột), nhiễu pose nhả + xung vận tốc nhỏ (lệch/văng), tăng ma sát ngón (dính). Không teleport. Riêng spill của BatteriesSort hiện là đặt lại pose vào catch tray (code cũ) → **sửa thành vật lý**: lệch xy khi nhả để pin trượt vào máng dẫn xuống tray.
- Tỉ lệ inject ghi trong `configs/pol/tasks/*.yaml`, chốt trước Stage B, không đổi sau khi nhìn kết quả.

### 4.1 Kiểm tra "revealed/unrevealed" (Definition 1 của paper)
Script `check_revealed.py`: với mỗi loại lỗi, lấy frame tại re-encounter (gripper quay lại trong bán kính ρ của event), train logistic regression trên SigLIP pooled feature (chỉ frame hiện tại) dự đoán "event này thất bại" với nhãn sim; AUROC trên val.
- AUROC ≥ 0.9 → revealed; < 0.9 → unrevealed. Kết quả ghi `runs/revealed/*.json` và vào bảng phụ lục.
- Nếu một loại lỗi "unrevealed" theo thiết kế nhưng đo ra revealed → sửa scene (nắp, góc camera) trước khi đi tiếp.

### 4.2 Shortcut probes (giữ từ EEM)
Linear probe trên frame đơn: outcome ẩn (pin tốt/hỏng, khối trong/ngoài thùng) phải ≤ 0.6 AUC khi chưa có evidence. `probe.max_auc` trong paper.

## 5. Dữ liệu

### 5.1 Sinh demo (sim venv)
- Planner: MIKASA motion-planning primitives (đã sinh `jem/envs/mikasa_primitives.py`) + planner GatherAndRecall (`motion_planning_gather_and_recall.py`, cần sinh tương tự).
- **Belief-expert** cho mỗi task: hành động theo posterior tính từ evidence **quan sát được** (đã có cho BatteriesSort ở `jem/envs/expert.py`; viết mới cho SealedBin: thấy khối văng trong cửa sổ nhả → nhặt lại ngay; không thấy (bị che) → posterior theo ε → khi gripper đi ngang sau thùng/nhấc nắp thì kiểm; GatherInject: thấy là sửa).
- Số episode (seed train 0…N−1; val = 10% cuối theo seed):

| Task | Episode | Ước lượng bước/ep | Ghi chú |
|---|---:|---:|---|
| BatteriesSort-3 | 1000 | ~900 | |
| BatteriesSort-6 | 500 | ~1800 | dùng cho ngoại suy N (train N=3 → test N=6 ở bảng Gen) — **không đưa vào train của bảng Gen** |
| SealedBin-3 | 500 | ~450 | |
| SealedBin-5 | 1000 | ~700 | |
| GatherInject-3/5 | 500 / 500 | ~450 / ~700 | |

- Định dạng: mỗi episode một `.npz` (schema `jem.data.Episode`: `rgb [T,128,128,6] uint8`, `proprio [T,7]`, `action [T,7]`, `timestamp`, `language`) + `.log.json` (log evaluator-only: event, outcome thật, loại lỗi, stratum). Log **không bao giờ** vào policy hay Stage A train của POL (chỉ dùng cho: Oracle, Markov/AGM-style head có nhãn, metric, NMI).
- Song song: `collect.py --workers 10` (mỗi worker một process sim, một seed range). Ước lượng 3–5 ep/phút/worker.
- Dung lượng (đã sửa): tổng ~3,3M frame; RGB 98 KB/frame thô, nén npz còn ~30 KB → **~100 GB RGB**. Server cần ≥ **400 GB** đĩa trống (RGB + feature + checkpoint).

### 5.2 Tiền xử lý (policy venv)
1. `precompute_features.py`: SigLIP So400m/14 của π0 (lấy từ checkpoint π0 đã convert, đảm bảo đúng encoder) trên **mọi frame** của cả 2 cam, ảnh 128→224 resize-with-pad.
   - Lưới 16×16 patch → avg-pool **4×4** → 16 token/cam × 2 cam = 32 token × 1152.
   - PCA → 256 chiều/token, fit trên 50k frame train (seed cố định), lưu `pca.npz`.
   - Lưu `pooled [T,2,1152]` fp16 cho **mọi** frame (4,6 KB/frame, ~15 GB), dùng cho context và các baseline đọc mọi bước (KEM, AURA, CheckVLA).
   - Lưu token PCA `[32,256]` fp16 **chỉ** ở frame policy-call (t mod 8 = 0) và trong cửa sổ event (pre/post/re-encounter), tức khoảng 25% frame (~13 GB). Tổng feature ~30 GB.
2. `extract_events.py`: chạy FSM gripper (§6.1) trên proprio → danh sách event `{t, type, x_ee, window_pre, window_post}`; ghép re-encounter ứng viên theo bán kính ρ; lưu `events/<task>.parquet`.
3. Normalizer: mean/std action & state từ **train IDs**, ghi `norm_stats.json` (định dạng openpi).

### 5.3 Tải tài nguyên
- π0 base: `gs://openpi-assets/checkpoints/pi0_base` (JAX) → `examples/convert_jax_model_to_pytorch.py` → `checkpoints/pi0_base_pytorch/` (pin commit openpi `215abfb217dbac7d5f1273282331b9b1866c0479`).
- MIKASA: pin `3d5a21a…` (như cũ); kiểm lại success code BatteriesChecker của bản release so với commit này (việc tồn từ EEM).
- RoboMME (P2): `lerobot/robomme` (HF) + `RoboMME/robomme_policy_learning` (client eval websocket).
- StateMem: `anonymous.4open.science/r/Anonymous0311` (tải tay, ghi hash).
- Script: `scripts/pol/download_assets.sh` ghi hash mọi thứ vào `runs/assets-manifest.json`.

## 6. Phương pháp (POL) — chi tiết cài đặt

### 6.1 Event detector (AGM-style, ghi công AGM)
Gripper width w_t = proprio[6]. FSM trễ (hysteretic), ngưỡng hiệu chỉnh từ histogram dữ liệu train (`calibrate_fsm.py`):
- `OPEN`: w > w_open; `CLOSED_EMPTY`: w < w_empty; `LOADED`: w_empty ≤ w ≤ w_open trong ≥ 3 bước liên tiếp sau lệnh đóng.
- Event: `G+` (vào LOADED), `R+` (LOADED → OPEN), `R0` (CLOSED_EMPTY → OPEN).
- Cửa sổ: `post = [t, t+16)` frame, `pre = [t−8, t)`. Key `κ = (x_ee, type)`.

### 6.2 Outcome model (Stage A)
Đầu vào mỗi event j: `c_j` = pooled feature trung bình trên cửa sổ pre (2×1152 → Linear → 512); `a_j` = action trong `[t−8, t+8)` (16×7 → MLP → 128); `v1_j` = token PCA của cửa sổ post (16 frame × 32 token × 256, lấy trung bình theo thời gian 4 khối → 4×32×256); `v2_j` = như v1 tại re-encounter + chuỗi w_t của re-encounter grasp (16 số).

Mô-đun:
- Prior `p(z|c,a)`: MLP 640→256→M.
- Encoder view `f(v)`: Transformer 2 lớp, d=256, 4 head, trên 128 token (4×32) + CLS → 256.
- Hypothesis embedding `u_z = MLP([e_z, onehot(loại event)])`, e_z ∈ R^{M×128} — **không** có context (sửa 2026-10-27: với `u_z = MLP([g(c), e_z])` score nhận ra view của chính event chỉ qua c, z không được dùng, pilot suy biến q đều).
- Phần cảnh của view được khử bằng hồi quy từ pre-window của chính view đó: `m_k(c) = MLP(pooled) ≈ v̄_k` (MSE, `scene`), encoder nhận `v − sg(m_k(c))`. Context vì thế vẫn "giải thích" phần dự đoán được của view (bố cục, task) nhưng không thể ghép anchor với ứng viên.
- Score `s(v,z) = ⟨u_z, f(v − m(c))⟩ / τ`, τ = 0.1; hai bộ (s1 cho post, s2 cho re-encounter) không chia sẻ trọng số f.
- Head dự đoán `ĝ(z,c,a)` → 32×256 (trung bình theo thời gian của post), dùng cho ω.

Huấn luyện EM (offline, chỉ trên feature):
- Posterior (inference, ledger): `q(z_j) ∝ p(z|c,a)·exp(s1(v1,z)+s2(v2,z))` (liệt kê M). Event không có re-encounter: chỉ dùng s1.
- E-step target khi train: `q_v(z_j) ∝ exp(s1+s2)` — **chỉ từ view**; prior được chưng cất từ q_v. (Nếu đưa prior vào target, prior thấy một nhiễu cân bằng trong context — ví dụ vùng outcome có bị che không — và khóa cụm theo nhiễu đó.)
- M-step loss = `−Σ_z q_v(z)[log p(z|c,a)] + InfoNCE_1 + InfoNCE_2 + λ_g (Σ_z q(z)‖ĝ(z)−v̄1‖² + scene) + λ_u KL(q̄‖Uniform) − λ_i I(z1; z2) − λ_i1 I(z1a; z1b)`,
  trong đó InfoNCE_k lấy trọng số q_v, và `I(z1; z2)` là mutual information kiểu IIC (Ji et al. 2019) giữa `softmax(s1)` và `softmax(s2)` trên các event có re-encounter (hai view của cùng một outcome). Lý do: với u_z chung cho mọi event và q đều, InfoNCE đạt cực đại khi score phẳng (Jensen), nên một mình nó không chọn được phân hoạch; sự đồng thuận giữa hai view thì chọn được.
  *Sửa 2026-10-28:* `I(z1; z2)` chỉ thưởng cho cái mà cả hai view cùng thấy, mà lỗi im lặng thì theo thiết kế không có ở v2 (lamp BatteriesSort đã tắt lúc gặp lại; round 1: POL 0.56 trong khi probe tuyến tính trên v1 đạt 0.93). Thêm `I(z1a; z1b)`: IIC giữa hai augmentation ngẫu nhiên của chính v1 (bỏ ngẫu nhiên khối thời gian và token, nhiễu feature).
  InfoNCE: dương = (v của chính event, z), âm = v của 63 event khác trong batch, trong đó 32 là **hard negative** cùng loại event, context gần nhất (kNN trên g(c), cập nhật mỗi 1k bước).
- M ∈ {2,4,6,8}, chọn theo **val InfoNCE + log-likelihood** (không dùng nhãn). Mặc định M=6.
- λ_g=1.0, λ_u=0.1, λ_i=1.0, λ_i1=1.0. AdamW lr 3e-4, wd 0.01, batch 512 event, 20k bước, warmup 1k + cosine. ~1–2 GPU-h (chạy được trên 3050 với batch 128).

### 6.3 Observability và trạng thái
- `ω_j = Σ_p Σ_z p(z|c,a)‖ĝ_p(z)−ḡ_p‖²/σ_p²` (σ_p² = variance patch trên train).
- `H_j = H(softmax ℓ_j)`. Resolved nếu `H_j < η` (η = 0.3 nat); pending-unobservable nếu `H_j ≥ η ∧ ω_j < ω0`; pending-ambiguous còn lại. ω0 = phân vị 20% của ω trên val.

### 6.4 Ledger
- K slot (mặc định **8**; sweep {4,8,16,32,64}; 64 lớn hơn số event tối đa của mọi episode nên được báo là "∞"). Entry 50 số: `PE(x_ee)` 12, type one-hot 3, `softmax ℓ` M(≤8, pad), ω 1, H 1, age 1, summary của c 24 (paper phải ghi 24, không phải 32).
- Ghi: mọi event. Đầy → evict entry có H nhỏ nhất (đã resolved), hòa thì cũ nhất.
- **Liên kết re-encounter (sửa theo phản biện):** điểm liên kết mềm giữa event mới r và entry j:
  `π(j|r) ∝ exp(−‖x_r−x_j‖²/2ρ² + β·cos(h(c_j),h(c_r)))`, ρ = 6 cm, h = projection 64-d học trong Stage A (contrastive giữa pre-window của event và re-encounter đã ghép theo không gian, có lọc). Cập nhật: `ℓ_j ← ℓ_j + π(j|r)·s2(v2_r, ·)`. β chọn trên val. Ablation: β=0 (chỉ không gian).
- Ledger chạy được trong numpy (sim/eval) và torch (Stage B precompute), cùng một code.

### 6.5 π0 + ledger (Stage B)
- `PI0WithMemory(PI0Pytorch)`: override `embed_suffix(state, x_t, t, memory)` để chèn **K token memory trước state token**, `att_mask = [1, 0, …, 0]` (khối riêng; token memory attend prefix + nhau; state và action attend memory). Phải override **cả** `forward(observation, actions, memory)`, `sample_actions(..., memory)` và `denoise_step(..., memory)` vì cả ba gọi `embed_suffix`; memory đi theo tham số riêng, không qua `Observation`. Không dùng `scripts/train_pytorch.py` của openpi (không có freeze/param group); viết vòng lặp riêng, chỉ tái dùng model, preprocessing và loader trọng số.
- Precision: model bf16 như openpi, nhưng tham số **train được** có bản master fp32 (copy grad bf16 → fp32, AdamW trên fp32, copy lại bf16), tránh update bị làm tròn mất ở lr 5e-5. Thêm ~3,6 GB VRAM.
- Token: `MLP(50 → 1024 → 1024)` (width của gemma_300m), + learned type embedding cho "slot trống".
- Freeze: toàn bộ `paligemma` (SigLIP + Gemma 2B). Train: `gemma_expert`, `action_in_proj`, `action_out_proj`, `state_proj`, `action_time_mlp_*`, token MLP.
- Hyperparams (mặc định, dùng chung **mọi** variant):

| Tham số | Giá trị |
|---|---|
| action_horizon H / thực thi K_exec | 16 / 8 |
| ảnh | base = top, left_wrist = wrist, right_wrist = mask 0 |
| precision | bf16 toàn phần (openpi PyTorch) |
| optimizer | AdamW β=(0.9,0.95), eps 1e-8, wd 1e-10, clip 1.0 |
| lr | expert 5e-5; token MLP 1e-4; warmup 1000; cosine về 10% |
| batch | 64 (grad checkpointing bật) |
| bước | 20k (≈ 3 epoch trên ~410k call-sample) |
| tập train | **một model đa-task cho mỗi (variant, seed)**: BS3, BS6, SB3, SB5, GI3, GI5; normalizer chung. Bảng Gen-objects dùng run riêng chỉ train trên N=3 (BS3, SB3, GI3) |
| sample | mỗi policy call (t mod 8 = 0) của demo; target = action[t:t+16] |
| memory tokens | precompute offline cho mọi call (ledger replay theo thời gian, chỉ dùng frame ≤ t) |
| eval checkpoint | checkpoint có val flow-loss thấp nhất trong {10k,15k,20k} (chốt trước) |

- Ước lượng: ~0.6 s/bước trên H100 → ~3,5 h/run.

### 6.6 Inference / policy server
- `serve_policy.py` (policy venv): websocket theo `openpi_client`; trạng thái memory theo `episode_id`.
- Message client → server mỗi call: `{images: top, wrist (128×128×3), state 7D, prompt, chunk_frames [≤8,2,128,128,3], chunk_proprio [≤8,7], reset}`.
- Server: SigLIP trên chunk_frames (batched) → PCA → FSM/ledger update → token → π0 sample 10 bước flow → trả 16 action (client thực thi 8).
- Đo latency/VRAM mỗi call, log W&B.

## 7. Baselines (cùng feature, cùng Stage B, cùng byte memory = K×50 số)

Mọi baseline là một `MemoryProducer` với interface `reset() / observe(t, feats, pooled, proprio, events) / tokens() -> [K,50]`. Phần học được của baseline train ở **Stage A** với cùng aux self-supervised (InfoNCE dự đoán feature tương lai), rồi đóng băng.

| Variant | Lưu gì | Ghi khi nào | Nhãn sim? |
|---|---|---|---|
| B0 | không (K token rỗng) | — | không |
| Frames | K frame gần nhất (pooled→PCA 50) | mỗi call | không |
| Random-write | K frame | ngẫu nhiên p=0.1 | không |
| Markov | 1 token: P(fail) từ classifier frame hiện tại (§4.1) | mỗi call | **có** (classifier) |
| AURA-gate | K frame | khi surprise của action predictor (trên feature) > ngưỡng | không |
| StateMem-style | 1 token cập nhật residual low-rank theo prediction error (port quy tắc từ code gốc) | mỗi bước | không |
| CheckVLA-style | K keyframe + 1 token risk | dự đoán feature k=8 bước có điều kiện action; discrepancy chuẩn hóa; ngưỡng conformal α=0.1 trên episode thành công | không |
| CheckVLA+FIFO | như trên + các chunk bị gắn cờ "che" (ω thấp theo predictor điểm) | | không |
| AGM-style | con trỏ subgoal + số đã xác minh | tại event FSM; verify head 2-lớp trên feature post-release | **có** (head verify, như AGM) |
| KEM | K keyframe | scorer học (BCE với nhãn "frame có thay đổi lớn" tự giám sát) | không |
| µVLA-style | K token recurrent (GRU trên pooled feature mỗi call) | mỗi call | không |
| **POL** | ledger K entry | mọi event | **không** |
| Oracle | outcome thật mỗi event | mọi event | có (upper bound) |

## 8. Đánh giá

### 8.1 Giao thức
- 50 episode/task, seed eval `4242424242 + i` (stream cố định, tách khỏi train/val).
- 3 training seed cho P0 variant (§10); 1 seed cho sweep.
- Paired bootstrap (train-seed × eval-seed cluster), 2000 lần; ngưỡng tương đương ±3 pp chốt trước. Tái dùng `jem.evaluation.paired_report` + `check_provenance` (mở rộng field: `feature_hash`, `stageA_hash`, `memory_bytes`).

### 8.2 Metrics (map sang key của paper)

| Metric | Định nghĩa | Key |
|---|---|---|
| Success rate | `success_once` của evaluator | `sr.<V>.<task>`, `sr.<V>.unrev`, `sr.<V>.rev` |
| Recovery rate | % lỗi inject được sửa trước khi hết episode | phụ lục |
| Time-to-recovery | bước từ lỗi → sửa xong | phụ lục |
| Re-check/object | số hành động kiểm lại / object; **unnecessary re-check** = kiểm lại khi outcome đã xác định được | `abl.*.tests` |
| Silent-failure AUROC | P(fail) suy từ memory tại mỗi event vs nhãn sim | `diag.*.auroc` |
| Observability AUROC | ω tách event "vùng outcome bị che" vs "thấy" trên **che tĩnh** | `diag.*.obs` |
| NMI | z học được vs outcome sim | `diag.*.nmi` |
| Calibration | ECE/Brier của P(outcome) theo event | phụ lục |
| CMA correct-switch | cặp history khác outcome, cùng hiện tại; % policy đổi đúng | `cma.*` |
| Compute | VRAM, latency/call, byte memory | phụ lục |
| Strata | SR theo loại lỗi: spill, behind-bin, occluded-lamp, slip-in-view, held-out | phụ lục |

### 8.3 Thí nghiệm ↔ bảng paper
| Bảng/Hình | Chạy gì |
|---|---|
| Table I (main) | 12 variant × {BS3, BS6, SB5, GatherInject, RoboMME-Cnt}; 3 seed cho P0 |
| Fig capacity | POL, CheckVLA+FIFO, KEM, µVLA, StateMem × K∈{4,8,16,32,64}; 1 seed; unrevealed set |
| Table II (diag) | Stage A only: POL, fake-action, variance-ω, M=1, CheckVLA score |
| Table III (ablation) | POL: no-resolve, ω≡1, M=1, call-frames-only, FIFO eviction, β=0 (thêm) |
| Table IV (gen) | held-out failure; train N=3 → test N=6; 25% demo |
| CMA | sinh cặp history bằng sim (cùng seed, đổi một outcome inject) |

## 9. Log W&B

- Project `pol`, entity từ env `WANDB_ENTITY`. Group = `stageA/<task-set>`, `stageB/<variant>`, `eval/<variant>`. Run name `{stage}-{variant}-K{K}-s{seed}`.
- `wandb.config` = toàn bộ YAML đã resolve + provenance (code hash, data hash, feature hash, openpi commit, MIKASA commit).
- Stage A (mỗi 20 bước): loss từng thành phần, entropy q(z), histogram sử dụng z, lr, grad-norm; mỗi 1k bước: val InfoNCE, (chỉ đánh giá) NMI và AUROC ω.
- Stage B (mỗi 20 bước): flow loss, grad-norm, lr, step time, samples/s, VRAM peak; mỗi 2k bước: val flow loss; ảnh camera đầu tiên.
- Eval: `wandb.Table` một dòng/episode (task, seed, success, stratum, số re-check, latency); video 3 episode đầu/task (`wandb.Video`, 10 fps); summary SR và CI.
- Artifact: chỉ trọng số train được (action expert + token MLP, ~700 MB) và Stage A (~50 MB).

## 10. Ngân sách và thứ tự chạy

| Pha | Việc | Máy | Ước lượng | Gate đi tiếp |
|---|---|---|---|---|
| 0 | Viết code + test CPU/local 3050 (env, collector, FSM, ledger, Stage A nhỏ) | local | — | test xanh |
| 1 | Setup 2 venv, convert π0, smoke server+client 1 episode | GPU | 2 h | 1 episode chạy hết |
| 2 | Sinh demo, check_revealed, probes | CPU nhiều nhân | 1–2 ngày | revealed/unrevealed đúng thiết kế; probe ≤ 0.6 |
| 3 | Stage A + **gate N1/G1/G2** | 1 GPU | 10 h | xem §10.1 |
| 4 | Stage B P0: B0, Markov, CheckVLA+FIFO, AGM-style, KEM, µVLA, POL × 3 seed (21 run) | 1–2 GPU | ~105 GPU-h | — |
| 5 | Eval P0 | GPU + CPU | ~40 h | H1 |
| 6 | P1: Frames, Random, AURA, StateMem, CheckVLA, Oracle × 3 seed (18 run) + eval | | ~120 GPU-h | — |
| 7 | Sweep capacity (20 run), ablation (6), gen (6) + eval | | ~200 GPU-h | — |
| 8 | (P2) RoboMME-Counting | | ~40 GPU-h | — |
| 9 | `report.py` → `results.tex`, hình | CPU | — | không còn `[?]` ngoài real robot |

Tổng ≈ 450–550 GPU-h. Fail gate ở pha 3 → dừng, chuyển Plan B (paper benchmark) mà không tốn pha 4–8.

### 10.1 Gate ở pha 3 (đo trước khi có Stage B, chỉ trên feature + nhãn sim để đánh giá)
Đơn vị là **event tại thời điểm quyết định** (lúc phân loại pin / lúc bấm xác nhận SealedBin / lúc đặt GatherInject).
- **Nhãn của gate** (`labels.GATE_TARGET`, *sửa 2026-10-28*): outcome mà quyết định phụ thuộc và bị ẩn lúc quyết định (§4). BatteriesSort: **pin hỏng vs tốt** (spill lộ qua proprio lúc gặp lại nên bỏ ra); SealedBin: **văng ra sau thùng vs trong thùng** (văng ngang thấy được nên bỏ ra); GatherInject (revealed): mọi lỗi vs trong đích. Bản trước dùng "thất bại" (spill, mọi cube ngoài đích), toàn lỗi nhìn thấy được, nên Markov đạt 0.96 ở round 1.
- **N1 (cần memory):** AUROC phát hiện nhãn gate của (a) classifier Markov trên ảnh + proprio hiện tại (train trên cùng nhãn), (b) posterior POL. Qua nếu trên tập unrevealed `AUROC_POL − AUROC_Markov ≥ 0.10` **và** `AUROC_Markov ≤ 0.75`; trên tập revealed `|AUROC_POL − AUROC_Markov| ≤ 0.03`. *Sửa 2026-10-28:* AUROC của một tập = **trung bình AUROC từng task** trong tập (`per_task`), P(nhãn | z) hiệu chỉnh riêng từng task trên event train có gặp lại; gộp event của nhiều task thưởng cho tỉ lệ nền (round 1: xáo evidence vẫn đạt 0.71 khi gộp), nên bản gộp chỉ để tham khảo (`pooled`).
- **G1 (phát hiện lỗi im lặng):** AUROC_POL ≥ 0.80 trên unrevealed; cùng model khi **xáo evidence** (v1, v2 lấy từ event khác cùng task và loại, giữ context và action) ≤ 0.60. *Sửa 2026-10-28:* bản trước dùng action giả, nhưng các lỗi được inject không phụ thuộc action nên một POL đúng cũng không đổi theo action; action giả vẫn được báo (`pol_fake_actions`) nhưng không gate.
- **G2 (observability):** AUROC của −ω tách event "vùng outcome bị che" (shutter hạ / văng ra sau thùng / wrist bị che) khỏi event "thấy được", chỉ trên các event có che **tĩnh**: ≥ 0.80; bản dùng variance thay ω phải thấp hơn rõ.
- Fail N1 hoặc G1 → dừng (Plan B). Fail G2 → bỏ ω khỏi claim, chạy tiếp.

## 11. Bố cục code

```
code/joint-execution-memory/
  src/pol/
    config.py          # dataclass + YAML loader, provenance hash
    events.py          # FSM gripper, cửa sổ, key, ghép re-encounter
    features.py        # SigLIP extract, PCA, shard IO
    outcome_model.py   # prior, encoder view, scores, ĝ, EM step
    ledger.py          # Ledger (numpy/torch), write/evict/associate/resolve, featurize
    memories/          # MemoryProducer + 12 baseline (một file/variant)
    pi0_memory.py      # PI0WithMemory, load/convert, freeze
    data.py            # dataset Stage A (event tuple) & Stage B (call sample), normalizer
    train_stage_a.py   # vòng lặp + W&B
    train_stage_b.py   # vòng lặp + W&B, resume, val, checkpoint
    serve.py           # websocket policy server
    eval_client.py     # chạy trong sim venv: env + injection + logging episode
    metrics.py         # SR, recovery, AUROC, NMI, ECE, CMA
    report.py          # gom kết quả → results.tex keys, capacity.pdf
  src/jem/envs/
    batteries_sort.py  # (có) + spill vật lý + held-out sticking
    sealed_bin.py      # MỚI, kế thừa GatherAndRecall
    gather_inject.py   # MỚI (revealed control)
    injection.py       # MỚI: wrapper inject vật lý dùng chung
    expert.py          # (có) + expert SealedBin/GatherInject
  scripts/pol/         # setup_sim_env.sh, setup_policy_env.sh, download_assets.sh, collect.py,
                       # check_revealed.py, probe.py, precompute_features.py, extract_events.py,
                       # run_stage_a.sh, run_stage_b.sh, run_eval.sh, run_all.sh
  configs/pol/         # tasks/*.yaml, stage_a.yaml, stage_b.yaml, variants/*.yaml, eval.yaml
  tests/pol/           # test CPU: FSM, ledger, association, EM trên dữ liệu giả, mask attention π0 (fake model)
```

Tái dùng từ `jem`: `envs/batteries_sort.py`, `envs/expert.py`, `envs/mikasa_primitives.py`, `evaluation.py`
(paired bootstrap, provenance), `data.Episode`, chính sách allowlist observation. Bỏ: `memory.py`,
`gate.py`, `backbone.py` (OFT), `policy.py`, `training.py` của EEM (giữ trong repo, không dùng).

## 12. Phản hồi cho các điểm phản biện (ảnh chụp review)

| Điểm | Đánh giá | Xử lý trong thiết kế |
|---|---|---|
| Phụ thuộc event detector của AGM → chỉ prehensile | Đúng; là lựa chọn phạm vi, không phải novelty | Ghi công AGM; phạm vi nêu rõ trong abstract/limitations; detector là module thay được |
| Spatial key tĩnh, heuristic | Đúng, điểm yếu thật | §6.4: liên kết mềm không gian + ngữ cảnh học được; ablation β=0 |
| Verification chỉ là imitation | Đúng | Giữ là limitation; báo re-check thừa; không thêm VOI (cần utility theo schema) |
| Gợi ý 1: dùng c_j khi matching | Đồng ý | §6.4 |
| Gợi ý 2: tô đậm khác biệt với POMDP cổ điển | Đồng ý | Sửa Related Work + đoạn sau Prop. 1: POMDP cho luật cập nhật, POL học **không gian giả thuyết và mô hình quan sát** không nhãn trên latent của VLA pretrained |

## 12b. Sửa paper cho khớp design (làm cùng lúc với code)
- K "∞" → ghi rõ "64 (≥ số event tối đa)"; summary của c = 24 chiều.
- GRU và µVLA-style gộp làm một baseline ("µVLA-style recurrent tokens (GRU)"); Random-write đưa vào phụ lục, không thêm cột.
- Cột "MIKASA-inj" đổi tên thành "GatherInject" = trung bình GI3 và GI5.
- "All cubes look identical" → "objects whose outcome is hidden look identical within a task" (SealedBin dùng cube xám; GatherInject giữ màu gốc).
- BatteriesSort: lamp Flash, phân loại sau khi test hết; test revealed và Markov dùng ảnh + proprio.

## 13. Quyết định của người dùng (đã chốt 2026-10-26)

1. **RoboMME-Counting: làm luôn.**
   - Dữ liệu `lerobot/robomme` (LeRobot v3, 256², 10 fps), chỉ suite Counting (BinFill, PickXTimes, SwingXTimes).
   - Action space khác MIKASA (khớp tuyệt đối 8D), nên dùng **model Stage B riêng** cho RoboMME. Variant: B0, AGM-style, KEM, POL; 3 seed. Normalizer riêng.
   - Stage A của POL chạy lại trên feature RoboMME. FSM dùng kênh gripper của state 8D.
   - Eval: dựng venv thứ ba `.venv-robomme` theo `RoboMME/robomme_policy_learning/examples/robomme`, dùng `eval.py` của họ làm client, nối websocket vào `serve_policy.py` của mình qua một adapter observation.
   - Tham chiếu: AGM báo 100% PickXTimes, 84% BinFill (π0.5).
2. **2× H100 80GB, ≥ 400 GB đĩa.**
   - Stage B chạy **2 run độc lập song song**, mỗi GPU một run, không dùng DDP.
   - Eval: một GPU chạy policy server, sim chạy song song nhiều process CPU.
   - Lịch ước tính khoảng 2 tuần cho pha 4–8.
3. **W&B: mặc định.** Project `pol`; entity lấy từ `WANDB_ENTITY` hoặc tài khoản đang đăng nhập; không ghi cứng trong code.

## 14. Để lại cho người xây dựng (minor, tự quyết)
- Ngưỡng FSM cụ thể (từ `calibrate_fsm.py`).
- Kích thước MLP nhỏ, số head, có dùng `torch.compile` không.
- Định dạng shard (npy vs zarr).
