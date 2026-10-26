# TODO — EEM: việc còn lại để chạy được trên server

Cập nhật: 26-10-2026. File này dành cho **session tiếp theo**: đọc mục 0 là đủ ngữ cảnh để bắt đầu.
Mỗi việc có: file cần sửa, việc cụ thể, **tiêu chí xong**. Ước lượng công sức là thô.

---

## 0. Ngữ cảnh trong 10 dòng

- **Ý tưởng:** Epistemic Execution Memory (EEM). Memory của VLA phải nhớ robot **đã xác minh được gì** (kể cả evidence âm), và chỉ kiểm lại khi đáng. Proposal: [`latex-paper/joint-outcome-proposal/PROPOSAL.md`](../../latex-paper/joint-outcome-proposal/PROPOSAL.md). Paper (RSS): [`latex-paper/eem-paper/`](../../latex-paper/eem-paper/README.md).
- **Code hiện có** (`src/jem/`): memories GRU, MU, B2, B2MLP, KEM, EEM; gate (IL + VOI); policy; training episode-causal; evaluation có kiểm provenance; env BatteriesSort; belief-expert; collector. Kiến trúc: [`docs/architecture.md`](docs/architecture.md). Quy trình server: [`docs/RUNBOOK.md`](docs/RUNBOOK.md).
- **Đã kiểm:** 72 unit test pass trên CPU với tiny fixture backbone; paper build được; `fill_results.py` điền số đúng với dữ liệu giả.
- **Chưa từng chạy:** simulator (env BatteriesSort, collector), mọi bước 7B, `setup_server.sh`. Máy dev không có ManiSkill/GPU.
- **Môi trường test local:** venv CPU tạm nằm trong scratchpad của session cũ và sẽ mất. Tạo lại bằng:
  `python3 -m venv .venv-cpu && .venv-cpu/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu && .venv-cpu/bin/pip install numpy pytest`, rồi chạy
  `JEM_PYTHON=.venv-cpu/bin/python bash scripts/verify_local.sh`.
- **Quyết định đã chốt:** server 1× A100/H100 80GB; venue RSS; bản proposal cũ đã nằm trong Thùng rác.
- **Kết quả duy nhất hiện có** là mô hình POMDP trừu tượng (`scripts/necessity_gate_sorting.py --v2-only`); không có số liệu model.

---

## 1. Vì sao còn thiếu nhiều

1. **Máy dev không có simulator và GPU.** Mọi thứ phụ thuộc ManiSkill hoặc 7B chỉ viết được, không chạy được. Đó là lý do env, collector và setup vẫn "chưa test".
2. **Các lượt trước ưu tiên phần logic có thể kiểm trên CPU** (memory, gate, causality, provenance). Hạ tầng training thực tế (đọc dữ liệu kiểu streaming, resume, validation, logging) chưa được làm, và lẽ ra phải nói rõ ngay khi báo "dọn code xong".
3. **Hai yêu cầu mới thay đổi interface:** dùng π0/π0.5 thay OFT (head flow matching thay cho L1), và dùng dataset có sẵn (ảnh 256², action 8D/16D thay vì 128² và 7D). Code hiện hard-code `128×128×6`, proprio 7D, action 7D ở khoảng 20 chỗ.

---

## 2. Quyết định cần người dùng chốt trước (chặn WP2, WP4)

| # | Câu hỏi | Khuyến nghị | Ảnh hưởng |
|---|---|---|---|
| D1 | Backbone: OpenVLA-OFT (7B), π0 hay **π0.5**? | **π0.5**: baseline tham chiếu của MIKASA-Robo-VLA; RoboMME và MEM xây trên π0.5; RoboMME có sẵn code MME-VLA trên π0.5. Nhỏ hơn OFT (khoảng 3B) | WP2. Cần kiểm openpi có bản PyTorch dùng được, vì code `jem` là PyTorch |
| D2 | Dataset theo pha nào? | Pha A: MIKASA-Robo-VLA → Pha B: RoboMME → Pha C: BatteriesSort → Pha D (tùy chọn): RMBench | Mục 3, WP1, WP4 |
| D3 | Có cho phép train memory **không có nhãn auxiliary** trên dataset ngoài không? | Có, nhưng **mọi variant cùng tắt aux** và ghi `aux=none` vào provenance | WP3.6 |
| D4 | Ngân sách GPU | Lộ trình tối thiểu trước: 3 variant (B0, KEM, EEM) × 1 seed | Mục 5 |

---

## 3. Dataset có sẵn: dùng được vào việc gì

Gate (định nghĩa ở PROPOSAL §3): **G-a** = kết quả hành động của chính robot làm đổi hành động sau. Đây là điều kiện để test **claim chính** của EEM.

| Dataset | Sim / robot | Ảnh | Action | Dữ liệu | Đạt G-a? | Dùng vào |
|---|---|---|---|---|---|---|
| **MIKASA-Robo-VLA** (90 task) | ManiSkill3, Panda | 128², 2 cam | 7D ee-delta chuẩn hóa | LeRobot v3 + RLDS, 250 demo/task, MIT. Đã ghim revision và audit 4 task (`docs/data-audit.md`) | Không (đã rà 26 family) | **Pha A**: kiểm toàn bộ pipeline, kết quả "EEM không làm hại trên memory task chuẩn". BatteriesChecker là negative control. **Tương thích code nhất** |
| **RoboMME** (16 task) | ManiSkill3, Panda | 256², front + wrist | 8D khớp tuyệt đối (hoặc 7D EEF tuyệt đối) | `lerobot/robomme`, LeRobot v3, khoảng 768K frame, 10 fps, 100 demo/task, có nhãn subgoal ngôn ngữ | Một phần: suite **Counting** (BinFill, PickXtimes, SwingXtimes) đếm số lần **đã hoàn thành**, nên phân biệt "đã thử" và "đã xong" khi pick hụt. Các suite khác là cue từ môi trường | **Pha B**: external validation; Counting là phép thử phụ cho "đếm cái đã xác minh". Cần fact dạng **bộ đếm**, không phải L×Q (WP2.5) |
| RMBench (9 task) | RoboTwin 2.0, **2 tay** | khác | **16D** bimanual | 50 demo/task, code `robotwin-Platform/rmbench` | **Có**: Battery Try (thử đến khi đúng) | Pha D tùy chọn. Port nặng (2 tay, sim khác) |
| RoboTwin-MeM (EventVLA) | RoboTwin 2.0, 2 tay | khác | bimanual | Code + data ở `InternRobotics/EventVLA` | Có ở Find Seal (tìm thấy thì dừng); evidence luôn quan sát được | Tùy chọn: control "evidence quan sát được" |
| LIBERO-Mem | LIBERO | — | — | Chưa kiểm phát hành | Không | Bỏ qua |
| **BatteriesSort** (tự xây) | ManiSkill3, Panda (kế thừa MIKASA) | 128², 2 cam | 7D ee-delta | Tự sinh bằng collector | **Có, cả G-a/G-b/G-c** | **Pha C: claim chính** |

**Kết luận:** chưa có dataset sẵn nào test được lõi của EEM (evidence âm về kết quả hành động của chính robot). Dataset sẵn có giúp **giảm rủi ro hạ tầng** (Pha A) và cho **external validity** (Pha B), nhưng **không thay được BatteriesSort**.

---

## 4. Work packages

Ký hiệu: P0 = phải xong trước khi thuê GPU dài ngày; P1 = cần cho kết quả chính; P2 = paper.

### WP1 — Tổng quát hóa tầng dữ liệu (P0, khoảng 2–3 ngày, làm trên CPU)

| # | Việc | File | Tiêu chí xong |
|---|---|---|---|
| 1.1 | Thay hard-code `128×128×6`, proprio 7, action 7 bằng `DataSpec(image_size, cameras, proprio_dim, action_dim, action_kind)` lấy từ config | `data.py`, `policy.py`, `environment.py`, `fixtures.py` | Test chạy với hai spec: (128², 7D) và (256², 8D) |
| 1.2 | **Đọc dữ liệu theo kiểu streaming.** `cli.load_dataset` đang nạp toàn bộ episode vào RAM: 1000 episode × khoảng 1000 frame × 98 KB ≈ **98 GB** với 128²; RoboMME 256² còn lớn hơn | `cli.py`, `data.py` | `EpisodeIndex` load lười từng episode (memmap/npz); RSS ổn định khi lặp qua 1000 episode |
| 1.3 | Converter **LeRobot v3 → Episode** cho toàn dataset (hiện `extract_real_samples.py` chỉ giải mã episode 0 theo byte-range) | `scripts/convert_lerobot.py` (mới) | Chuyển đủ 1 task MIKASA và 1 task RoboMME; kiểm số frame và timestamp khớp metadata; có test với một parquet nhỏ |
| 1.4 | Fit normalizer theo streaming, chỉ trên train IDs | `data.py` (`TrainNormalizer.fit_stream`) | Kết quả bằng hàm `fit` hiện tại trên dữ liệu nhỏ |
| 1.5 | Action tuyệt đối (RoboMME): normalizer + quy ước gripper riêng; không ép về ee-delta | `data.py`, config dataset | Round-trip transform/inverse trên dữ liệu RoboMME |
| 1.6 | Config cho từng dataset: `configs/datasets/{mikasa,robomme,batteriessort}.json` (spec, task list, split, revision) | `configs/` | `cli train --dataset` đọc được |

### WP2 — Tách backbone (P0, khoảng 3–4 ngày, phần π0 cần GPU để kiểm cuối)

| # | Việc | File | Tiêu chí xong |
|---|---|---|---|
| 2.1 | **Chuyển action loss vào backbone:** `backbone.action_loss(encoded, memory_tokens, target, mask)` và `backbone.sample(encoded, memory_tokens)`. `training.py` hiện hard-code L1 lên output của `decode` | `backbone.py`, `training.py`, `fixtures.py` | OFT giữ L1; TinyBackbone có cả hai chế độ L1 và flow matching để test |
| 2.2 | Bỏ hard-code 7D trong `L1ActionHead` và constants OFT | `backbone.py` | OFT chạy được với action_dim từ DataSpec |
| 2.3 | **Adapter π0/π0.5** (theo D1): memory tokens nối vào prefix của PaliGemma; action expert attend tới đó; loss flow matching; sampling khoảng 10 bước; LoRA. Cố định chunk K giống nhau cho mọi variant (gợi ý K=8–10 cho 10 Hz) | `backbone_pi0.py` (mới) | Smoke forward/backward trên GPU; ghi VRAM và latency; test với fake model giống `test_oft.py` |
| 2.4 | Tổng quát hóa loader: `load_backbone(kind, ...)` thay cho `load_local_oft` ở `cli.py`, `scripts/oft_smoke.py`, `scripts/evaluate_oft.py`; đổi tên script thành `smoke.py`, `evaluate.py` | 3 file trên | Một cờ `--backbone {oft,pi0,pi05}` |
| 2.5 | Fact dạng **bộ đếm** cho RoboMME Counting (belief trên {0..Nmax} số lần đã hoàn thành, evidence = hoàn thành quan sát được) | `memory.py` (schema fact cấu hình được) | Test với chuỗi hoàn thành/thất bại giả |

### WP3 — Hạ tầng training (P0, khoảng 2–3 ngày, CPU)

| # | Việc | File | Tiêu chí xong |
|---|---|---|---|
| 3.1 | **Resume** sau khi server ngắt: lưu adapter + optimizer + RNG + vị trí epoch/episode; `--resume` | `cli.py`, `checkpointing.py` | Test: ngắt giữa chừng rồi resume cho đúng loss sequence |
| 3.2 | LR schedule (warmup + cosine), gradient accumulation qua nhiều episode | `cli.py`, config | Ghi trong provenance `tuning_budget` |
| 3.3 | **Validation và chọn checkpoint chốt trước** (proposal yêu cầu; code hiện lấy epoch cuối) | `cli.py` | val loss mỗi N episode; chọn theo val, ghi lý do vào `training.json` |
| 3.4 | Logging (TensorBoard, tùy chọn W&B) | `cli.py` | Có loss/aux/thời gian/VRAM |
| 3.5 | Determinism: seed mọi RNG, cờ cuDNN | `cli.py` | Hai lần chạy CPU cùng seed cho cùng loss |
| 3.6 | Chế độ **`aux=none`** cho dataset ngoài, áp dụng cho mọi variant; gate G3 chỉ áp cho BatteriesSort | `training.py`, `cli.py` | Provenance ghi `aux`; aggregator từ chối so sánh lệch `aux` |
| 3.7 | Nối các tham số memory từ config: `kem_capacity`, `kem_commit`, `evidence_dim`, `roles`, số token MU (hiện policy dùng giá trị mặc định) | `policy.py` | Thay config thì model thay theo; có test |
| 3.8 | `roles` theo dataset (RoboMME có task tới 15 cube; MIKASA 3–6) | config | — |
| 3.9 | Bug: `label_hash` tính trên tích chéo folder × episode, nên có thể hash nhầm file của task khác cùng số seed | `cli.py` dòng ~161 | Hash đúng tập file đã train; có test |
| 3.10 | `gate_kwargs` của EEM-VOI đang là hằng số (spill 0.2…) và phải lấy từ cấu hình env/task | `configs/variants/EEM-VOI.json`, `evaluate.py` | Mỗi task dùng tham số của chính nó |

### WP4 — Harness đánh giá (P1)

| # | Việc | File | Tiêu chí xong |
|---|---|---|---|
| 4.1 | Tổng quát hóa runner MIKASA (hiện frame bị reshape cứng `128×128×6` ở `environment.py:131`) | `environment.py` | Chạy với DataSpec |
| 4.2 | Adapter cho **evaluator RoboMME** (`RoboMME/robomme_benchmark`), kể cả task "Video*": đưa video lịch sử vào memory trước bước đầu (pre-roll vào luồng evidence) | `envs/robomme_eval.py` (mới) | 1 episode dummy chạy hết; số episode/seed theo protocol RoboMME |
| 4.3 | Ghi `stratum` và `tests_per_battery` cho task ngoài (hiện chỉ BatteriesSort có) | `evaluate.py` | `fill_results.py` xử lý được khi thiếu |
| 4.4 | (Pha D) adapter RoboTwin/RMBench 2 tay, 16D | mới | Tùy chọn |

### WP5 — Baseline còn thiếu so với proposal (P1)

| # | Baseline | Cách làm | Tiêu chí xong |
|---|---|---|---|
| 5.1 | Checklist + stage | Memory deterministic theo role: cờ "đã xong", không lưu kết quả | Variant `CHECKLIST` + test |
| 5.2 | Monitor-only (kiểu Sentinel) | Không history; head phân loại status từ frame hiện tại → re-test | Variant `MONITOR` + test |
| 5.3 | Always-verify, Oracle-belief | Scripted trong BatteriesSort, dùng primitive + belief đúng | Chạy qua collector/evaluator |
| 5.4 | MemoryVLA (E1) | Port từ `third_party/memoryvla` (license chưa xác minh) | Tùy chọn; ghi rõ là system comparison |

### WP6 — Dựng BatteriesSort trên server (P1, Pha C)

Theo `docs/RUNBOOK.md` Stage 1–2:
- `scripts/check_batteries_sort.py` exit 0.
- Chỉnh hình học (bin, catch tray, hood, shutter); render ảnh để kiểm hood che việc seated và shutter che lamp.
- Collector với 20 episode pilot: belief-expert phải thành công; strata (spill/occluded/noisy) đúng tỉ lệ cấu hình.
- Probe AUC ≤ 0.6.
- **Script tạo label certificate** (hiện phải viết tay; nên có `scripts/make_certificate.py` tính hash).
- Đối chiếu success code BatteriesChecker của bản release MIKASA-90 với commit đã audit `3d5a21a…` (cần cho luận điểm C2).

### WP7 — Setup server (P0, làm ngay khi có máy)

- Chạy `scripts/setup_server.sh`. Rủi ro đã biết: OFT ghim torch 2.2.0, MIKASA ghim 2.2.1; nếu chọn π0.5 thì thêm openpi.
- Kiểm Vulkan cho SAPIEN.
- Cập nhật script theo D1 (cài openpi hoặc OFT).

### WP8 — Paper (P2)

- Thêm bảng cho Pha A/B (MIKASA, RoboMME) vào `latex-paper/eem-paper/main.tex` và các key `\res{...}` tương ứng trong `fill_results.py`.
- Đổi mô tả backbone theo D1 (§IV "Backbone and training", Table 3).

---

## 5. Thứ tự thực hiện đề xuất

| Pha | Máy | Việc | Gate để đi tiếp |
|---|---|---|---|
| 0 | CPU (máy dev) | WP1, WP2.1–2.2, WP3, WP5.1–5.2; test xanh | `verify_local.sh` pass với hai DataSpec |
| A | GPU | WP7 → WP2.3 smoke → MIKASA-Robo-VLA: B0, KEM, EEM (aux=none), 1 seed, khoảng 5–10 task gồm BatteriesChecker | Pipeline chạy hết từ train đến `fill_results`; số liệu hợp lý |
| B | GPU | WP4.2 + WP2.5 → RoboMME (Counting trước) | External validation |
| C | GPU | WP6 → BatteriesSort: B0, KEM, EEM → mở rộng baseline/seed nếu H2 và H5 đứng | Kết quả claim chính |
| D | GPU (tùy chọn) | RMBench Battery Try | — |

**Định nghĩa xong cho cả dự án:** `fill_results.py` không còn key `[?]` trong các bảng của Pha C; có ít nhất 3 seed cho EEM và KEM; mọi so sánh đi qua `check_provenance`.
