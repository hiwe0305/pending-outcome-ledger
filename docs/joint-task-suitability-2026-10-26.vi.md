# Audit: BatteriesChecker có cần joint execution belief không?

Ngày: 2026-10-26. Kết luận áp dụng cho snapshot MIKASA local được pin tại `3d5a21a99e5df2bcc34c612040b269e394b09438`, không tự động áp dụng cho mọi phiên bản benchmark.

## Quyết định

**Không dùng BatteriesChecker làm bằng chứng chính cho tính cần thiết của joint belief giữa location L và outcome Q.** Checklist theo battery identity, current perception, và finite-state controller là baseline đủ về logic nhiệm vụ khi nhận biết đúng các event. Chưa có rollout chứng minh độ tin cậy của baseline từ RGB. Belief vẫn có thể giúp khi perception/action không chắc chắn; đó là giả thuyết khác với “phải nhớ tương quan tốt/hỏng–vị trí mới giải được task”.

## 1. Những gì code thực sự chấm

Các đường dẫn bên dưới tương đối với `code/joint-execution-memory/third_party/mikasa/mikasa_robo_suite/vla/`.

| Bằng chứng | Locator |
|---|---|
| Working batteries được lấy từ active set; target count bằng số working được sinh | `memory_envs/batteries_checker_hard_vla.py:398–403,463–473` |
| Released insertion vào socket chuyển INSERT → RETURN; quality chỉ điều khiển lamp ở đoạn này | cùng file `:510–552` |
| Active battery về home slot và được thả chuyển RETURN → CONFIRM | cùng file `:583–607` |
| Bấm button ở CONFIRM đánh dấu checked cho cả pin tốt và hỏng; simulator tự cập nhật found_working bằng working_mask | cùng file `:609–627` |
| Success cần all_checked, all_returned, đủ found_working | cùng file `:629–653` |
| Easy tự teleport pin về home và chuyển CONFIRM | `memory_envs/batteries_checker_easy_vla.py:38–76` |
| Collector duyệt pin theo index; pick → insert → extract → return → confirm, không rẽ nhánh theo lamp | `utils/motion_planning/motion_planning_batteries_checker_hard.py:891–937` |

Gọi C_i là checked, W_i là quality thật, F_i là found_working. Reset cho C_i=F_i=0 trên active set. Một confirmation hợp lệ đặt C_i=1 và F_i := F_i OR (W_i AND newly_checked). Do đó trên các trạng thái hợp lệ:

`F_i = C_i AND W_i`.

Khi mọi active battery đã checked, tổng F_i bằng tổng W_i, chính là target_working_count. Điều kiện “đủ pin tốt” vì vậy đã được bảo đảm bởi all_checked; policy không cần tự phân loại pin tốt/hỏng để đạt success. Chữ “observing the lamp result” trong language instruction không được evaluator kiểm tra bằng một câu trả lời hoặc hành động phân loại riêng.

Controller đủ về logic: chọn slot chưa checked → pick/insert → xác nhận insertion → lấy ra/đặt về đúng slot → xác nhận return → bấm button → xác nhận confirmation → đánh dấu checked. Với thứ tự cố định và không lỗi, checklist thậm chí có thể rút thành cursor k và stage. Không được đánh dấu hoàn tất chỉ vì đã phát lệnh: cần event detection/recovery. Collector dùng privileged state, nên không được gọi nó là baseline RGB công bằng.

Q phải được định nghĩa nhất quán: (a) working_mask là hidden physical quality; (b) tested-working/tested-not-working là accumulated evidence của proposal; (c) checked là hoàn thành protocol có button confirmation. Ba thứ này khác nhau. Nếu Q đổi thành stage/test-validity, cần viết lại claim; đó không còn cùng giả thuyết outcome quality ban đầu.

## 2. Dữ liệu được kiểm tra trực tiếp

Đọc cả bốn numeric Parquet, metadata nguồn, và schema/shapes của bốn NPZ RGB sample bằng `python code/joint-execution-memory/scripts/audit_joint_task_suitability.py`.

| Task | Episodes | Frames | Mean length | RGB sample frames |
|---|---:|---:|---:|---:|
| Easy-3 | 250 | 91,268 | 365.1 | 362 |
| Easy-6 | 250 | 182,094 | 728.4 | 731 |
| Hard-3 | 250 | 176,397 | 705.6 | 684 |
| Hard-6 | 250 | 355,221 | 1420.9 | 1425 |

Tổng 1,000 episodes, 804,980 frames numeric. RGB local đã giải mã chỉ gồm 4 episodes; chưa xem toàn bộ video của 1,000 episodes trong audit này. Numeric columns: observation.state, action, timestamp, frame_index, episode_index, index, task_index. Không có L/Q/E, working identity, checked/event labels mỗi frame. Metadata success_once có 250 true ở mỗi task, không phải rollout mới hay kết quả của model chúng ta. Không suy ra “không có lỗi trung gian” từ success_once.

**Phát hiện provenance:** độ dài theo episode index trong Parquet không khớp thứ tự episode_lengths ở source metadata trên cả bốn task, dù multiset độ dài khớp hoàn toàn. Ví dụ Hard-3 episode 0 dài 684 frame, phần tử metadata đầu là 700. Do đó không được ghép `episode_seeds[i]` vào Parquet episode i khi chưa tìm conversion mapping. Các seed sample gắn bằng index trong audit cũ chưa được xác minh; không dùng để replay/gán nhãn. Multiset khớp gợi ý reorder nhưng chưa chứng minh nguyên nhân hay mapping.

Script cũng kiểm tra 72 quality assignments (toàn bộ binary masks với N=3 và N=6; superset của working counts cấu hình) trên logic confirmation trừu tượng: hoàn tất mọi battery luôn tìm đủ working count. Đây là sanity check cho suy luận trên code, **không phải simulator test**. Python hiện dùng chưa có mani_skill/sapien.

## 3. Task có sẵn thay thế

| Task đã đọc source | Thông tin cần giữ | Đánh giá |
|---|---|---|
| ShellGameShuffleColorLampTouch / Long | Mapping màu–cốc–slot qua chuỗi swap; lamp chọn màu sau shuffle | Ứng viên tốt nhất cho binding/tracking; không tự chứng minh joint execution outcome |
| GatherAndRecall | Progress đặt cube + màu flash toàn cục | Hai memory tách biệt có thể đủ; negative control tốt |
| TimedTransfer | Thời điểm signal + vị trí cube | Timer + perception là baseline mạnh; không có outcome identity binding rõ |

Source shell: `memory_envs/shell_game_shuffle_color_lamp_touch_vla.py:320–343` sinh swaps/permutation/target; `:383–402` ẩn ball và bật target lamp; `:405–451` animate cup; `:454–461` chấm khoảng cách TCP tới slot của đúng màu. Long variant `:539–546` có horizon600 và 5–15 swaps. Task được đăng ký sẵn; không phải task mới do chúng ta tạo. Official listing: https://mikasarobo.github.io/vla_environments/index.html . Chưa tải/kiểm tra dataset của task shell trong audit này.

Ví dụ hợp lệ ở short variant: ban đầu [red,green,blue] tại slots [0,1,2]. History A swap(0,1) rồi swap(1,2) cho red ở slot2; history B swap(1,2) rồi swap(0,1) cho red ở slot1. Cả hai đều có checklist “đã thấy ba màu”, “đã shuffle hai lần”, cùng tập slot tham gia; target lamp cùng red nhưng hành động đúng khác. Cần giữ **association**, không chỉ checklist hoặc hai danh sách mất liên kết.

Giới hạn quan trọng: deterministic permutation tracker lưu association đã giải được ví dụ này. Vì color gắn cố định với cup identity trong code, một bộ location tracker indexed by known color cũng đủ. Không được gọi ví dụ trên là chứng minh rằng full probabilistic joint distribution tốt hơn object-indexed marginals. Neural memories có decoder chung còn có thể ngầm mã hóa association. Q=color ở đây cũng không phải Q=execution outcome. Shuffle là scripted, không do action robot tạo; task phù hợp kiểm thử representation hơn action-conditioned world model.

## 4. Thí nghiệm và gate trước khi train lớn

1. **Batteries negative control:** cùng perception/backbone/data/controller, so checklist+stage, independent belief, joint belief. Thêm lamp-masked evaluation riêng, không đổi evaluator. Model có thể suy giảm vì distribution shift, nên ưu tiên train/evaluate cùng masking protocol và so sánh paired seeds. Nếu checklist giải tốt khi không thấy lamp, claim outcome-memory necessity bị bác bỏ. Báo success, repeat-check count, protocol violations; không chỉ action loss.
2. **Shell nguyên bản:** so deterministic permutation tracker, per-color location marginals, capacity-matched recurrent memory, và distribution over 6 permutations. Equal observation rate/history, latency, decoder và train budget. Đầu tiên xác nhận RGB tracking và không lộ oracle_info/target slot qua wrapper. Đánh giá target-slot accuracy trước motor controller để tách lỗi memory và control.
3. **Uncertainty stress test:** trên cùng shell task, thêm frame-drop/occlusion được khai báo là evaluation modification; phân tầng ambiguity và số swap. Báo final success, permutation NLL, per-color location NLL/Brier, calibration, recovery sau quan sát mới, paired-bootstrap CI qua episodes và nhiều training seeds. Các level như 0/10/30% frame-drop là đề xuất, chưa chạy. Không có guarantee rằng bất định mạnh hơn luôn làm joint có lợi.
4. **Gate về necessity:** phải tìm các history sinh được từ simulator với cùng marginals của đúng latent variables đang đề xuất nhưng posterior joint khác, và khác optimal continuation sau cùng admissible future evidence. Nếu không tìm thấy, chỉ claim relational memory/binding; không claim cần joint uncertainty. Với terminal query chọn một màu, per-color marginal có thể đã đủ để quyết định; full permutation belief phải chứng minh lợi ích ở update/recovery, không chỉ ở representation size.
5. **Giữ nguyên hướng L×execution-Q:** chưa tìm thấy task nguyên bản đã audit nào chứng minh đủ yêu cầu. Có thể làm extension BatteriesChecker yêu cầu đặt pin tốt/hỏng vào hai nơi sau một quá trình gây uncertainty về association, nhưng phải gắn nhãn benchmark extension, thiết kế shortcut controls, và so với object-indexed deterministic memory. Đây là hướng đề xuất, chưa là existing task hay novelty được xác minh.

Quyết định đầu tư: giữ BatteriesChecker làm control; dùng shell có sẵn cho pilot binding và uncertainty, trước khi train VLA lớn. Chỉ tiếp tục claim joint execution belief sau khi vượt qua gate4 trên task phù hợp. Không lấy memory horizon dài hoặc model joint thắng một baseline yếu làm bằng chứng cho correlation necessity.

Artifact máy đọc: `../runs/joint-task-suitability-2026-10-26.json` (hash source, data stats, counterexample). Audit không sửa simulator, dữ liệu gốc hay proposal.
