# Điều chỉnh tham số phát hiện khuẩn lạc (Improved Only v2.1)

## 1. Mục đích

Phiên bản v2.1 có bảng **Parameters...** trong GUI. Người dùng có thể thay đổi tham số và **Apply + Rerun** để xem ảnh phát hiện lại theo một cấu hình duy nhất. Những giá trị mặc định giữ nguyên hành vi của v2.0; không có tối ưu tham số tự động.

Các phép phát hiện **không đọc tọa độ từ JSON**. Ground Truth chỉ sử dụng để đánh giá kết quả. Trong một ảnh, thay đổi `iou_threshold` chỉ thay đổi việc so khớp TP/FP/FN, không thay đổi số box dự đoán.

## 2. Hướng dẫn thực hiện

1. Chạy `python main.py` và chọn ảnh hoặc thư mục dataset.
2. Chọn **Parameters...**.
3. Chỉnh các ô số hoặc thanh trượt ở các tab `Dish / Background`, `Segmentation`, `Watershed / Boxes`, `Edge HSV`, `Evaluation`.
4. Chọn **Apply + Rerun** để kiểm tra thay đổi trên ảnh hiện tại. Trường hợp chưa mở ảnh, tham số sẽ áp dụng khi mở ảnh tiếp theo.
5. Chọn **Save Preset** để lưu toàn bộ giá trị hiện có thành JSON (nếu chưa Apply, chỉ preset được lưu). **Load Preset** để nạp, rồi chọn **Apply + Rerun**.
6. Chọn **Run entire folder** để dùng bộ tham số đã Apply. Project ghi thêm `results/config_used.json` làm bằng chứng tái lập kết quả.

**Lưu ý:** Không thể Apply một bộ cấu hình mới khi batch đang chạy. Batch giữ nguyên một bộ tham số từ đầu đến cuối, tránh thay đổi giữa chừng.

## 3. Những tham số quan trọng

| Tên trong code | Mặc định | Vai trò |
|---|---:|---|
| `dish_shrink` | 96 | Phần trăm ellipse giữ trong ROI, quá nhỏ có thể mất khuẩn lạc sát mép |
| `gaussian_sigma` | 1.0 | Độ làm mượt ảnh trước khi chuyển Lab; 0 = tắt |
| `bg_kernel` | 121 | Kernel median ước lượng nền, nhập số lẻ |
| `l_weight` | 1.0 | Trọng số khác biệt sáng `ΔL` trong khoảng cách màu Lab |
| `thr_low`, `thr_high` | 3.5 / 8.0 | Ngưỡng thấp/cao của Hysteresis tính theo MAD chuẩn hóa |
| `multiscale_sigma` | 2.0 | Làm mượt bản đồ khoảng cách màu ở nhánh quy mô lớn; 0 = tắt |
| `opening_radius` | 2 | Bán kính Opening; kernel `(2r+1)×(2r+1)` |
| `hole_max` | 300 | Diện tích lỗ tối đa được lấp |
| `min_area`, `max_area` | 8 / 25000 | Lọc vùng liên thông theo diện tích |
| `ws_smooth` | 1.0 | Làm mượt bản đồ Distance Transform trước Watershed |
| `ws_alpha` | 1.0 | Điều khiển khoảng cách tối thiểu khi giữ marker |
| `box_pad` | 1 | Số pixel nới thêm mỗi cạnh box phía trong |
| `dedup_iou` | 0.30 | Ngưỡng loại box trùng giữa nhánh trong đĩa và nhánh sát vành |
| `edge_r_min`, `edge_r_max` | 0.90 / 1.09 | Khoảng bán kính tương đối của vùng tìm sát vành |
| `edge_h_low`, `edge_h_high` | 20 / 42 | Khoảng Hue (OpenCV 0–179) của nhánh sát vành |
| `edge_s_min` | 80 | Saturation tối thiểu để nhận ứng viên sát vành |
| `edge_v_low`, `edge_v_high` | 140 / 210 | Khoảng Value (OpenCV 0–255) của nhánh sát vành |
| `edge_min_area`, `edge_max_area` | 250 / 6000 | Lọc ứng viên sát vành theo diện tích |
| `iou_threshold` | 0.50 | Ngưỡng đánh giá trùng khớp dự đoán–annotation |

Các tham số hợp lệ được kiểm tra trong `agar_colony/config.py`. Quan hệ `low <= high` và `min_area <= max_area` phải được duy trì. `bg_kernel` phải là số lẻ.

## 4. Chạy lại preset với CLI

```bash
python -m agar_colony.cli --dataset examples/countable --output results --preset presets/default.json
```

Để thử một bộ đã lưu bằng GUI:

```bash
python -m agar_colony.cli --dataset "D:/AGAR/countable" --output "D:/AGAR_results" --preset "D:/AGAR/presets/experiment_A.json"
```

Lệnh cho phép ghi đè riêng ngưỡng đánh giá mà không sửa JSON:

```bash
python -m agar_colony.cli --dataset examples/countable --output results --preset presets/default.json --iou 0.70
```

`results/config_used.json` lưu cấu hình **thực tế đã được áp dụng** (kể cả những giá trị ghi đè trên CLI).

## 5. Cách điều chỉnh đúng về mặt khoa học

- Bắt đầu từ `presets/default.json` và chỉ thay đổi **1 nhóm tham số mỗi lần**.
- Xem kỹ ảnh trung gian (`Dish mask`, `Color distance`, `Hysteresis`, `Morphology`, `Watershed labels`, `Edge mask`) để biết lỗi xuất hiện ở bước nào.
- Với khuẩn lạc sát mép: đánh giá `dish_shrink` và các ngưỡng `edge_*`; nếu mask mở quá rộng sẽ nhận nhầm chữ/vành đĩa.
- Với khuẩn lạc nhỏ: xem `min_area`, `opening_radius`, `thr_low/high`; tránh làm lớn mọi bounding box để tăng F1 một ảnh.
- Với vùng dính nhau: chỉnh `ws_smooth` và `ws_alpha`, theo dõi under-segmentation và over-segmentation.
- Thử trên **validation** rồi cố định preset khi chạy **test**; không tối ưu trên tập test.
- Hệ thống hiện **chỉ phát hiện khuẩn lạc**. Các class hiển thị khi sửa tay không phải dự đoán phân loại loài.

## 6. Hướng debug theo module

| Vấn đề | Module |
|---|---|
| Đĩa/ROI bị lệch | `dish.py` + `config.py` |
| Kết quả không đổi khi chỉnh background | `preprocessing.py`, `detection.py` |
| Hysteresis/Morphology sai | `segmentation.py` |
| Watershed chia sai | `segmentation.py::split_colonies` |
| Khuẩn lạc sát vành bị thiếu | `edge.py` |
| Nhãn TP/FP/FN thay đổi khi chỉnh IoU | `evaluation.py` |
| GUI không áp dụng preset | `gui_params.py`, `gui.py` |
| Batch không dùng preset | `batch.py` và `config_used.json` |
