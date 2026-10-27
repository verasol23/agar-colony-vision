# Pipeline Improved duy nhất và hướng dẫn debug

## 1. Luồng thuật toán

```text
Image (JPG/PNG)
  └─> detect_dish: Otsu + morphological closing, Hough fallback
        └─> crop_dish: working image 1024×1024
              └─> find_agar_boundary: polar transform + ellipse
                    ├─> Inner dish mask (96% default)
                    │     └─> Inpainting printed ink -> Gaussian -> Lab
                    │           └─> Median/Gaussian background estimation
                    │                 └─> weighted Lab distance
                    │                       └─> Median/MAD thresholds -> Hysteresis
                    │                             └─> suppress ink / rim artifacts
                    │                                   └─> Opening + fill small holes
                    │                                         └─> Distance transform + Watershed
                    │                                               └─> Inner colony boxes
                    │
                    └─> Annular HSV mask near agar boundary
                          └─> Color threshold + morphology + geometry
                                └─> Edge colony boxes

                  Merge inner boxes + nonduplicate edge boxes (IoU < 0.30)
                                └─> Detected colony boxes
                                      ├─> GUI / CSV / preview
                                      └─> GT JSON (ONLY for evaluation) -> IoU 0.50
```

Không dùng JSON để phát hiện. Nhãn chỉ phục vụ đánh giá và review. Phép chênh lệch Lab trong code là khoảng cách màu Euclid có trọng số, **không phải CIEDE2000**.

`config.py` giữ các ngưỡng cơ bản và màu HSV sát vành. Box nội vùng giữ kích thước tự nhiên từ Watershed, **không ép min_side=50 như bản Improved cũ**.

## 2. Debug theo hiện tượng

| Hiện tượng | Kiểm tra file | Chọn stage |
|---|---|---|
| Crop sai, đĩa lệch | `dish.py` | Cropped dish |
| Bỏ sót khuẩn lạc tại mép | `dish.py`, `edge.py` | Dish mask / Edge mask |
| Nền khác màu, ánh sáng không đều | `preprocessing.py` | Estimated background |
| Khuẩn lạc gần màu nền | `detection.py`, `segmentation.py` | Color distance / Hysteresis |
| Nhiễu, khuẩn lạc dính | `segmentation.py` | Morphology / Watershed labels |
| Box quá lớn hoặc lệch | `boxes.py`, `edge.py` | Detection |
| TP/FP/FN không hợp lý | `evaluation.py`, `annotations.py` | Evaluation |
| Ảnh lỗi lúc batch | `batch.py` | errors.csv / logs/batch.log |
| GUI không hiển thị box | `gui.py`, `gui_panels.py` | UI |

## 3. Cách đọc kết quả

- Prediction: các box sinh ra từ pipeline, `class = unclassified`.
- GT: JSON có `labels`, gồm tọa độ `x,y,width,height` trên ảnh gốc.
- TP: prediction ghép được với một GT tại IoU ≥ 0.50.
- FP: prediction không được ghép.
- FN: GT không được ghép.
- Precision = TP/(TP+FP); Recall = TP/(TP+FN); F1 = 2PR/(P+R).
- Chỉ tổng hợp TP/FP/FN từ ảnh có annotation hợp lệ. **F1 trên một ảnh không đại diện cho dataset**.

## 4. Các hạn chế cần ghi trong báo cáo

- Nhánh HSV sát vành dùng khoảng màu thử nghiệm trên 1000.jpg; có thể hoạt động khác với nền/ánh sáng/loài khác.
- Thuật toán chưa được điều chỉnh bằng validation tách biệt và **chưa có số liệu batch 120 ảnh sau khi sửa box**.
- Không có bộ tự phân loại năm loài; các nhãn loài khi xuất YOLO là do người duyệt gán.
- Ảnh gốc demo có 4 GT nhưng nhánh Improved cố định chỉ khớp 3 box với IoU 0.5; không được tự nhận chính xác tuyệt đối.


## Điều chỉnh tham số

Phiên bản v2.1 cho phép chỉnh trực tiếp tất cả ngưỡng xử lý chính trong **Parameters...**. Tham số nằm trong dataclass `DetectorConfig` ở `agar_colony/config.py` và được truyền đến `detection.py`, `segmentation.py`, `edge.py`. Giá trị mặc định tái hiện pipeline v2.0. Batch ghi `config_used.json` để tái lập thực nghiệm. Xem [PARAMETERS.md](PARAMETERS.md).
