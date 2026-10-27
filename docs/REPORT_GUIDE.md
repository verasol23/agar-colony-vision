# Gợi ý nội dung báo cáo đồ án Machine Vision

**Tên đề tài:** Phát hiện và gán nhãn khuẩn lạc trên đĩa thạch bằng xử lý ảnh truyền thống.

## 1. Mục tiêu
- Xác định vùng đĩa thạch và khuẩn lạc từ ảnh chụp.
- Tách khuẩn lạc khỏi nền, tạo bounding box và đếm số lượng.
- Hỗ trợ chuyên gia hiệu chỉnh nhãn trên giao diện và xuất dữ liệu.
- Đánh giá định lượng theo annotation gốc ở mức đối tượng.

## 2. Phương pháp đề xuất — Improved

Mô tả theo sơ đồ trong `docs/PIPELINE.md`. Hai nhánh **nội vùng** (Lab, Hysteresis, Morphology, Watershed) và **sát vành** (HSV annulus) được hợp nhất, sau đó loại box trùng. Không giới thiệu Baseline như một chế độ phần mềm hiện tại.

## 3. Kiến trúc phần mềm

`GUI/CLI -> detection.py -> dish.py + preprocessing.py + segmentation.py + boxes.py + edge.py -> evaluation.py -> batch.py -> CSV / overlays`.

Giải thích lý do tách mỗi module: dễ thay tham số, kiểm thử độc lập và truy lỗi bằng hình ảnh các stage.

## 4. Thử nghiệm

Dùng folder `countable/` và annotation JSON cùng tên; ngưỡng IoU 0.50. Trình bày TP/FP/FN, Precision/Recall/F1 và các ảnh minh họa trường hợp bị bỏ sót/nhận nhầm. Nêu rõ micro-average toàn dataset.

**Số liệu đã kiểm chứng (một ảnh mẫu 1000.jpg):** GT=4, Pred=4, TP=3, FP=1, FN=1, Precision=0.750, Recall=0.750, F1=0.750.

**Không dùng kết quả này làm kết luận toàn bộ AGAR.** Phải chạy project trên dataset thật và sử dụng CSV mới trước khi báo cáo số liệu tổng hợp.

## 5. Hạn chế

Chưa có phân loại loài tự động; ROI dễ bỏ sót vùng sát mép; ngưỡng HSV phụ thuộc dữ liệu; phương pháp truyền thống nhạy với thay đổi ánh sáng, màu đĩa và số lượng khuẩn lạc dính nhau.
