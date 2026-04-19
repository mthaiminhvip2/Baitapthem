# Learnable Inverse Kinematic Solver Assignment

Project này đọc hai file `.pkl`, so sánh pose theo từng `frame_id`, chạy bước sửa pose qua adapter `inference.py`, rồi in lại tập `M`, tập `D`, `Q1`, `Q3`, `mean` trước và sau sửa. Chương trình cũng tạo `results.html` để nộp/chấm khi chọn tất cả frame.

## Chạy nhanh

```powershell
python main.py
```

Nếu sau 5 giây không nhập gì, chương trình dùng:

- `wham_cam1.pkl`
- `wham_cam2.pkl`
- `reduced=True`
- `frame_id=all`

Chạy không hỏi tương tác:

```powershell
python main.py --no-prompt --file1 wham_cam1.pkl --file2 wham_cam2.pkl --frame-id all --reduced true
```

## Dữ liệu mẫu để thử

Khi chưa có hai file thật:

```powershell
python sample_data.py --output samples
python main.py --no-prompt --file1 samples\wham_cam1.pkl --file2 samples\wham_cam2.pkl --frame-id all --reduced true
```

## Dùng SMPL và checkpoint thật

Nếu có file SMPL bản quyền và checkpoint pretrained của Learnable-SMPLify, truyền thêm:

```powershell
python main.py --file1 wham_cam1.pkl --file2 wham_cam2.pkl `
  --smpl-model-dir path\to\SMPL-family\smpl `
  --learnable-smplify-src path\to\Learnable-SMPLify\src `
  --checkpoint path\to\checkpoint.pth `
  --frame-id all
```

Nếu thiếu checkpoint hoặc repo, `inference.py` dùng fallback smoothing để vẫn tạo được báo cáo trước/sau. Trong `results.json` có ghi rõ `refined_by` và trạng thái inference.

## Ý nghĩa kết quả

- `M`: các pose có mâu thuẫn hướng trước/sau mặt phẳng thân người giữa hai file.
- `D`: các pose cùng hướng nhưng sai khác hình học lớn theo vector khoảng cách nội tại.
- `Q1`, `Q3`, `mean`: thống kê sai số khoảng cách của các cặp pose còn lại, tức không thuộc `M` và không thuộc `D`.

Nguồn tham khảo chính:

- Learnable-SMPLify official repo: <https://github.com/Charrrrrlie/Learnable-SMPLify>
- Gợi ý phân tích hướng/khoảng cách trong đề: <https://www.onlinegdb.com/fork/wX2tnDjQX0>
- Gợi ý timeout và đọc WHAM PKL trong đề: <https://www.onlinegdb.com/mWxgZWyoL>
