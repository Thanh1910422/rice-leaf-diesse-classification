# Phân loại bệnh lá lúa - MobileNetV2 và DenseNet121 (học chuyển giao)

Bộ dữ liệu: **Rice_Leaf_AUG** - 6 lớp: Bacterial Leaf Blight, Brown Spot,
Healthy Rice Leaf, Leaf Blast, Leaf scald, Sheath Blight.

## 1. Cấu trúc: dùng chung ảnh gốc, mỗi mô hình một thư mục độc lập

```
Rice_Leaf_Disease_Classification/
├── app.py                          # ứng dụng web (tự hiện ô chọn mô hình)
├── src/
│   ├── mobilenetv2/                # MÔ HÌNH 1 - giữ nguyên như trước khi so sánh
│   │   ├── config.py               #   cấu hình RIÊNG (đường dẫn + tham số)
│   │   ├── image_utils.py          #   xử lý ảnh RIÊNG
│   │   ├── prepare_data.py         #   xử lý dữ liệu RIÊNG (khử trùng, chia tập)
│   │   ├── data_loader.py          #   đọc dữ liệu RIÊNG
│   │   ├── model.py                #   kiến trúc MobileNetV2
│   │   ├── train.py | evaluate.py
│   │   └── predict.py | gradcam.py | export_misclassified.py
│   ├── densenet121/                # MÔ HÌNH 2 - cùng các file như trên
│   │   ├── config.py  image_utils.py  prepare_data.py  data_loader.py
│   │   ├── model.py                #   kiến trúc DenseNet121
│   │   ├── train.py | evaluate.py
│   │   └── predict.py | gradcam.py | export_misclassified.py
│   └── compare_models.py           # đọc kết quả 2 thư mục và so sánh
├── data/
│   ├── raw/Rice_Leaf_AUG/<6 lớp>/  # ẢNH DÙNG CHUNG (gốc + aug_*) - không bị sửa
│   ├── processed/mobilenetv2/      # riêng MobileNetV2
│   ├── processed/densenet121/      # riêng DenseNet121
│   ├── splits/mobilenetv2/         # train/val/test.json riêng MobileNetV2
│   └── splits/densenet121/         # train/val/test.json riêng DenseNet121
├── models/
│   ├── mobilenetv2/                # mobilenetv2_rice_leaf_final.keras
│   └── densenet121/                # densenet121_rice_leaf_final.keras
└── outputs/
    ├── mobilenetv2/                # logs, plots, reports, gradcam, misclassified
    ├── densenet121/
    └── comparison/                 # bảng + biểu đồ so sánh + McNemar
```

## 2. Trước khi chạy

Xoá thư mục `src\` cũ (đừng giải nén đè) và kết quả cũ để không lẫn:

```powershell
Remove-Item -Recurse -Force src
# giải nén file zip vào thư mục dự án
Remove-Item -Recurse -Force data\splits, data\processed, models, outputs
```

`data\raw\Rice_Leaf_AUG` giữ nguyên. Có thể xoá `data\raw\RiceLeafDisease-BD5`.

## 3. Chạy

```powershell
pip install -r requirements.txt

# MÔ HÌNH 1
python -m src.mobilenetv2.prepare_data
python -m src.mobilenetv2.train
python -m src.mobilenetv2.evaluate

# MÔ HÌNH 2
python -m src.densenet121.prepare_data
python -m src.densenet121.train          # CPU: chậm hơn MobileNetV2 nhiều lần
python -m src.densenet121.evaluate

# SO SÁNH
python -m src.compare_models
streamlit run app.py
```

Công cụ khác của từng mô hình:

```powershell
python -m src.mobilenetv2.gradcam
python -m src.densenet121.gradcam
python -m src.densenet121.export_misclassified
python -m src.densenet121.predict duong\dan\anh.jpg
```

## 4. MobileNetV2 giữ nguyên như trước khi thêm DenseNet121

`src/mobilenetv2/` là code MobileNetV2 trước khi đưa DenseNet121 vào so sánh:
`model.py` và `data_loader.py` giống hệt; tham số giữ nguyên (ảnh 224x224,
batch 32, 10 + 40 epoch, LR 1e-3 / 1e-4, fine-tune 60 lớp cuối, dropout 0.3,
label smoothing 0.1, early stopping patience 8, trọng số lớp).
Chỉ THÊM phần ghi kết quả để so sánh (đánh dấu `[THEM DE SO SANH]`):
`train.py` lưu `training_summary.json` (thời gian train, số tham số);
`evaluate.py` lưu `test_predictions.npz` và thời gian dự đoán 1 ảnh.

## 5. DenseNet121 khác MobileNetV2 ở đâu

Tham số trong `src/densenet121/config.py` GIỐNG HỆT `src/mobilenetv2/config.py`,
trừ các điểm bắt buộc theo mạng nền (đánh dấu `[KHAC MOBILENETV2]`):

| | MobileNetV2 | DenseNet121 |
| --- | --- | --- |
| Mạng nền (ImageNet) | ~2.3 triệu tham số, 154 lớp | ~7 triệu tham số, 427 lớp |
| Chuẩn hoá pixel (trong `model.py`) | x/127.5 - 1 | (x/255 - mean)/std ImageNet |
| Phần fine-tune (`FINE_TUNE_LAST_LAYERS`) | 60 lớp cuối | dense block 4 (`"conv5_block1"`) |
| `BATCH_SIZE` | 32 | 16 (DenseNet tốn RAM hơn) |

Vì 2 file `prepare_data.py` giống nhau và cùng tham số chia dữ liệu (seed 42),
2 mô hình có **cùng tập train/val/test**. `compare_models` kiểm tra lại điều này
và báo lỗi nếu khác (ví dụ khi sửa tham số chia dữ liệu chỉ ở một bên).

## 6. Xử lý dữ liệu (`prepare_data` của mỗi mô hình)

Bộ Rice_Leaf_AUG có 3.829 file: 898 ảnh gốc + 2.931 ảnh `aug_*` (bản xoay/lật/
zoom sinh sẵn từ ảnh gốc). Mặc định `USE_AUG_IMAGES = True`: **dùng tất cả**.

| Bước | Ý nghĩa |
| --- | --- |
| Bỏ ảnh trùng y hệt | Cùng nội dung file (MD5) -> giữ 1 |
| Bỏ ảnh gốc gần trùng | Cùng một chiếc lá (tương quan >= 0.95) -> giữ 1 ảnh nét nhất |
| Gom ảnh chụp liên tiếp | Chụp cách nhau <= 10 giây và giống nhau -> luôn chung một tập |
| **Ghép ảnh `aug_*` với ảnh gốc** | Mỗi ảnh `aug_*` được ghép với ảnh gốc giống nhất (cùng lớp, so với cả bản lật/xoay của ảnh gốc, đặc trưng MobileNetV2 ImageNet) và xếp **cùng tập** với ảnh gốc đó |
| Chia 70/15/15 | Phân tầng theo lớp, theo nhóm, seed 42 |

Vì sao phải ghép: bản xoay của một ảnh test mà nằm trong train thì mô hình được
chấm trên ảnh gần như đã thấy -> accuracy bị thổi phồng. Ảnh `aug_*` "khó xác
định" (giống 2 ảnh gốc ở 2 nhóm gần ngang nhau) -> gộp 2 nhóm đó chung một tập.

Hai thư mục cố ý dùng CÙNG cách ghép (MobileNetV2) nên có cùng train/val/test.
Đặt `USE_AUG_IMAGES = False` (ở CẢ 2 file `config.py`) để chỉ dùng ảnh gốc.

Sau `prepare_data`, kiểm tra:
- Bảng train/val/test, các dòng "trong do anh goc / anh aug_*", và dòng
  `Ty le chia cua tung lop: OK` (có CANH BAO thì không train).
- `outputs\<mô hình>\reports\aug_matching_preview.jpg`: mỗi cặp = ảnh `aug_*` |
  ảnh gốc được ghép. Hai ảnh phải là **cùng một chiếc lá** (bị xoay/lật/zoom).
  Danh sách đầy đủ: `data\splits\<mô hình>\aug_matching.csv`.
- `outputs\<mô hình>\reports\near_duplicate_preview.jpg`.

Lưu ý: dữ liệu nhiều gấp ~4 lần nên mỗi epoch train lâu hơn ~4 lần; tập test có
cả ảnh gốc lẫn bản `aug_*` của chúng (các bản này giống nhau nên con số accuracy
không "độc lập" bằng số ảnh - nên ghi rõ trong báo cáo).

## 7. Kết quả (`<mô hình>` = mobilenetv2 hoặc densenet121)

| File | Nội dung |
| --- | --- |
| `models/<mô hình>/<mô hình>_rice_leaf_final.keras` | Mô hình cuối |
| `outputs/<mô hình>/plots/` | training_curves, confusion_matrix, roc_curves |
| `outputs/<mô hình>/reports/classification_report.txt` | Accuracy, F1, MCC, ROC-AUC từng lớp |
| `outputs/<mô hình>/training_summary.json` | Số tham số, thời gian train |
| `outputs/<mô hình>/gradcam/` | Grad-CAM |
| `outputs/<mô hình>/misclassified/` | Ảnh đoán sai |
| `outputs/comparison/comparison_report.md` | So sánh 2 mô hình + kiểm định McNemar |
| `outputs/comparison/*.png` | F1 từng lớp, val accuracy theo epoch, 2 confusion matrix |

Kiểm định **McNemar** cho biết chênh lệch giữa 2 mô hình trên cùng tập test có ý
nghĩa thống kê hay không (p < 0.05) - nên đưa vào báo cáo thay vì chỉ so accuracy.
