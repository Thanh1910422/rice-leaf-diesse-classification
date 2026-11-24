# So sánh MobileNetV2 và DenseNet121 - bộ dữ liệu Rice_Leaf_AUG

Tập test: 229 ảnh CHUNG của 2 mô hình (tập test riêng: MobileNetV2 570 ảnh, DenseNet121 570 ảnh). Mọi chỉ số được tính lại trên tập chung này.

| Chỉ số | MobileNetV2 | DenseNet121 |
|---|---|---|
| Accuracy (%) | 95.2 | 97.82 |
| KTC 95% accuracy (%) | 91.6–97.3 | 95.0–99.1 |
| Precision macro | 0.9537 | 0.9787 |
| Recall macro | 0.9524 | 0.9788 |
| F1 macro | 0.9528 | 0.9786 |
| MCC | 0.9424 | 0.9739 |
| ROC-AUC macro | 0.9946 | 0.9979 |
| Test loss | 0.211 | 0.1719 |
| Số tham số (triệu) | — | 7.04 |
| Dung lượng mô hình (MB) | — | 44.7 |
| Thời gian train (phút) | — | 217.7 |
| Thời gian dự đoán 1 ảnh (ms, CPU) | — | 633.0 |

## F1-score và recall từng lớp

| Lớp | F1 MobileNetV2 | F1 DenseNet121 | Recall MobileNetV2 | Recall DenseNet121 |
|---|---|---|---|---|
| Bacterial Leaf Blight | 0.9367 | 0.9474 | 0.9487 | 0.9231 |
| Brown Spot | 0.9315 | 0.9867 | 0.9189 | 1.0000 |
| Healthy Rice Leaf | 0.9859 | 1.0000 | 0.9722 | 1.0000 |
| Leaf Blast | 0.9000 | 0.9620 | 0.9231 | 0.9744 |
| Leaf scald | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Sheath Blight | 0.9630 | 0.9756 | 0.9512 | 0.9756 |

## Kiểm định McNemar (cùng tập test)

- Ảnh chỉ MobileNetV2 đúng (DenseNet121 sai): **3**
- Ảnh chỉ DenseNet121 đúng (MobileNetV2 sai): **9**
- p-value (McNemar chính xác): **0.1460** → khác biệt **không có ý nghĩa thống kê** (p ≥ 0.05): trên tập test này chưa đủ bằng chứng để kết luận mô hình nào chính xác hơn.

## Biểu đồ

- `per_class_f1.png`: F1-score từng lớp
- `val_accuracy_curves.png`: validation accuracy theo epoch
- `confusion_matrices.png`: confusion matrix của 2 mô hình

*Ghi chú: mô hình thiếu `training_summary.json` (train bằng code cũ) nên không có số tham số / thời gian train. Train lại để có đủ số liệu.*