# 原始实验结果

README 中所有数字的出处，便于核对与复现。

| 文件 | 内容 | 条件 |
|---|---|---|
| `detection_ablation.csv` | E0–E3 检测消融 | 各自 `best.pt`，VisDrone val 548 图 / 38,759 实例，`max_det=1000` |
| `mot_global_summary.csv` | 3 个 tracker 的全量 MOT 指标 | VisDrone2019-MOT-val 7 序列 / 2,846 帧 / 112,915 GT，`conf=0.25` |
| `mot_per_sequence.csv` | 逐序列 MOT 指标 | 同上 |
| `mot_global_summary_conf010.csv` | `conf=0.10` 的全量 MOT 指标 | 同上，仅跟踪器输入阈值改为 0.10 |
| `mot_conf_ablation.csv` | `conf` 0.25 vs 0.10 对比汇总 | 见 README「跟踪阈值敏感性」 |
| `bench_backends.csv` | PyTorch / ONNX / TensorRT 三后端 | RTX 4090，`imgsz=960`，`batch=1`，warmup 30，300 张图 |
| `bench_pipeline.csv` | 检测 / 检测+跟踪流水线开销 | PyTorch 后端，同一序列 wall-clock 计时 |

MOT 指标由 `src/evaluate_mot.py`（自研 HOTA / CLEAR / Identity 实现）计算，尚未与 TrackEval 交叉验证，见 README「已知局限」。
