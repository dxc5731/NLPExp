# CRF 四轮原始训练记录

| 轮次 | run 后缀 | 内容 |
|---|---|---|
| R1 | 71838655 | 初始 LR 2e-5、Dropout 0.1；E1/E2 后中断，保留中断材料 |
| R1 恢复 | 76608cc3 | 从 checkpoint 恢复，完成 E3–E5；不能当作第五轮实验 |
| R2 | 8aad7b72 | Dropout 0.2，最佳 E3 |
| R3 / B1 | 95abb833 | LR 1e-5、Dropout 0.1，最佳 E5，正式模型 |
| R4 / B2 | 86767c7e | 边界辅助 λ=0.2，最佳 E3，未选作正式模型 |

history.csv 保留原始数值；metrics/config/source_manifest 等与相应 run 一起留档。
training_events.jsonl 取自原 run.log，排除高频 train_batch 行，保留开始、checkpoint、完整 Epoch 等事件及源行号。
完整日志保留在负责人本地，不上传约 36 MB 的重复批次记录。五个 run 合计 20 个完整 Epoch，未完成 Epoch 不补点。
路径中的 ${PROJECT}、${USER_HOME} 等是归档占位，不是可直接执行的命令。
原件和公开副本 SHA-256 见 ../../delivery/provenance.json；副本内嵌的旧哈希仍指向历史原件，未伪改为副本哈希。
详细分析和训练曲线见 ../analysis/ 与 ../../report_assets/。
