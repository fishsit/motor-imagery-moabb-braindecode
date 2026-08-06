# Motor Imagery BCI Reproduction

基于 **MOABB、MNE、Braindecode 与 PyTorch** 的运动想象脑机接口复现项目。

本项目以 **BNCI2014_001（BCI Competition IV 2a）** 为主要数据集，完成了从数据加载、预处理、传统机器学习基线，到 EEGNet、ShallowFBCSPNet、Deep4Net 深度学习模型训练与评价的完整流程。

项目重点包括：

- 四分类运动想象 EEG 解码；
- CSP + SVM 传统基线；
- EEGNet、ShallowFBCSPNet、Deep4Net；
- Cropped decoding；
- Exponential Moving Standardization；
- Leave-One-Subject-Out 跨被试评价；
- Run-wise 交叉验证与跨会话测试；
- 自动生成实验汇总、模型对比表和结果图。

---

## 1. 项目目标

本项目用于系统学习和复现运动想象 BCI 算法，主要目标为：

1. 熟悉 EEG 数据的加载、通道选择、滤波、标准化和 Epoch 构建；
2. 掌握 CSP + SVM 传统运动想象分类流程；
3. 理解 EEGNet、ShallowFBCSPNet 和 Deep4Net 的结构与训练方式；
4. 掌握 Cropped decoding 和 Dense Prediction；
5. 建立规范的训练集、验证集和测试集划分流程；
6. 比较跨被试与被试内跨会话两类评价协议；
7. 形成可复现、可扩展的 BCI 实验仓库。

---

## 2. 数据集

项目当前主要使用：

```text
BNCI2014_001
```

对应 BCI Competition IV 2a 四分类运动想象数据。

### 数据集基本信息

| 项目 | 内容 |
|---|---|
| 被试数量 | 9 |
| EEG 通道 | 22 |
| EOG 通道 | 3 |
| 采样率 | 250 Hz |
| 会话数量 | 每名被试 2 个会话 |
| 每个会话 Run 数 | 6 |
| 每个 Run Trial 数 | 48 |
| 每名被试 Trial 数 | 576 |
| 总 Trial 数 | 5184 |
| 分类数量 | 4 |

### 类别映射

| 标签 | 类别 |
|---:|---|
| 0 | feet |
| 1 | left_hand |
| 2 | right_hand |
| 3 | tongue |

---

## 3. 环境配置

已验证环境：

| 软件 | 版本 |
|---|---:|
| Python | 3.11.9 |
| PyTorch | 2.13.0+cpu |
| MOABB | 1.5.0 |
| Braindecode | 1.7.0 |
| MNE | 1.12.1 |
| NumPy | 2.4.6 |
| Pandas | 3.0.5 |
| Scikit-learn | 1.9.0 |
| 操作系统 | Windows 11 |

当前实验可在 CPU 环境运行，但 Deep4Net 多折交叉验证耗时较长。

### 创建虚拟环境

```powershell
python -m venv .venv
```

激活环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

安装依赖：

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

检查环境：

```powershell
python .\scripts\01_data_processing\check_environment.py
```

---

## 4. 项目结构

清理后的 `scripts/` 目录仅保留主要数据处理、正式训练、评价和汇总脚本。

```text
motor-imagery-moabb-braindecode/
├─ configs/
├─ data/
├─ docs/
├─ experiments/
│  └─ archive/
│     ├─ cropped_development/
│     ├─ csp_development/
│     ├─ deep4net_debug/
│     └─ deep4net_subject1/
├─ models/
├─ notebooks/
├─ results/
│  ├─ figures/
│  ├─ logs/
│  └─ metrics/
├─ scripts/
│  ├─ 01_data_processing/
│  │  ├─ check_environment.py
│  │  ├─ load_bnci2014.py
│  │  ├─ create_epochs.py
│  │  └─ preprocess_eeg.py
│  ├─ 02_baseline_csp_svm/
│  │  └─ loso_full_csp_svm.py
│  ├─ 03_deep_learning/
│  │  ├─ train_eegnet.py
│  │  ├─ plot_eegnet_history.py
│  │  ├─ evaluate_eegnet.py
│  │  ├─ train_shallowfbcspnet.py
│  │  ├─ plot_shallow_history.py
│  │  └─ evaluate_shallowfbcspnet.py
│  ├─ 04_cropped_training/
│  │  ├─ train_single_subject_cropped.py
│  │  └─ train_deep4net_runwise_cv.py
│  └─ 05_loso_evaluation/
│     ├─ train_single_fold.py
│     ├─ run_all_folds.py
│     ├─ plot_loso_results.py
│     ├─ generate_deep4net_summary.py
│     └─ generate_final_comparison.py
├─ .gitignore
├─ README.md
└─ requirements.txt
```

早期教学脚本、结构测试脚本、诊断脚本和被最终方案替代的实验脚本已移动至：

```text
experiments/archive/
```

这样既保持正式目录简洁，也保留实验开发过程。

---

## 5. 实验流程

```text
BNCI2014_001
      ↓
加载 EEG 数据
      ↓
选择 22 个 EEG 通道
      ↓
4–38 Hz 带通滤波
      ↓
伏特转换为微伏
      ↓
Exponential Moving Standardization
      ↓
Epoch / Windows 构建
      ↓
传统机器学习与深度学习训练
      ↓
交叉验证与跨会话测试
      ↓
生成指标、混淆矩阵和模型对比结果
```

---

## 6. 传统机器学习基线

### CSP + SVM

CSP + SVM 用作传统运动想象分类基线。

运行完整 LOSO：

```powershell
python .\scripts\02_baseline_csp_svm\loso_full_csp_svm.py
```

当前记录结果：

```text
Mean Accuracy = 38.89%
Std Accuracy  = 13.29%
```

该实验采用 9 名被试 Leave-One-Subject-Out 评价协议。

---

## 7. ShallowFBCSPNet

ShallowFBCSPNet 是针对 EEG 解码设计的浅层卷积神经网络，其结构与传统 FBCSP 思路具有较强联系。

项目已完成：

- 单被试 Cropped training；
- Exponential Moving Standardization；
- 多被试 LOSO；
- 各折模型训练与结果汇总。

### 运行 LOSO

```powershell
python .\scripts\05_loso_evaluation\run_all_folds.py
```

绘制结果：

```powershell
python .\scripts\05_loso_evaluation\plot_loso_results.py
```

### LOSO 结果

```text
Mean Accuracy          = 46.70%
Std Accuracy           = 15.98%
Mean Balanced Accuracy = 46.70%
Mean Macro-F1          = 42.96%
```

与 CSP + SVM 相比，平均准确率提升：

```text
46.70% - 38.89% = 7.81 个百分点
```

---

## 8. Deep4Net

Deep4Net 使用多层卷积结构和 Dense Prediction 进行 EEG 解码。

### 当前正式配置

```text
Input length   = 1000 samples
EEG channels   = 22
Classes        = 4
Filter         = 4–38 Hz
Dropout        = 0.25
Optimizer      = Adam
Learning rate  = 0.001
Weight decay   = 0
CV epochs      = 35
```

### Run-wise 交叉验证协议

每名被试的 `0train` 会话包含 6 个 run。

```text
Fold 1：run 0 验证，其余 run 训练
Fold 2：run 1 验证，其余 run 训练
...
Fold 6：run 5 验证，其余 run 训练
```

对六折每个 Epoch 的验证准确率求平均，选择平均验证准确率最高的 Epoch。随后：

1. 使用完整 `0train` 重新训练；
2. 训练轮数固定为所选 Epoch；
3. 在训练完成后评价一次 `1test`。

运行示例：

```powershell
python .\scripts\04_cropped_training\train_deep4net_runwise_cv.py --subject 3
```

批量运行 Subject 4–9：

```powershell
4..9 | ForEach-Object {
    python .\scripts\04_cropped_training\train_deep4net_runwise_cv.py `
        --subject $_ 2>&1 |
        Tee-Object ".\results\logs\deep4net_subject$($_)_runwise_cv.log"
}
```

### Deep4Net 各被试结果

| Subject | CV Accuracy | Test Accuracy | Macro-F1 | Selected Epoch |
|---:|---:|---:|---:|---:|
| 3 | 71.18% | 63.54% | 63.11% | 32 |
| 4 | 46.88% | 36.11% | 30.82% | 30 |
| 5 | 34.72% | 28.13% | 19.70% | 23 |
| 6 | 39.24% | 32.99% | 27.24% | 27 |
| 7 | 54.51% | 52.08% | 47.16% | 32 |
| 8 | 65.63% | 46.88% | 45.24% | 32 |
| 9 | 64.93% | 63.89% | 59.30% | 35 |
| **Mean ± Std** | **53.87%** | **46.23% ± 14.44%** | **41.80%** | — |

Deep4Net 在不同被试之间存在较明显性能差异，测试准确率范围为：

```text
28.13% – 63.89%
```

这反映了运动想象 EEG 较强的个体差异和跨会话分布变化。

### 生成 Deep4Net 汇总

```powershell
python .\scripts\05_loso_evaluation\generate_deep4net_summary.py
```

输出：

```text
results/metrics/deep4net_all_subject_summary.csv
results/metrics/deep4net_overall_summary.txt
```

---

## 9. EEGNet

项目已完成 EEGNet 的基础训练、EMS 标准化尝试和训练曲线绘制。

现有结果主要属于开发阶段结果，尚未完成与 CSP + SVM、ShallowFBCSPNet 或 Deep4Net 完全一致的正式评价协议，因此暂不纳入最终模型排名。

相关脚本：

```text
scripts/03_deep_learning/train_eegnet.py
scripts/03_deep_learning/plot_eegnet_history.py
scripts/03_deep_learning/evaluate_eegnet.py
```

---

## 10. 最终模型对比

运行：

```powershell
python .\scripts\05_loso_evaluation\generate_final_comparison.py
```

生成：

```text
results/metrics/model_comparison_final.csv
results/metrics/model_comparison_multi_subject.csv
results/metrics/model_comparison_single_subject.csv
results/metrics/model_comparison_final.md
results/metrics/model_comparison_notes.txt
results/figures/model_comparison_accuracy.png
results/figures/model_comparison_macro_f1.png
```

### 多被试总体结果

| 模型 | 评价协议 | 被试 | Accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| CSP + SVM | 9-subject LOSO | 1–9 | 38.89% ± 13.29% | — |
| ShallowFBCSPNet | 9-subject LOSO | 1–9 | **46.70% ± 15.98%** | **42.96%** |
| Deep4Net | 0train 内部 run-wise CV，1test 跨会话测试 | 3–9 | 46.23% ± 14.44% | 41.80% |

> CSP + SVM 与 ShallowFBCSPNet 使用相同的 LOSO 协议，可以进行较直接比较。Deep4Net 使用被试内跨会话协议，回答的是不同实验问题，因此不能简单作为同一排行榜进行比较。

### 单被试开发结果

| 模型 | Subject | Accuracy | Macro-F1 | 说明 |
|---|---:|---:|---:|---|
| ShallowFBCSPNet + EMS + Cropped | 1 | 64.93% | — | 单被试跨会话开发实验 |
| Deep4Net 原始配置 | 1 | 45.49% | 37.80% | Dropout=0.5，AdamW，Weight Decay=5e-4 |
| Deep4Net 低正则化配置 | 1 | 63.89% | 63.10% | Dropout=0.25，Adam，Weight Decay=0 |
| EEGNet | 1 | — | — | 尚未完成协议匹配的正式评价 |

---

## 11. 主要实验结论

1. 在相同 9 被试 LOSO 协议下，ShallowFBCSPNet 将 CSP + SVM 的平均准确率从 38.89% 提升至 46.70%。
2. Deep4Net 在 Subject 3–9 被试内跨会话测试中的平均准确率为 46.23%，平均 Macro-F1 为 41.80%。
3. Deep4Net 的测试准确率在 28.13% 到 63.89% 之间，说明模型效果受到明显的被试差异影响。
4. 原始 Deep4Net 配置正则化过强，降低 Dropout、改用 Adam 并关闭权重衰减后，Subject 1 的准确率由 45.49% 提升至 63.89%。
5. 某些被试仍存在明显类别预测偏置，跨会话分布变化和个体差异仍是主要挑战。
6. 单独查看 Accuracy 不足以反映类别均衡性，因此实验同时使用 Balanced Accuracy、Macro-F1、分类报告和混淆矩阵进行分析。

---

## 12. 结果文件

### 指标

```text
results/metrics/
```

主要包含：

- 各模型训练历史；
- LOSO 各折结果；
- Deep4Net 各被试指标；
- 分类报告；
- 预测结果；
- 最终模型对比表。

### 图像

```text
results/figures/
```

主要包含：

- Accuracy 曲线；
- Loss 曲线；
- 混淆矩阵；
- LOSO 结果图；
- 模型总体对比图。

### 模型

```text
models/
```

模型权重文件默认不上传 GitHub，相关规则已写入 `.gitignore`。

---

## 13. 文档

```text
docs/
├─ MI_BCI_experiment_report.md
├─ model_comparison.md
├─ model_comparison_generated.md
└─ LOSO_analysis.md
```

文档包括实验报告、模型比较和 LOSO 分析。

---

## 14. Git 使用建议

查看状态：

```powershell
git status
```

提交代码：

```powershell
git add -A
git commit -m "Update BCI experiments and documentation"
git push
```

数据集和模型文件通常较大，不应直接上传：

```text
data/
models/*.pth
```

---

## 15. 后续工作

后续可以继续完成：

- EEGNet 协议匹配评价；
- Deep4Net 全 9 被试统一实验；
- 更严格的嵌套交叉验证；
- 数据增强与类间决策边界优化；
- 多被试预训练与单被试微调；
- Riemannian Geometry 基线；
- Transformer 类 EEG 模型；
- 实时 EEG 数据流与在线推理；
- 与已有 EEG Data Manager 项目连接。

---

## 16. 项目状态

当前已完成：

```text
数据加载与检查                 PASS
EEG预处理                      PASS
Epoch与Windows构建             PASS
CSP + SVM基线                  PASS
ShallowFBCSPNet单被试训练       PASS
ShallowFBCSPNet LOSO            PASS
Deep4Net结构与训练流程验证      PASS
Deep4Net run-wise CV            PASS
Deep4Net跨会话评价              PASS
全被试结果汇总                  PASS
最终模型对比                    PASS
脚本目录精简                    PASS
```

本项目已经形成从数据处理、传统基线、深度学习训练，到规范评价和结果汇总的完整运动想象 BCI 复现流程。
