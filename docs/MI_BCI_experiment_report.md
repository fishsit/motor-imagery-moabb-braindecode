# 基于 MOABB 与 Braindecode 的运动想象脑机接口实验报告

## 1. 项目概述

本项目面向运动想象脑机接口（Motor Imagery Brain-Computer Interface，MI-BCI），基于公开数据集 **BNCI2014_001（BCI Competition IV 2a）**，使用 MOABB、MNE、Braindecode、Scikit-learn 与 PyTorch，完成从脑电数据读取、信号预处理、传统机器学习建模，到深度学习训练和跨被试泛化评价的完整实验流程。

项目的主要目标不是追求单次实验中的最高准确率，而是构建一套：

- 可重复；
- 可比较；
- 可扩展；
- 适合后续迁移学习和实时推理；

的运动想象 EEG 研究框架。

---

## 2. 研究内容

本项目完成了以下工作：

1. 使用 MOABB 自动下载和管理 BNCI2014_001；
2. 使用 MNE 与 Braindecode 读取 EEG 数据；
3. 分析 Subject、Session、Run、Channel、Sampling Rate 和 Event；
4. 将连续 EEG 切分为运动想象 Epoch；
5. 完成 EEG 通道筛选、带通滤波和动态标准化；
6. 建立 CSP + SVM 传统机器学习基线；
7. 训练 EEGNet 与 ShallowFBCSPNet；
8. 引入 Exponential Moving Standardization（EMS）；
9. 使用 WindowsDataset、Dense Prediction 和 CroppedLoss 实现 Cropped Training；
10. 完成单被试跨会话、多被试跨会话和 9 折 LOSO 跨被试实验；
11. 保存模型、训练历史、评价指标和可视化结果；
12. 对跨被试差异和类别混淆进行分析。

---

## 3. 数据集

### 3.1 数据集名称

**BNCI2014_001**

也称为：

**BCI Competition IV 2a**

### 3.2 数据集参数

| 项目 | 数值 |
| --- | ---: |
| 被试数量 | 9 |
| 每名被试会话数 | 2 |
| 每个会话运行数 | 6 |
| 每个运行试次数 | 48 |
| 每名被试总试次数 | 576 |
| 全部被试总试次数 | 5184 |
| 运动想象类别 | 4 |
| EEG 通道 | 22 |
| EOG 通道 | 3 |
| 刺激通道 | 1 |
| 采样率 | 250 Hz |
| 单次试次时间范围 | 0～4 s |
| 单次试次采样点 | 1001 |

四种运动想象类别为：

- `feet`
- `left_hand`
- `right_hand`
- `tongue`

---

## 4. 实验环境

| 软件 | 版本 |
| --- | --- |
| Python | 3.11.9 |
| PyTorch | 2.13.0+cpu |
| MOABB | 1.5.0 |
| Braindecode | 1.7.0 |
| MNE | 1.12.1 |
| NumPy | 2.4.6 |
| Pandas | 3.0.5 |
| Scikit-learn | 1.9.0 |
| 操作系统 | Windows 11 |

当前实验主要在 CPU 环境中运行。

---

## 5. 数据处理流程

```text
BNCI2014_001 原始 EEG
            ↓
      MOABB 自动下载与管理
            ↓
       MNE / Braindecode 读取
            ↓
解析 Subject / Session / Run / Event
            ↓
       保留 22 个 EEG 通道
            ↓
        V 转换为 μV
            ↓
        4～38 Hz 带通滤波
            ↓
Exponential Moving Standardization
            ↓
      Epoch / WindowsDataset 构建
            ↓
 传统模型与深度学习模型训练
            ↓
 单被试 / 跨会话 / LOSO 评价
```

### 5.1 通道选择

原始数据包含：

```text
22 EEG + 3 EOG + 1 Stim = 26 通道
```

建模时仅保留 22 个 EEG 通道。

### 5.2 滤波

传统 CSP + SVM 基线主要使用：

```text
8～30 Hz
```

Braindecode Cropped Training 流程使用：

```text
4～38 Hz
```

### 5.3 单位转换

MNE 中 EEG 默认以伏特存储，深度学习流程中转换为微伏：

```text
V × 1e6 → μV
```

### 5.4 指数移动标准化

使用 Braindecode 的 Exponential Moving Standardization，主要作用是减小：

- EEG 幅值尺度差异；
- 信号基线漂移；
- 长时间采集状态变化；
- 不同记录之间的分布波动。

---

## 6. 模型与方法

### 6.1 CSP + SVM

传统基线流程：

```text
EEG
 ↓
带通滤波
 ↓
CSP 空间滤波
 ↓
对数方差特征
 ↓
SVM 分类
```

该方法具有结构清晰、可解释性较强和适合小样本等特点。

### 6.2 EEGNet

EEGNet 是面向 EEG 设计的轻量级卷积神经网络，主要包括：

```text
时间卷积
  ↓
深度空间卷积
  ↓
可分离卷积
  ↓
分类器
```

### 6.3 ShallowFBCSPNet

ShallowFBCSPNet 将 FBCSP 思想融入卷积网络，主要结构为：

```text
时间卷积
  ↓
空间卷积
  ↓
平方非线性
  ↓
平均池化
  ↓
对数变换
  ↓
分类
```

### 6.4 Cropped Training

Cropped Training 的核心流程为：

```text
ShallowFBCSPNet
        ↓
to_dense_prediction_model()
        ↓
多个时间位置的密集预测
        ↓
CroppedLoss
        ↓
Trial-level prediction
```

与每个 trial 只进行一次预测相比，该方法能够更充分利用 trial 内的时序信息。

---

## 7. 实验协议

### 7.1 单被试随机划分

将同一被试或混合被试的 trial 随机划分为训练集和测试集。

该方法适合快速检查训练流程，但不代表跨被试泛化能力。

### 7.2 单被试跨会话

```text
Subject 1 的 0train → 训练
Subject 1 的 1test  → 验证
```

### 7.3 多被试跨会话

```text
9 名被试的 0train → 训练
9 名被试的 1test  → 验证
```

### 7.4 LOSO 跨被试

采用 Leave-One-Subject-Out：

```text
8 名被试 → 训练
1 名完全未见被试 → 测试
```

共进行 9 折。

每一折包含：

1. 仅使用训练被试的两个会话选择最佳 epoch；
2. 使用 8 名训练被试的全部会话重新训练新模型；
3. 在保留被试的全部数据上进行 trial-level 测试。

测试被试不参与：

- 梯度更新；
- EarlyStopping；
- 最佳 epoch 选择；
- 模型参数更新。

---

## 8. 实验结果

### 8.1 CSP + SVM

| 实验 | 数据范围 | Accuracy | STD |
| --- | --- | ---: | ---: |
| 单被试四分类 | Subject 1 单个运行 | 66.67% | — |
| 多被试随机划分 | 9 名被试，每人单个运行 | 44.44% | — |
| 部分数据 LOSO | 9 名被试，每人单个运行 | 38.89% | 13.29% |
| 完整数据 LOSO | 9 名被试全部会话和运行 | 36.59% | 10.77% |

### 8.2 深度学习阶段性结果

| 模型 | 划分方式 | Accuracy |
| --- | --- | ---: |
| EEGNet | 混合随机划分基线 | 41.75% |
| ShallowFBCSPNet | 混合随机划分，无 EMS | 29.51% |
| ShallowFBCSPNet + EMS | 混合随机划分 | 53.23% |
| ShallowFBCSPNet + EMS + Cropped | Subject 1 跨会话 | 64.93% |
| ShallowFBCSPNet + EMS + Cropped | 9 被试跨会话 | 60.15% |

> 上述实验采用的数据划分方式不同，因此不能只根据准确率高低进行严格模型排名。

### 8.3 ShallowFBCSPNet 完整 9 折 LOSO

| Subject | Accuracy | Balanced Accuracy | Macro-F1 | Best Epoch |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 64.76% | 64.76% | 60.05% | 17 |
| 2 | 29.69% | 29.69% | 25.70% | 17 |
| 3 | 66.32% | 66.32% | 66.53% | 17 |
| 4 | 37.15% | 37.15% | 32.54% | 20 |
| 5 | 25.69% | 25.69% | 11.99% | 20 |
| 6 | 30.90% | 30.90% | 27.03% | 18 |
| 7 | 41.84% | 41.84% | 42.24% | 16 |
| 8 | 58.33% | 58.33% | 55.85% | 20 |
| 9 | 65.62% | 65.62% | 64.75% | 15 |

汇总指标：

| 指标 | 结果 |
| --- | ---: |
| Mean Accuracy | 46.70% |
| Accuracy STD | 15.98% |
| Mean Balanced Accuracy | 46.70% |
| Mean Macro-F1 | 42.96% |
| 随机水平 | 25.00% |

---

## 9. 可视化结果

### 9.1 各被试 LOSO 准确率

```markdown
![LOSO Accuracy](../results/figures/loso_accuracy_subjects.png)
```

### 9.2 汇总混淆矩阵

```markdown
![LOSO Confusion Matrix](../results/figures/loso_confusion_matrix.png)
```

汇总混淆矩阵：

```text
[[516, 306, 194, 280],
 [191, 650, 226, 229],
 [160, 286, 682, 168],
 [277, 275, 171, 573]]
```

---

## 10. 结论

本项目完成了从传统 CSP + SVM、EEGNet、ShallowFBCSPNet，到 EMS、Cropped Training 和完整 LOSO 评价的实验闭环。

主要结论包括：

1. 深度模型能够学习比单一 CSP 空间特征更丰富的时空模式；
2. EMS 对 ShallowFBCSPNet 的训练稳定性影响明显；
3. Cropped Training 能够提升 trial 内时序信息利用率；
4. 跨被试性能明显低于同被试跨会话性能；
5. 被试之间差异较大，是当前系统的主要瓶颈；
6. 后续需要重点研究迁移学习、领域自适应和少样本校准。

---

## 11. 局限性

当前实验仍存在以下限制：

- 部分模型使用的划分协议不完全一致；
- 部分结果只运行单个随机种子；
- 未完成 EEGNet 的严格 LOSO；
- 未进行统计显著性检验；
- 未对滤波频带、窗口长度和学习率进行系统搜索；
- 未进行跨数据集验证；
- 未完成实时推理延迟测试。

---

## 12. 后续工作

后续计划包括：

- 统一 CSP-SVM、EEGNet 和 ShallowFBCSPNet 的 LOSO 协议；
- 增加 Deep4Net；
- 增加 EEGConformer；
- 增加黎曼几何分类基线；
- 研究迁移学习和领域自适应；
- 增加多随机种子重复实验；
- 增加统计显著性检验；
- 构建实时推理接口；
- 与 C/C++ Data Manager 和网络转发模块集成。
