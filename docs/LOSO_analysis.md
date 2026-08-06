# ShallowFBCSPNet 9 折 LOSO 跨被试实验分析

## 1. 实验目的

LOSO（Leave-One-Subject-Out）用于评价模型对完全未见被试的泛化能力。

运动想象 EEG 存在明显个体差异，因此同被试随机划分或跨会话实验不能完全代表新用户直接使用模型时的性能。

本项目采用 9 名被试，进行 9 折 LOSO：

```text
Fold 1：Subject 2～9 训练，Subject 1 测试
Fold 2：Subject 1、3～9 训练，Subject 2 测试
...
Fold 9：Subject 1～8 训练，Subject 9 测试
```

---

## 2. 无测试泄漏的训练流程

每一折分为三个阶段。

### 阶段 1：内部选择最佳 Epoch

仅使用 8 名训练被试：

```text
训练被试 0train → 内部训练
训练被试 1test  → 内部验证
```

根据内部验证准确率选择最佳 epoch。

### 阶段 2：重新训练

创建全新随机初始化模型，并使用 8 名训练被试的全部会话进行训练：

```text
8 名训练被试的 0train + 1test → 最终训练
```

### 阶段 3：测试保留被试

仅在完全未参与训练和模型选择的测试被试上评价：

```text
保留被试全部会话 → Trial-level 测试
```

测试被试不参与：

- 梯度更新；
- EarlyStopping；
- epoch 选择；
- 超参数更新。

---

## 3. 评价指标

本实验报告：

- Accuracy；
- Balanced Accuracy；
- Macro-F1；
- Confusion Matrix；
- 各被试最佳 Epoch；
- 9 折均值和标准差。

由于每名被试四类 trial 数量相同，因此：

```text
Accuracy = Balanced Accuracy
```

在当前数据分布下二者数值一致。

---

## 4. 逐被试结果

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

---

## 5. 汇总结果

| 指标 | 结果 |
| --- | ---: |
| Mean Accuracy | 46.70% |
| Accuracy STD | 15.98% |
| Mean Balanced Accuracy | 46.70% |
| Mean Macro-F1 | 42.96% |
| 四分类随机水平 | 25.00% |

结果高于四分类随机水平，但不同被试之间波动较大。

---

## 6. 被试差异分析

### 6.1 表现较好的被试

| Subject | Accuracy |
| ---: | ---: |
| 3 | 66.32% |
| 9 | 65.62% |
| 1 | 64.76% |
| 8 | 58.33% |

这些被试的运动想象模式与训练被试共享特征较明显，模型具有较好的跨被试迁移效果。

### 6.2 表现较差的被试

| Subject | Accuracy |
| ---: | ---: |
| 5 | 25.69% |
| 2 | 29.69% |
| 6 | 30.90% |

Subject 5 接近 25% 随机水平，说明通用模型几乎没有学习到能够适应该被试的稳定模式。

### 6.3 标准差

```text
STD = 15.98%
```

较大的标准差说明：

- 被试间 EEG 分布差异明显；
- 单一平均准确率不足以描述模型表现；
- 必须报告逐被试结果；
- 后续需要被试适配方法。

---

## 7. 汇总混淆矩阵

```text
[[516, 306, 194, 280],
 [191, 650, 226, 229],
 [160, 286, 682, 168],
 [277, 275, 171, 573]]
```

类别顺序为：

```text
feet
left_hand
right_hand
tongue
```

每个类别总样本数为：

```text
1296
```

---

## 8. 各类别召回率

| 类别 | 正确数 / 总数 | Recall |
| --- | ---: | ---: |
| feet | 516 / 1296 | 39.81% |
| left_hand | 650 / 1296 | 50.15% |
| right_hand | 682 / 1296 | 52.62% |
| tongue | 573 / 1296 | 44.21% |

观察结果：

- `right_hand` 召回率最高；
- `left_hand` 次之；
- `tongue` 居中；
- `feet` 最难识别。

---

## 9. 主要混淆关系

### 9.1 feet 的混淆

真实 `feet` 被预测为：

- `left_hand`：306；
- `tongue`：280；
- `right_hand`：194。

### 9.2 left_hand 与 right_hand

两类之间存在明显混淆：

```text
left_hand → right_hand：226
right_hand → left_hand：286
```

这说明跨被试条件下左右手空间模式并未完全对齐。

### 9.3 tongue 的混淆

真实 `tongue` 经常被预测为：

- `feet`：277；
- `left_hand`：275。

---

## 10. 与 CSP + SVM LOSO 对比

| 方法 | Mean Accuracy | STD |
| --- | ---: | ---: |
| CSP + SVM | 36.59% | 10.77% |
| ShallowFBCSPNet + EMS + Cropped | 46.70% | 15.98% |

平均准确率绝对提升：

```text
10.11 个百分点
```

但 ShallowFBCSPNet 的被试间波动也更大：

```text
15.98% > 10.77%
```

这说明深度模型提升了部分被试的性能，但对困难被试的适应仍不稳定。

---

## 11. 结论

本次 LOSO 实验说明：

1. ShallowFBCSPNet + EMS + Cropped 能够学习一定的跨被试运动想象特征；
2. 平均准确率达到 46.70%，高于 25% 随机水平；
3. 不同被试间差异明显；
4. 部分被试准确率超过 64%，部分接近随机水平；
5. Macro-F1 低于 Accuracy，说明类别预测质量并不完全均衡；
6. 未来必须重点解决被试域偏移问题。

---

## 12. 后续优化方向

### 12.1 迁移学习

先训练通用模型，再使用少量新被试数据微调。

### 12.2 领域自适应

对齐训练被试和测试被试的特征分布。

### 12.3 被试级标准化

研究每名被试独立的归一化、协方差对齐或欧氏对齐。

### 12.4 黎曼几何

增加协方差矩阵和切空间分类基线。

### 12.5 EEGConformer

使用 CNN 提取局部模式，并通过 Transformer 建模长程依赖。

### 12.6 少样本校准

研究少量 trial 对新用户性能的提升幅度。

### 12.7 多随机种子实验

当前结果主要使用固定随机种子，应增加重复实验并报告置信区间。

---

## 13. 结果文件

```text
results/metrics/loso_summary.csv
results/metrics/loso_all_results.npz
results/metrics/loso_subject_*_result.npz
results/figures/loso_accuracy_subjects.png
results/figures/loso_confusion_matrix.png
```
