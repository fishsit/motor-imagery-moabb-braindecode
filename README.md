# 基于 MOABB 与 Braindecode 的运动想象脑电分类

本项目基于公开运动想象脑电数据集 **BNCI2014_001**，使用 **MOABB、MNE、Braindecode、Scikit-learn 和 PyTorch**，逐步复现运动想象脑机接口的数据读取、信号预处理、传统机器学习分类、跨被试验证及后续深度学习实验流程。

当前已完成传统机器学习基线阶段，建立了从真实脑电数据到分类结果的完整处理链路。

## 一、项目目标

本项目主要完成以下任务：

- 使用 MOABB 自动下载和管理公开脑电数据集；
- 使用 Braindecode 和 MNE 读取真实运动想象脑电数据；
- 分析被试、会话、运行、通道、采样率和事件标签；
- 将连续脑电信号切分为独立运动想象试次；
- 完成脑电通道选择和 8～30 Hz 带通滤波；
- 使用共同空间模式（CSP）提取空间特征；
- 使用支持向量机（SVM）完成四分类运动想象识别；
- 完成单被试、多被试和留一被试交叉验证实验；
- 后续使用 EEGNet、ShallowFBCSPNet 和 Deep4Net 进行深度学习实验。

## 二、数据集

项目使用 **BNCI2014_001** 数据集，也称为 **BCI Competition IV 2a**。

| 项目 | 数据 |
| --- | --- |
| 被试数量 | 9 |
| 会话数量 | 每名被试 2 次 |
| 运动想象类别 | 4 类 |
| 脑电通道 | 22 个 |
| 眼电通道 | 3 个 |
| 采样率 | 250 Hz |
| 数据范式 | 运动想象 |

四种运动想象类别为：

- 左手；
- 右手；
- 双脚；
- 舌头。

MOABB 在首次运行时会自动下载数据。未单独设置数据目录时，数据通常保存在用户目录的 MNE 数据文件夹中，不会直接写入本项目仓库。

## 三、技术栈

- Python
- NumPy
- SciPy
- MNE-Python
- MOABB
- Braindecode
- Scikit-learn
- PyTorch

## 四、算法流程

```text
BNCI2014_001 真实脑电数据
            ↓
       MOABB 自动下载
            ↓
 Braindecode / MNE 读取数据
            ↓
 解析被试、会话、运行和事件
            ↓
      连续脑电切分为 Epoch
            ↓
       保留 22 个 EEG 通道
            ↓
        8～30 Hz 带通滤波
            ↓
       CSP 空间特征提取
            ↓
          SVM 分类
            ↓
 单被试 / 多被试 / 跨被试评估
```

每个运动想象试次最终形成如下三维数据：

```text
样本数量 × 脑电通道数量 × 时间采样点数量
```

例如，单个运行中的数据形状为：

```text
48 × 22 × 1001
```

其中：

- 48 表示 48 个运动想象试次；
- 22 表示 22 个脑电通道；
- 1001 表示截取 0～4 秒并包含两个端点后的采样点数量。

## 五、项目结构

```text
motor-imagery-moabb-braindecode
├── README.md
├── requirements.txt
├── configs
│   └── .gitkeep
├── notebooks
│   └── .gitkeep
├── scripts
│   ├── 01_check_environment.py
│   ├── 02_load_bnci2014_001.py
│   ├── 03_create_epochs.py
│   ├── 04_preprocess_eeg.py
│   ├── 05_csp_svm.py
│   ├── 06_csp_svm_multiclass.py
│   ├── 07_mult_subject_csp_svm.py
│   ├── 08_loso_csp_svm.py
│   └── 09_loso_full_bnci.py
└── results
    ├── figures
    │   └── .gitkeep
    ├── metrics
    │   └── .gitkeep
    └── models
        └── .gitkeep
```

## 六、脚本说明

| 脚本 | 作用 |
| --- | --- |
| `01_check_environment.py` | 检查 Python、PyTorch、MNE、MOABB 和 Braindecode 环境 |
| `02_load_bnci2014_001.py` | 下载并读取被试 1，查看会话、运行、通道、采样率和事件标签 |
| `03_create_epochs.py` | 根据事件标注将连续脑电切分为运动想象试次 |
| `04_preprocess_eeg.py` | 保留脑电通道，完成 8～30 Hz 滤波和数据标准化 |
| `05_csp_svm.py` | 完成左手与右手二分类 CSP-SVM 实验 |
| `06_csp_svm_multiclass.py` | 完成单被试四分类 CSP-SVM 实验 |
| `07_mult_subject_csp_svm.py` | 合并 9 名被试的部分数据进行多被试随机划分实验 |
| `08_loso_csp_svm.py` | 使用部分数据完成留一被试交叉验证 |
| `09_loso_full_bnci.py` | 使用全部会话和运行完成留一被试交叉验证 |

## 七、环境安装

### 1. 创建虚拟环境

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. 安装依赖

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. 检查环境

```powershell
python .\scripts\01_check_environment.py
```

## 八、运行方法

### 1. 读取真实脑电数据

```powershell
python .\scripts\02_load_bnci2014_001.py
```

### 2. 创建运动想象试次

```powershell
python .\scripts\03_create_epochs.py
```

### 3. 完成脑电预处理

```powershell
python .\scripts\04_preprocess_eeg.py
```

### 4. 运行单被试四分类实验

```powershell
python .\scripts\06_csp_svm_multiclass.py
```

### 5. 运行多被试实验

```powershell
python .\scripts\07_mult_subject_csp_svm.py
```

### 6. 运行留一被试交叉验证

```powershell
python .\scripts\08_loso_csp_svm.py
```

### 7. 使用完整会话和运行进行留一被试验证

```powershell
python .\scripts\09_loso_full_bnci.py
```

## 九、当前实验结果

| 实验 | 数据范围 | 模型 | 准确率 | 标准差 |
| --- | --- | --- | ---: | ---: |
| 单被试四分类 | 被试 1 的单个运行 | CSP + SVM | 66.67% | — |
| 多被试随机划分 | 9 名被试，每人单个运行 | CSP + SVM | 44.44% | — |
| 留一被试验证 | 9 名被试，每人单个运行 | CSP + SVM | 38.89% | 13.29% |
| 完整留一被试验证 | 9 名被试的全部会话和运行 | CSP + SVM | 36.59% | 10.77% |

四分类任务的随机猜测准确率为 25%。目前各项实验结果均高于随机水平，说明模型能够从脑电数据中提取部分运动想象信息。

不过需要注意：

- 单被试随机划分实验的数据量较小，准确率波动较大；
- 多被试随机划分中，训练集和测试集可能包含相同被试的数据；
- 留一被试验证更接近新用户直接使用模型的实际场景；
- 当前结果用于验证处理流程和建立传统算法基线，不代表经过充分调参后的最优性能。

## 十、阶段性结论

目前实验呈现出以下趋势：

```text
单被试分类准确率
        >
多被试随机划分准确率
        >
跨被试分类准确率
```

这说明不同被试之间存在明显的脑电分布差异。传统 CSP-SVM 能够完成基础运动想象分类，但跨被试泛化能力仍然有限。

这一问题也是运动想象脑机接口中的核心研究方向，通常需要通过以下方法进一步改善：

- 更合理的时间窗口和频带选择；
- 滤波器组共同空间模式；
- 黎曼几何分类；
- 迁移学习；
- 领域自适应；
- 深度学习；
- 自监督脑电表征学习。

## 十一、后续计划

下一阶段将进入 Braindecode 深度学习实验。

计划依次完成：

- [ ] 将 MNE 数据转换为 Braindecode 窗口数据集；
- [ ] 构建 PyTorch 数据加载器；
- [ ] 训练 EEGNet；
- [ ] 训练 ShallowFBCSPNet；
- [ ] 训练 Deep4Net；
- [ ] 保存模型、损失和准确率；
- [ ] 绘制训练曲线与混淆矩阵；
- [ ] 对比传统 CSP-SVM 与深度学习模型；
- [ ] 探索跨被试迁移学习和领域自适应；
- [ ] 探索实时脑电数据接入与在线推理。

## 十二、项目状态

当前版本完成了传统运动想象脑电分类基线：

- [x] 项目环境搭建；
- [x] 真实脑电数据自动下载；
- [x] 数据结构检查；
- [x] 事件解析；
- [x] Epoch 切分；
- [x] 通道选择；
- [x] 带通滤波；
- [x] CSP 特征提取；
- [x] SVM 二分类；
- [x] SVM 四分类；
- [x] 多被试实验；
- [x] 留一被试交叉验证；
- [x] 完整会话和运行实验；
- [ ] Braindecode 深度学习；
- [ ] 模型保存和实验结果可视化；
- [ ] 实时推理。

## 十三、说明

本项目主要用于学习和复现运动想象脑电分类的标准数据处理与建模流程。

实验结果会受到数据划分、随机种子、预处理方法、时间窗口、频带、模型参数和软件版本等因素影响。后续将逐步补充配置文件、结果保存、可视化和可重复实验设置。
