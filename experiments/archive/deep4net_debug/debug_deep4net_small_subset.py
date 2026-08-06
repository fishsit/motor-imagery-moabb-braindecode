"""
Deep4Net 小规模平衡数据过拟合测试。

目的：
    判断当前 Deep4Net + Dense Prediction + CroppedLoss
    训练链路是否具备正常学习能力。

测试方法：
    1. 加载 Subject 1 的 0train 会话；
    2. 每个类别随机选择 16 个样本；
    3. 共使用 64 个平衡样本；
    4. 关闭 Dropout 和权重衰减；
    5. 在同一批数据上训练和评价；
    6. 检查模型能否达到接近 100% 的训练准确率。

诊断标准：
    训练准确率 >= 95%：
        模型与损失流程基本正常，正式实验主要是优化配置或数据问题。

    训练准确率长期低于 80%：
        需要继续检查模型、CroppedLoss、预测聚合或输入数据。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from skorch.helper import predefined_split
from torch.utils.data import DataLoader, Subset

from braindecode import EEGClassifier
from braindecode.datasets import MOABBDataset
from braindecode.models import Deep4Net
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    exponential_moving_standardize,
    preprocess,
)
from braindecode.training import CroppedLoss
from braindecode.util import set_random_seeds


# ============================================================
# 1. 实验配置
# ============================================================

SUBJECT_ID = 1

N_CHANS = 22
N_CLASSES = 4
N_TIMES = 1000

LOW_CUT_HZ = 4.0
HIGH_CUT_HZ = 38.0

FACTOR_NEW = 1e-3
INIT_BLOCK_SIZE = 1000

SEED = 20260806

# 每个类别选择的样本数
SAMPLES_PER_CLASS = 16

# 16 × 4 = 64 个训练样本
EXPECTED_TOTAL_SAMPLES = (
    SAMPLES_PER_CLASS
    * N_CLASSES
)

MAX_EPOCHS = 100
BATCH_SIZE = 32

LEARNING_RATE = 1e-3

CLASS_NAMES = [
    "feet",
    "left_hand",
    "right_hand",
    "tongue",
]

RESULTS_DIR = Path("results/metrics")
MODELS_DIR = Path("models")

MODEL_PATH = (
    MODELS_DIR
    / "deep4net_small_subset_debug.pth"
)

RESULT_PATH = (
    RESULTS_DIR
    / "deep4net_small_subset_debug.txt"
)


# ============================================================
# 2. 数据辅助函数
# ============================================================

def scale_to_microvolts(
    data: np.ndarray,
) -> np.ndarray:
    """将 EEG 从伏特转换为微伏。"""
    return data * 1e6


def create_balanced_subset(
    dataset,
    samples_per_class: int,
    seed: int,
) -> Subset:
    """
    从数据集中为每个类别随机选择相同数量的样本。

    Parameters
    ----------
    dataset:
        原始训练数据集。

    samples_per_class:
        每个类别选择的样本数量。

    seed:
        随机种子。

    Returns
    -------
    Subset
        类别平衡的小规模训练集。
    """

    indices_by_class: dict[int, list[int]] = {
        class_id: []
        for class_id in range(N_CLASSES)
    }

    print("\nCollecting labels...")

    for sample_index in range(len(dataset)):
        _, target, _ = dataset[sample_index]

        target = int(target)

        if target not in indices_by_class:
            raise ValueError(
                f"Unexpected class label: {target}"
            )

        indices_by_class[target].append(
            sample_index
        )

    rng = np.random.default_rng(seed)

    selected_indices: list[int] = []

    for class_id in range(N_CLASSES):
        class_indices = np.asarray(
            indices_by_class[class_id],
            dtype=np.int64,
        )

        if len(class_indices) < samples_per_class:
            raise RuntimeError(
                f"Class {class_id} only has "
                f"{len(class_indices)} samples, "
                f"but {samples_per_class} are required."
            )

        rng.shuffle(class_indices)

        selected = class_indices[
            :samples_per_class
        ]

        selected_indices.extend(
            selected.tolist()
        )

        print(
            f"Class {class_id} "
            f"({CLASS_NAMES[class_id]}): "
            f"selected {len(selected)} samples"
        )

    # 再次打乱四类样本顺序
    rng.shuffle(selected_indices)

    subset = Subset(
        dataset,
        selected_indices,
    )

    if len(subset) != EXPECTED_TOTAL_SAMPLES:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_SAMPLES} samples, "
            f"but received {len(subset)}."
        )

    return subset


# ============================================================
# 3. 模型辅助函数
# ============================================================

def create_model() -> Deep4Net:
    """
    创建用于小样本记忆测试的 Deep4Net。

    与正式模型相比：
        drop_prob 从 0.5 改为 0.0。

    原因：
        本实验不是测试泛化能力，而是测试模型能否记住训练数据。
    """

    model = Deep4Net(
        n_chans=N_CHANS,
        n_outputs=N_CLASSES,
        n_times=N_TIMES,
        final_conv_length=2,

        # 小样本过拟合测试关闭 Dropout
        drop_prob=0.0,
    )

    model.to_dense_prediction_model()

    return model


def collect_predictions(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    在小规模训练集上进行预测。

    模型输出：
        batch × classes × temporal_predictions

    对时间预测维度求平均后，得到每个样本的最终预测。
    """

    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []

    model.eval()

    with torch.no_grad():
        for batch in data_loader:
            inputs = batch[0].to(
                device=device,
                dtype=torch.float32,
            )

            targets = batch[1].to(
                device=device,
                dtype=torch.long,
            )

            outputs = model(inputs)

            if outputs.ndim != 3:
                raise RuntimeError(
                    "Expected output shape "
                    "(batch, classes, predictions), "
                    f"but received {tuple(outputs.shape)}."
                )

            # 与 CroppedLoss 使用相同的时间预测聚合思路
            scores = outputs.mean(
                dim=2
            )

            predictions = torch.argmax(
                scores,
                dim=1,
            )

            all_targets.append(
                targets.cpu().numpy()
            )

            all_predictions.append(
                predictions.cpu().numpy()
            )

    y_true = np.concatenate(
        all_targets,
        axis=0,
    )

    y_pred = np.concatenate(
        all_predictions,
        axis=0,
    )

    return y_true, y_pred


# ============================================================
# 4. 主程序
# ============================================================

def main() -> None:
    """运行 Deep4Net 小样本过拟合测试。"""

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    use_cuda = torch.cuda.is_available()

    device = torch.device(
        "cuda" if use_cuda else "cpu"
    )

    set_random_seeds(
        seed=SEED,
        cuda=use_cuda,
    )

    print("=" * 72)
    print("Deep4Net Small Balanced Subset Overfit Test")
    print("=" * 72)

    print(f"Subject ID       : {SUBJECT_ID}")
    print(f"Device           : {device}")
    print(f"Samples/class    : {SAMPLES_PER_CLASS}")
    print(f"Total samples    : {EXPECTED_TOTAL_SAMPLES}")
    print(f"Maximum epochs   : {MAX_EPOCHS}")
    print(f"Learning rate    : {LEARNING_RATE}")
    print("Dropout          : 0.0")
    print("Weight decay     : 0.0")

    # ========================================================
    # 5. 加载数据
    # ========================================================

    print("\n[1/7] Loading BNCI2014_001...")

    dataset = MOABBDataset(
        dataset_name="BNCI2014_001",
        subject_ids=[SUBJECT_ID],
    )

    # ========================================================
    # 6. 预处理
    # ========================================================

    print("\n[2/7] Preprocessing EEG...")

    preprocessors = [
        Preprocessor(
            "pick",
            picks="eeg",
        ),

        Preprocessor(
            scale_to_microvolts,
        ),

        Preprocessor(
            "filter",
            l_freq=LOW_CUT_HZ,
            h_freq=HIGH_CUT_HZ,
        ),

        Preprocessor(
            exponential_moving_standardize,
            factor_new=FACTOR_NEW,
            init_block_size=INIT_BLOCK_SIZE,
        ),
    ]

    preprocess(
        dataset,
        preprocessors,
        n_jobs=1,
    )

    # ========================================================
    # 7. 创建模型
    # ========================================================

    print("\n[3/7] Creating Deep4Net...")

    model = create_model()

    output_shape = model.get_output_shape()

    n_preds_per_input = int(
        output_shape[2]
    )

    print(f"Model output     : {output_shape}")
    print(f"Predictions/input: {n_preds_per_input}")

    # ========================================================
    # 8. 创建窗口并获取训练会话
    # ========================================================

    print("\n[4/7] Creating windows...")

    windows_dataset = create_windows_from_events(
        dataset,

        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,

        window_size_samples=N_TIMES,

        window_stride_samples=n_preds_per_input,

        drop_last_window=False,
        preload=True,
    )

    split_datasets = windows_dataset.split(
        "session"
    )

    if "0train" not in split_datasets:
        raise KeyError(
            "Session '0train' was not found."
        )

    train_set = split_datasets["0train"]

    print(f"Full train set   : {len(train_set)}")

    # ========================================================
    # 9. 创建平衡小样本子集
    # ========================================================

    print("\n[5/7] Creating balanced subset...")

    small_subset = create_balanced_subset(
        dataset=train_set,
        samples_per_class=SAMPLES_PER_CLASS,
        seed=SEED,
    )

    selected_labels = []

    for sample_index in range(
        len(small_subset)
    ):
        _, target, _ = small_subset[
            sample_index
        ]

        selected_labels.append(
            int(target)
        )

    label_counts = Counter(
        selected_labels
    )

    print(
        "Selected class counts:",
        dict(label_counts),
    )

    # ========================================================
    # 10. 创建分类器
    # ========================================================

    print("\n[6/7] Creating EEGClassifier...")

    classifier = EEGClassifier(
        module=model,

        cropped=True,

        criterion=CroppedLoss,

        criterion__loss_function=(
            torch.nn.functional.cross_entropy
        ),

        # 小样本记忆测试使用 Adam
        optimizer=torch.optim.Adam,

        optimizer__lr=LEARNING_RATE,

        # 关闭权重衰减
        optimizer__weight_decay=0.0,

        # 不设置验证集
        train_split=None,

        batch_size=BATCH_SIZE,

        max_epochs=MAX_EPOCHS,

        iterator_train__shuffle=True,

        iterator_train__drop_last=False,

        callbacks=[
            "accuracy",
        ],

        device=device,

        classes=list(
            range(N_CLASSES)
        ),

        verbose=1,
    )

    # ========================================================
    # 11. 训练
    # ========================================================

    print("\n[7/7] Starting small-subset training...")

    classifier.fit(
        small_subset,
        y=None,
    )

    # ========================================================
    # 12. 在同一训练子集上评价
    # ========================================================

    y_true, y_pred = collect_predictions(
        model=classifier.module_,
        dataset=small_subset,
        device=device,
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    confusion = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(N_CLASSES)),
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(N_CLASSES)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    predicted_counts = np.bincount(
        y_pred,
        minlength=N_CLASSES,
    )

    torch.save(
        classifier.module_.state_dict(),
        MODEL_PATH,
    )

    result_text = (
        "Deep4Net Small Balanced Subset Overfit Test\n"
        "===========================================\n"
        f"Samples per class : {SAMPLES_PER_CLASS}\n"
        f"Total samples     : {len(y_true)}\n"
        f"Training accuracy : {accuracy:.4f}\n"
        f"Predicted counts  : {predicted_counts.tolist()}\n\n"
        "Classification report\n"
        "---------------------\n"
        f"{report}\n"
        "Confusion matrix\n"
        "----------------\n"
        f"{confusion}\n"
    )

    RESULT_PATH.write_text(
        result_text,
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("Small-subset test completed")
    print("=" * 72)

    print(f"Training accuracy : {accuracy:.4f}")

    print(
        "Predicted counts  :",
        predicted_counts.tolist(),
    )

    print("\nClassification report:")
    print(report)

    print("Confusion matrix:")
    print(confusion)

    print(f"\nModel saved to    : {MODEL_PATH}")
    print(f"Result saved to   : {RESULT_PATH}")

    if accuracy >= 0.95:
        print(
            "\nDiagnosis: PASS — the model can memorize "
            "the small balanced dataset."
        )

    elif accuracy >= 0.80:
        print(
            "\nDiagnosis: PARTIAL — the model learns the subset, "
            "but optimization can still be improved."
        )

    else:
        print(
            "\nDiagnosis: FAIL — the model cannot memorize the "
            "small balanced subset. Check the training pipeline."
        )


if __name__ == "__main__":
    main()