"""
Deep4Net Subject 1 训练集与验证集类别诊断。

目的：
    使用同一个最佳模型分别评价：

    1. 0train 训练会话
    2. 1test 验证会话

通过比较两个会话的混淆矩阵，判断：

    1. 模型训练时是否已经发生类别塌缩；
    2. tongue 类别是否在训练集中也无法识别；
    3. 问题主要来自模型优化，还是跨会话分布偏移。

输入：
    models/deep4net_subject1_best.pth

输出：
    results/metrics/deep4net_subject1_diagnostic_summary.csv

    results/metrics/deep4net_subject1_train_report.txt
    results/metrics/deep4net_subject1_valid_report.txt

    results/metrics/deep4net_subject1_train_predictions.csv
    results/metrics/deep4net_subject1_valid_predictions.csv

    results/figures/deep4net_subject1_train_confusion_matrix.png
    results/figures/deep4net_subject1_valid_confusion_matrix.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# 防止无图形界面的环境中绘图报错
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from braindecode.datasets import MOABBDataset
from braindecode.models import Deep4Net
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    exponential_moving_standardize,
    preprocess,
)
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
BATCH_SIZE = 32

CLASS_NAMES = [
    "feet",
    "left_hand",
    "right_hand",
    "tongue",
]

MODELS_DIR = Path("models")
METRICS_DIR = Path("results/metrics")
FIGURES_DIR = Path("results/figures")

BEST_MODEL_PATH = (
    MODELS_DIR
    / "deep4net_subject1_best.pth"
)

SUMMARY_PATH = (
    METRICS_DIR
    / "deep4net_subject1_diagnostic_summary.csv"
)


# ============================================================
# 2. 数据处理与模型辅助函数
# ============================================================

def scale_to_microvolts(
    data: np.ndarray,
) -> np.ndarray:
    """将 EEG 从伏特转换为微伏。"""
    return data * 1e6


def create_model() -> Deep4Net:
    """创建与训练阶段完全一致的 Deep4Net。"""

    model = Deep4Net(
        n_chans=N_CHANS,
        n_outputs=N_CLASSES,
        n_times=N_TIMES,
        final_conv_length=2,
        drop_prob=0.5,
    )

    model.to_dense_prediction_model()

    return model


def load_model_state(
    path: Path,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """加载最佳模型参数。"""

    try:
        state_dict = torch.load(
            path,
            map_location=device,
            weights_only=True,
        )

    except TypeError:
        # 兼容不支持 weights_only 的旧版 PyTorch
        state_dict = torch.load(
            path,
            map_location=device,
        )

    if not isinstance(state_dict, dict):
        raise TypeError(
            "Loaded model state is not a dictionary."
        )

    return state_dict


# ============================================================
# 3. 模型预测
# ============================================================

def collect_predictions(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    对一个数据集进行预测。

    Dense Prediction 输出形状：

        batch × classes × predictions

    对最后的时间预测维度求平均，得到每个窗口的类别分数。
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
    all_scores: list[np.ndarray] = []

    model.eval()

    with torch.no_grad():

        for batch in data_loader:

            if len(batch) < 2:
                raise RuntimeError(
                    "Unexpected dataset batch format."
                )

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
                    "Expected model output shape "
                    "(batch, classes, predictions), "
                    f"but received {tuple(outputs.shape)}."
                )

            # 聚合同一窗口中的多个时间位置预测
            window_scores = outputs.mean(
                dim=2
            )

            predictions = torch.argmax(
                window_scores,
                dim=1,
            )

            all_targets.append(
                targets.cpu().numpy()
            )

            all_predictions.append(
                predictions.cpu().numpy()
            )

            all_scores.append(
                window_scores.cpu().numpy()
            )

    if not all_targets:
        raise RuntimeError(
            "No predictions were generated."
        )

    y_true = np.concatenate(
        all_targets,
        axis=0,
    )

    y_pred = np.concatenate(
        all_predictions,
        axis=0,
    )

    scores = np.concatenate(
        all_scores,
        axis=0,
    )

    return y_true, y_pred, scores


# ============================================================
# 4. 混淆矩阵绘图
# ============================================================

def save_confusion_matrix(
    confusion: np.ndarray,
    split_name: str,
    output_path: Path,
) -> None:
    """保存训练集或验证集混淆矩阵。"""

    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion,
        display_labels=CLASS_NAMES,
    )

    figure, axis = plt.subplots(
        figsize=(8, 7)
    )

    display.plot(
        ax=axis,
        values_format="d",
        colorbar=True,
    )

    axis.set_title(
        f"Deep4Net Subject 1 - {split_name}"
    )

    axis.set_xlabel(
        "Predicted label"
    )

    axis.set_ylabel(
        "True label"
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# 5. 单个数据划分评价
# ============================================================

def evaluate_split(
    split_key: str,
    display_name: str,
    dataset,
    model: torch.nn.Module,
    device: torch.device,
) -> dict[str, object]:
    """评价一个会话并保存结果。"""

    print("\n" + "=" * 72)
    print(f"Evaluating: {display_name}")
    print("=" * 72)

    y_true, y_pred, scores = collect_predictions(
        model=model,
        dataset=dataset,
        device=device,
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            y_pred,
        )
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
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

    true_counts = np.bincount(
        y_true,
        minlength=N_CLASSES,
    )

    predicted_counts = np.bincount(
        y_pred,
        minlength=N_CLASSES,
    )

    print(f"Samples           : {len(y_true)}")
    print(f"Accuracy          : {accuracy:.4f}")
    print(
        "Balanced accuracy : "
        f"{balanced_accuracy:.4f}"
    )
    print(f"Macro-F1          : {macro_f1:.4f}")
    print(f"Weighted-F1       : {weighted_f1:.4f}")

    print("\nTrue class counts:")

    for class_name, count in zip(
        CLASS_NAMES,
        true_counts,
    ):
        print(
            f"  {class_name:<12}: {int(count)}"
        )

    print("\nPredicted class counts:")

    for class_name, count in zip(
        CLASS_NAMES,
        predicted_counts,
    ):
        percentage = (
            float(count)
            / len(y_pred)
            * 100.0
        )

        print(
            f"  {class_name:<12}: "
            f"{int(count):>3} "
            f"({percentage:6.2f}%)"
        )

    print("\nClassification report:")
    print(report)

    print("Confusion matrix:")
    print(confusion)

    report_path = (
        METRICS_DIR
        / f"deep4net_subject1_{split_key}_report.txt"
    )

    predictions_path = (
        METRICS_DIR
        / f"deep4net_subject1_{split_key}_predictions.csv"
    )

    confusion_path = (
        FIGURES_DIR
        / (
            "deep4net_subject1_"
            f"{split_key}_confusion_matrix.png"
        )
    )

    report_content = (
        f"Deep4Net Subject 1 - {display_name}\n"
        f"{'=' * 50}\n"
        f"Samples           : {len(y_true)}\n"
        f"Accuracy          : {accuracy:.4f}\n"
        f"Balanced accuracy : {balanced_accuracy:.4f}\n"
        f"Macro-F1          : {macro_f1:.4f}\n"
        f"Weighted-F1       : {weighted_f1:.4f}\n\n"
        "Classification report\n"
        "---------------------\n"
        f"{report}\n"
        "Confusion matrix\n"
        "----------------\n"
        f"{confusion}\n\n"
        "True class counts\n"
        "-----------------\n"
        f"{dict(zip(CLASS_NAMES, true_counts.tolist()))}\n\n"
        "Predicted class counts\n"
        "----------------------\n"
        f"{dict(zip(CLASS_NAMES, predicted_counts.tolist()))}\n"
    )

    report_path.write_text(
        report_content,
        encoding="utf-8",
    )

    predictions_df = pd.DataFrame(
        {
            "sample_index": np.arange(
                len(y_true)
            ),
            "true_label": y_true,
            "true_class": [
                CLASS_NAMES[label]
                for label in y_true
            ],
            "predicted_label": y_pred,
            "predicted_class": [
                CLASS_NAMES[label]
                for label in y_pred
            ],
            "score_feet": scores[:, 0],
            "score_left_hand": scores[:, 1],
            "score_right_hand": scores[:, 2],
            "score_tongue": scores[:, 3],
            "correct": y_true == y_pred,
        }
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
        encoding="utf-8-sig",
    )

    save_confusion_matrix(
        confusion=confusion,
        split_name=display_name,
        output_path=confusion_path,
    )

    print(f"\nReport saved      : {report_path}")
    print(f"Predictions saved : {predictions_path}")
    print(f"Confusion figure  : {confusion_path}")

    return {
        "split": split_key,
        "display_name": display_name,
        "n_samples": len(y_true),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "predicted_feet": int(
            predicted_counts[0]
        ),
        "predicted_left_hand": int(
            predicted_counts[1]
        ),
        "predicted_right_hand": int(
            predicted_counts[2]
        ),
        "predicted_tongue": int(
            predicted_counts[3]
        ),
    }


# ============================================================
# 6. 主程序
# ============================================================

def main() -> None:
    """运行训练集与验证集双重诊断。"""

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Best model file was not found: "
            f"{BEST_MODEL_PATH}"
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
    print("Deep4Net Subject 1 Train/Validation Diagnostic")
    print("=" * 72)

    print(f"Device           : {device}")
    print(f"Best model       : {BEST_MODEL_PATH}")

    # ========================================================
    # 7. 数据加载与预处理
    # ========================================================

    print("\n[1/5] Loading BNCI2014_001...")

    dataset = MOABBDataset(
        dataset_name="BNCI2014_001",
        subject_ids=[SUBJECT_ID],
    )

    print("\n[2/5] Preprocessing EEG...")

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

    n_chans = int(
        dataset.datasets[0].raw.info["nchan"]
    )

    if n_chans != N_CHANS:
        raise RuntimeError(
            f"Expected {N_CHANS} EEG channels, "
            f"but received {n_chans}."
        )

    # ========================================================
    # 8. 加载最佳模型
    # ========================================================

    print("\n[3/5] Loading best Deep4Net...")

    model = create_model()

    output_shape = model.get_output_shape()

    n_preds_per_input = int(
        output_shape[2]
    )

    state_dict = load_model_state(
        BEST_MODEL_PATH,
        device,
    )

    model.load_state_dict(
        state_dict
    )

    model.to(device)

    print(f"Model output     : {output_shape}")
    print(f"Predictions/input: {n_preds_per_input}")

    # ========================================================
    # 9. 创建窗口并划分会话
    # ========================================================

    print("\n[4/5] Creating windows...")

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

    required_sessions = [
        "0train",
        "1test",
    ]

    for session_name in required_sessions:

        if session_name not in split_datasets:
            raise KeyError(
                f"Session '{session_name}' was not found."
            )

    train_set = split_datasets["0train"]
    valid_set = split_datasets["1test"]

    print(f"Train windows    : {len(train_set)}")
    print(f"Valid windows    : {len(valid_set)}")

    # ========================================================
    # 10. 分别评价训练会话和验证会话
    # ========================================================

    print("\n[5/5] Evaluating both sessions...")

    train_result = evaluate_split(
        split_key="train",
        display_name="0train",
        dataset=train_set,
        model=model,
        device=device,
    )

    valid_result = evaluate_split(
        split_key="valid",
        display_name="1test",
        dataset=valid_set,
        model=model,
        device=device,
    )

    summary_df = pd.DataFrame(
        [
            train_result,
            valid_result,
        ]
    )

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 72)
    print("Diagnostic completed")
    print("=" * 72)

    print(summary_df.to_string(index=False))

    print(
        f"\nSummary saved to : {SUMMARY_PATH}"
    )

    print("\nDiagnostic interpretation:")

    print(
        "1. If tongue recall is also near zero on 0train, "
        "the model failed to learn this class during training."
    )

    print(
        "2. If tongue is learned on 0train but fails on 1test, "
        "the main issue is cross-session distribution shift."
    )

    print(
        "3. If all training classes perform well but validation "
        "performance drops, the model is overfitting."
    )

    print("\nResult: PASS")


if __name__ == "__main__":
    main()