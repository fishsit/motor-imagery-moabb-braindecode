"""
Deep4Net Subject 1 低正则化最佳模型独立评价。

本脚本执行：
1. 加载 BNCI2014_001 Subject 1；
2. 使用与低正则化训练完全一致的预处理；
3. 创建 Dropout=0.25 的 Deep4Net；
4. 加载验证准确率最高的模型参数；
5. 在 Subject 1 的 1test 会话上独立评价；
6. 计算 Accuracy、Balanced Accuracy、Macro-F1；
7. 输出分类报告、预测数量和混淆矩阵；
8. 绘制准确率曲线、损失曲线和混淆矩阵。

输入文件：
    models/deep4net_subject1_low_regularization_best.pth
    results/metrics/deep4net_subject1_low_regularization_history.csv

输出文件：
    results/metrics/
        deep4net_subject1_low_regularization_evaluation.csv
        deep4net_subject1_low_regularization_classification_report.txt
        deep4net_subject1_low_regularization_predictions.csv

    results/figures/
        deep4net_subject1_low_regularization_confusion_matrix.png
        deep4net_subject1_low_regularization_accuracy.png
        deep4net_subject1_low_regularization_loss.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# 防止无图形界面环境中绘图报错
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
# 1. 实验参数
# ============================================================

SUBJECT_ID = 1

N_CHANS = 22
N_CLASSES = 4

# 250 Hz × 4 秒
N_TIMES = 1000

LOW_CUT_HZ = 4.0
HIGH_CUT_HZ = 38.0

FACTOR_NEW = 1e-3
INIT_BLOCK_SIZE = 1000

SEED = 20260806

BATCH_SIZE = 32

# 必须与低正则化训练模型保持一致
DROP_PROBABILITY = 0.25

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
    / "deep4net_subject1_low_regularization_best.pth"
)

HISTORY_PATH = (
    METRICS_DIR
    / "deep4net_subject1_low_regularization_history.csv"
)

EVALUATION_PATH = (
    METRICS_DIR
    / "deep4net_subject1_low_regularization_evaluation.csv"
)

REPORT_PATH = (
    METRICS_DIR
    / "deep4net_subject1_low_regularization_classification_report.txt"
)

PREDICTIONS_PATH = (
    METRICS_DIR
    / "deep4net_subject1_low_regularization_predictions.csv"
)

CONFUSION_MATRIX_PATH = (
    FIGURES_DIR
    / "deep4net_subject1_low_regularization_confusion_matrix.png"
)

ACCURACY_CURVE_PATH = (
    FIGURES_DIR
    / "deep4net_subject1_low_regularization_accuracy.png"
)

LOSS_CURVE_PATH = (
    FIGURES_DIR
    / "deep4net_subject1_low_regularization_loss.png"
)


# ============================================================
# 2. 数据与模型辅助函数
# ============================================================

def scale_to_microvolts(
    data: np.ndarray,
) -> np.ndarray:
    """将 EEG 信号从伏特转换为微伏。"""
    return data * 1e6


def create_model() -> Deep4Net:
    """
    创建与低正则化训练阶段完全一致的 Deep4Net。

    关键参数：
        final_conv_length = 2
        drop_prob = 0.25
    """

    model = Deep4Net(
        n_chans=N_CHANS,
        n_outputs=N_CLASSES,
        n_times=N_TIMES,
        final_conv_length=2,
        drop_prob=DROP_PROBABILITY,
    )

    model.to_dense_prediction_model()

    return model


def load_model_state(
    path: Path,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    加载模型参数。

    优先使用 weights_only=True，兼容不支持该参数的旧版 PyTorch。
    """

    try:
        state_dict = torch.load(
            path,
            map_location=device,
            weights_only=True,
        )

    except TypeError:
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
    对验证集进行预测。

    Dense Prediction 模型输出形状：

        batch × classes × temporal_predictions

    对 temporal_predictions 维度求平均，
    得到每个窗口对应的最终四分类分数。
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
                    "Unexpected validation batch format."
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
                    "Expected output shape "
                    "(batch, classes, predictions), "
                    f"but received {tuple(outputs.shape)}."
                )

            # 对密集时间预测求平均
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
# 4. 绘制混淆矩阵
# ============================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """绘制并保存四分类混淆矩阵。"""

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(N_CLASSES)),
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
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
        "Deep4Net Subject 1 Low-Regularization\n"
        "Confusion Matrix"
    )

    axis.set_xlabel(
        "Predicted label"
    )

    axis.set_ylabel(
        "True label"
    )

    figure.tight_layout()

    figure.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return cm


# ============================================================
# 5. 绘制准确率曲线
# ============================================================

def plot_accuracy_curve(
    history_df: pd.DataFrame,
) -> tuple[int, float]:
    """绘制训练准确率和验证准确率曲线。"""

    required_columns = {
        "epoch",
        "train_accuracy",
        "valid_accuracy",
    }

    missing_columns = (
        required_columns
        - set(history_df.columns)
    )

    if missing_columns:
        raise KeyError(
            "History CSV is missing columns: "
            f"{sorted(missing_columns)}"
        )

    best_index = (
        history_df["valid_accuracy"]
        .astype(float)
        .idxmax()
    )

    best_epoch = int(
        history_df.loc[
            best_index,
            "epoch",
        ]
    )

    best_valid_accuracy = float(
        history_df.loc[
            best_index,
            "valid_accuracy",
        ]
    )

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    axis.plot(
        history_df["epoch"],
        history_df["train_accuracy"],
        marker="o",
        label="Train accuracy",
    )

    axis.plot(
        history_df["epoch"],
        history_df["valid_accuracy"],
        marker="o",
        label="Validation accuracy",
    )

    axis.axvline(
        best_epoch,
        linestyle="--",
        label=(
            f"Best epoch = {best_epoch}, "
            f"valid acc = {best_valid_accuracy:.4f}"
        ),
    )

    axis.axhline(
        0.25,
        linestyle=":",
        label="Chance level = 0.25",
    )

    axis.set_title(
        "Deep4Net Subject 1 Low-Regularization Accuracy"
    )

    axis.set_xlabel(
        "Epoch"
    )

    axis.set_ylabel(
        "Accuracy"
    )

    axis.set_ylim(
        0.0,
        1.0,
    )

    axis.grid(
        True,
        alpha=0.3,
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        ACCURACY_CURVE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return best_epoch, best_valid_accuracy


# ============================================================
# 6. 绘制损失曲线
# ============================================================

def plot_loss_curve(
    history_df: pd.DataFrame,
) -> None:
    """绘制训练损失和验证损失曲线。"""

    required_columns = {
        "epoch",
        "train_loss",
        "valid_loss",
    }

    missing_columns = (
        required_columns
        - set(history_df.columns)
    )

    if missing_columns:
        raise KeyError(
            "History CSV is missing columns: "
            f"{sorted(missing_columns)}"
        )

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    axis.plot(
        history_df["epoch"],
        history_df["train_loss"],
        marker="o",
        label="Train loss",
    )

    axis.plot(
        history_df["epoch"],
        history_df["valid_loss"],
        marker="o",
        label="Validation loss",
    )

    axis.set_title(
        "Deep4Net Subject 1 Low-Regularization Loss"
    )

    axis.set_xlabel(
        "Epoch"
    )

    axis.set_ylabel(
        "Loss"
    )

    axis.grid(
        True,
        alpha=0.3,
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        LOSS_CURVE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# 7. 主程序
# ============================================================

def main() -> None:
    """运行低正则化最佳模型独立评价。"""

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

    if not HISTORY_PATH.exists():
        raise FileNotFoundError(
            "Training history file was not found: "
            f"{HISTORY_PATH}"
        )

    use_cuda = torch.cuda.is_available()

    device = torch.device(
        "cuda" if use_cuda else "cpu"
    )

    set_random_seeds(
        seed=SEED,
        cuda=use_cuda,
    )

    print("=" * 76)
    print("Deep4Net Subject 1 Low-Regularization Evaluation")
    print("=" * 76)

    print(f"Device           : {device}")
    print(f"Model path       : {BEST_MODEL_PATH}")
    print(f"Dropout          : {DROP_PROBABILITY}")
    print(f"Validation data  : Subject 1 / 1test")

    # ========================================================
    # 8. 加载数据
    # ========================================================

    print("\n[1/7] Loading BNCI2014_001...")

    dataset = MOABBDataset(
        dataset_name="BNCI2014_001",
        subject_ids=[SUBJECT_ID],
    )

    # ========================================================
    # 9. 数据预处理
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

    n_chans = int(
        dataset.datasets[0].raw.info["nchan"]
    )

    if n_chans != N_CHANS:
        raise RuntimeError(
            f"Expected {N_CHANS} channels, "
            f"but received {n_chans}."
        )

    # ========================================================
    # 10. 创建模型并加载参数
    # ========================================================

    print("\n[3/7] Loading best low-regularization model...")

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
    # 11. 创建验证窗口
    # ========================================================

    print("\n[4/7] Creating validation windows...")

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

    if "1test" not in split_datasets:
        raise KeyError(
            "Session '1test' was not found."
        )

    valid_set = split_datasets["1test"]

    print(
        f"Validation windows: {len(valid_set)}"
    )

    if len(valid_set) != 288:
        print(
            "Warning: expected 288 validation windows, "
            f"but received {len(valid_set)}."
        )

    # ========================================================
    # 12. 执行独立预测
    # ========================================================

    print("\n[5/7] Predicting validation set...")

    y_true, y_pred, scores = collect_predictions(
        model=model,
        dataset=valid_set,
        device=device,
    )

    print(f"Targets shape    : {y_true.shape}")
    print(f"Predictions shape: {y_pred.shape}")
    print(f"Scores shape     : {scores.shape}")

    # ========================================================
    # 13. 计算评价指标
    # ========================================================

    print("\n[6/7] Calculating metrics...")

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

    true_counts = np.bincount(
        y_true,
        minlength=N_CLASSES,
    )

    print(
        f"Accuracy          : {accuracy:.4f}"
    )

    print(
        "Balanced accuracy : "
        f"{balanced_accuracy:.4f}"
    )

    print(
        f"Macro-F1          : {macro_f1:.4f}"
    )

    print(
        f"Weighted-F1       : {weighted_f1:.4f}"
    )

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

    # ========================================================
    # 14. 保存评价结果
    # ========================================================

    print("\n[7/7] Saving evaluation results...")

    evaluation_df = pd.DataFrame(
        [
            {
                "subject": SUBJECT_ID,
                "train_session": "0train",
                "test_session": "1test",
                "dropout": DROP_PROBABILITY,
                "optimizer": "Adam",
                "weight_decay": 0.0,
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
        ]
    )

    evaluation_df.to_csv(
        EVALUATION_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    report_content = (
        "Deep4Net Subject 1 Low-Regularization Evaluation\n"
        "================================================\n"
        f"Accuracy          : {accuracy:.4f}\n"
        f"Balanced accuracy : {balanced_accuracy:.4f}\n"
        f"Macro-F1          : {macro_f1:.4f}\n"
        f"Weighted-F1       : {weighted_f1:.4f}\n\n"
        "True class counts\n"
        "-----------------\n"
        f"{dict(zip(CLASS_NAMES, true_counts.tolist()))}\n\n"
        "Predicted class counts\n"
        "----------------------\n"
        f"{dict(zip(CLASS_NAMES, predicted_counts.tolist()))}\n\n"
        "Classification report\n"
        "---------------------\n"
        f"{report}\n"
    )

    REPORT_PATH.write_text(
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
        PREDICTIONS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    cm = plot_confusion_matrix(
        y_true,
        y_pred,
    )

    history_df = pd.read_csv(
        HISTORY_PATH
    )

    best_epoch, best_history_accuracy = (
        plot_accuracy_curve(
            history_df
        )
    )

    plot_loss_curve(
        history_df
    )

    print("\nConfusion matrix:")
    print(cm)

    print("\n" + "=" * 76)
    print("Evaluation completed")
    print("=" * 76)

    print(
        f"Accuracy          : {accuracy:.4f}"
    )

    print(
        "Balanced accuracy : "
        f"{balanced_accuracy:.4f}"
    )

    print(
        f"Macro-F1          : {macro_f1:.4f}"
    )

    print(
        f"Weighted-F1       : {weighted_f1:.4f}"
    )

    print(
        f"Best history epoch: {best_epoch}"
    )

    print(
        "History valid acc : "
        f"{best_history_accuracy:.4f}"
    )

    print(
        f"Predicted counts  : {predicted_counts.tolist()}"
    )

    print(
        f"Metrics saved to  : {EVALUATION_PATH}"
    )

    print(
        f"Report saved to   : {REPORT_PATH}"
    )

    print(
        f"Predictions saved : {PREDICTIONS_PATH}"
    )

    print(
        f"Confusion matrix  : {CONFUSION_MATRIX_PATH}"
    )

    print(
        f"Accuracy curve    : {ACCURACY_CURVE_PATH}"
    )

    print(
        f"Loss curve        : {LOSS_CURVE_PATH}"
    )

    print("\nResult: PASS")


if __name__ == "__main__":
    main()