"""
Deep4Net 单被试无测试集泄漏实验。

正确实验流程
------------
第一阶段：模型选择
    使用 0train 会话中的前 5 个 run 训练；
    使用最后 1 个 run 作为内部验证集；
    根据内部验证准确率选择最佳 Epoch。

第二阶段：最终训练
    重新初始化一个全新的 Deep4Net；
    使用完整的 0train 会话训练；
    训练轮数固定为第一阶段选出的最佳 Epoch。

第三阶段：最终测试
    只在训练全部结束后评价一次 1test；
    1test 不参与 Early Stopping、模型保存或超参数选择。

推荐首先运行：
    python scripts/04_cropped_training/train_deep4net_unbiased_subject.py --subject 2

说明：
    Subject 1 的 1test 已经被用于调参，因此不再是完全未见测试集。
    Subject 2 可以用于验证当前低正则化配置的真实泛化能力。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

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
from skorch.callbacks import Checkpoint, EarlyStopping
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
# 1. 固定实验参数
# ============================================================

N_CHANS = 22
N_CLASSES = 4
N_TIMES = 1000

LOW_CUT_HZ = 4.0
HIGH_CUT_HZ = 38.0

FACTOR_NEW = 1e-3
INIT_BLOCK_SIZE = 1000

SEED = 20260806

MAX_SELECTION_EPOCHS = 60
EARLY_STOPPING_PATIENCE = 10

BATCH_SIZE = 32
LEARNING_RATE = 1e-3

# 使用已经确认有效的低正则化配置
DROP_PROBABILITY = 0.25
WEIGHT_DECAY = 0.0

CLASS_NAMES = [
    "feet",
    "left_hand",
    "right_hand",
    "tongue",
]

MODELS_DIR = Path("models")
METRICS_DIR = Path("results/metrics")
FIGURES_DIR = Path("results/figures")


# ============================================================
# 2. 命令行参数
# ============================================================

def parse_arguments() -> argparse.Namespace:
    """读取被试编号。"""

    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate Deep4Net using a run-wise "
            "internal validation protocol."
        )
    )

    parser.add_argument(
        "--subject",
        type=int,
        required=True,
        choices=range(1, 10),
        metavar="1-9",
        help="BNCI2014_001 subject ID.",
    )

    return parser.parse_args()


# ============================================================
# 3. 通用辅助函数
# ============================================================

def scale_to_microvolts(
    data: np.ndarray,
) -> np.ndarray:
    """将 EEG 信号从伏特转换为微伏。"""

    return data * 1e6


def create_model() -> Deep4Net:
    """创建低正则化 Deep4Net。"""

    model = Deep4Net(
        n_chans=N_CHANS,
        n_outputs=N_CLASSES,
        n_times=N_TIMES,
        final_conv_length=2,
        drop_prob=DROP_PROBABILITY,
    )

    model.to_dense_prediction_model()

    return model


def count_trainable_parameters(
    model: torch.nn.Module,
) -> int:
    """统计模型可训练参数数量。"""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def history_to_dataframe(
    classifier: EEGClassifier,
) -> pd.DataFrame:
    """将 skorch History 转换成普通 DataFrame。"""

    rows: list[dict[str, object]] = []

    for record in classifier.history:
        row: dict[str, object] = {}

        for key, value in record.items():
            if key == "batches":
                continue

            if isinstance(value, np.generic):
                value = value.item()

            if isinstance(
                value,
                (str, int, float, bool),
            ) or value is None:
                row[key] = value

        rows.append(row)

    return pd.DataFrame(rows)


def remove_old_files(
    paths: list[Path],
) -> None:
    """删除本实验之前生成的同名文件。"""

    for path in paths:
        if path.exists():
            path.unlink()
            print(f"Removed old file : {path}")


# ============================================================
# 4. 0train 内部 Run 划分
# ============================================================

def create_internal_run_split(
    train_session,
) -> tuple[Subset, Subset, object, pd.DataFrame]:
    """
    将 0train 按 run 划分。

    最后一个 run：
        内部验证集。

    其余 run：
        内部训练集。
    """

    metadata = (
        train_session
        .get_metadata()
        .reset_index(drop=True)
    )

    if "run" not in metadata.columns:
        raise KeyError(
            "The window metadata does not contain a 'run' column. "
            f"Available columns: {metadata.columns.tolist()}"
        )

    run_order = (
        metadata["run"]
        .drop_duplicates()
        .tolist()
    )

    if len(run_order) < 2:
        raise RuntimeError(
            "At least two runs are required for an internal split."
        )

    # 使用按原数据顺序出现的最后一个 run 作为内部验证集
    internal_valid_run = run_order[-1]

    run_values = metadata["run"].to_numpy()

    valid_indices = np.flatnonzero(
        run_values == internal_valid_run
    ).tolist()

    train_indices = np.flatnonzero(
        run_values != internal_valid_run
    ).tolist()

    internal_train_set = Subset(
        train_session,
        train_indices,
    )

    internal_valid_set = Subset(
        train_session,
        valid_indices,
    )

    if len(internal_train_set) == 0:
        raise RuntimeError(
            "Internal training set is empty."
        )

    if len(internal_valid_set) == 0:
        raise RuntimeError(
            "Internal validation set is empty."
        )

    return (
        internal_train_set,
        internal_valid_set,
        internal_valid_run,
        metadata,
    )


# ============================================================
# 5. 模型预测
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
    对数据集执行 Dense Prediction。

    Deep4Net 输出：
        batch × classes × temporal_predictions

    对时间预测维度求平均，获得每个 trial 的类别分数。
    """

    loader = DataLoader(
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

        for batch in loader:

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

            all_scores.append(
                scores.cpu().numpy()
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
# 6. 混淆矩阵绘制
# ============================================================

def save_confusion_matrix(
    confusion: np.ndarray,
    subject_id: int,
    output_path: Path,
) -> None:
    """保存最终测试集混淆矩阵。"""

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
        f"Deep4Net Subject {subject_id}\n"
        "Held-out 1test Confusion Matrix"
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
# 7. 第一阶段：内部验证选择最佳 Epoch
# ============================================================

def run_model_selection(
    internal_train_set,
    internal_valid_set,
    device_name: str,
    checkpoint_path: Path,
) -> tuple[int, pd.DataFrame]:
    """使用 0train 内部验证集选择最佳 Epoch。"""

    print("\n" + "=" * 76)
    print("Stage 1: Internal model selection")
    print("=" * 76)

    set_random_seeds(
        seed=SEED,
        cuda=device_name == "cuda",
    )

    model = create_model()

    checkpoint = Checkpoint(
        monitor="valid_accuracy_best",
        dirname=str(MODELS_DIR),
        f_params=checkpoint_path.name,
        f_optimizer=None,
        f_criterion=None,
        f_history=None,
    )

    early_stopping = EarlyStopping(
        monitor="valid_accuracy",
        patience=EARLY_STOPPING_PATIENCE,
        threshold=1e-4,
        threshold_mode="abs",
        lower_is_better=False,
        load_best=False,
    )

    classifier = EEGClassifier(
        module=model,

        cropped=True,

        criterion=CroppedLoss,

        criterion__loss_function=(
            torch.nn.functional.cross_entropy
        ),

        optimizer=torch.optim.Adam,

        optimizer__lr=LEARNING_RATE,

        optimizer__weight_decay=WEIGHT_DECAY,

        train_split=predefined_split(
            internal_valid_set
        ),

        batch_size=BATCH_SIZE,

        max_epochs=MAX_SELECTION_EPOCHS,

        iterator_train__shuffle=True,

        iterator_train__drop_last=False,

        iterator_valid__drop_last=False,

        callbacks=[
            "accuracy",
            (
                "checkpoint",
                checkpoint,
            ),
            (
                "early_stopping",
                early_stopping,
            ),
        ],

        device=device_name,

        classes=list(
            range(N_CLASSES)
        ),

        verbose=1,
    )

    classifier.fit(
        internal_train_set,
        y=None,
    )

    history_df = history_to_dataframe(
        classifier
    )

    if history_df.empty:
        raise RuntimeError(
            "Model-selection history is empty."
        )

    if "valid_accuracy" not in history_df.columns:
        raise KeyError(
            "valid_accuracy was not found in model-selection history."
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

    best_train_accuracy = float(
        history_df.loc[
            best_index,
            "train_accuracy",
        ]
    )

    best_valid_accuracy = float(
        history_df.loc[
            best_index,
            "valid_accuracy",
        ]
    )

    print("\nInternal model selection completed")
    print(f"Best epoch       : {best_epoch}")
    print(f"Best train acc   : {best_train_accuracy:.4f}")
    print(f"Best internal acc: {best_valid_accuracy:.4f}")

    return best_epoch, history_df


# ============================================================
# 8. 第二阶段：完整 0train 最终训练
# ============================================================

def run_final_training(
    full_train_set,
    best_epoch: int,
    device_name: str,
) -> tuple[EEGClassifier, pd.DataFrame]:
    """
    使用完整 0train 重新训练。

    不使用 1test 作为验证集。
    不执行 Early Stopping。
    训练轮数固定为内部验证选出的最佳 Epoch。
    """

    print("\n" + "=" * 76)
    print("Stage 2: Final training on complete 0train")
    print("=" * 76)

    set_random_seeds(
        seed=SEED,
        cuda=device_name == "cuda",
    )

    final_model = create_model()

    final_classifier = EEGClassifier(
        module=final_model,

        cropped=True,

        criterion=CroppedLoss,

        criterion__loss_function=(
            torch.nn.functional.cross_entropy
        ),

        optimizer=torch.optim.Adam,

        optimizer__lr=LEARNING_RATE,

        optimizer__weight_decay=WEIGHT_DECAY,

        # 最终训练阶段不提供外部验证集
        train_split=None,

        batch_size=BATCH_SIZE,

        max_epochs=best_epoch,

        iterator_train__shuffle=True,

        iterator_train__drop_last=False,

        callbacks=[
            "accuracy",
        ],

        device=device_name,

        classes=list(
            range(N_CLASSES)
        ),

        verbose=1,
    )

    final_classifier.fit(
        full_train_set,
        y=None,
    )

    final_history_df = history_to_dataframe(
        final_classifier
    )

    if final_history_df.empty:
        raise RuntimeError(
            "Final training history is empty."
        )

    return final_classifier, final_history_df


# ============================================================
# 9. 主程序
# ============================================================

def main() -> None:
    """运行完整无泄漏实验。"""

    arguments = parse_arguments()

    subject_id = arguments.subject

    prefix = (
        f"deep4net_subject{subject_id}_runwise_unbiased"
    )

    selection_history_path = (
        METRICS_DIR
        / f"{prefix}_selection_history.csv"
    )

    final_history_path = (
        METRICS_DIR
        / f"{prefix}_final_history.csv"
    )

    selection_checkpoint_path = (
        MODELS_DIR
        / f"{prefix}_selection_best.pth"
    )

    final_model_path = (
        MODELS_DIR
        / f"{prefix}_final.pth"
    )

    metrics_path = (
        METRICS_DIR
        / f"{prefix}_test_metrics.csv"
    )

    report_path = (
        METRICS_DIR
        / f"{prefix}_classification_report.txt"
    )

    predictions_path = (
        METRICS_DIR
        / f"{prefix}_predictions.csv"
    )

    summary_path = (
        METRICS_DIR
        / f"{prefix}_summary.txt"
    )

    confusion_path = (
        FIGURES_DIR
        / f"{prefix}_confusion_matrix.png"
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_paths = [
        selection_history_path,
        final_history_path,
        selection_checkpoint_path,
        final_model_path,
        metrics_path,
        report_path,
        predictions_path,
        summary_path,
        confusion_path,
    ]

    remove_old_files(
        output_paths
    )

    use_cuda = torch.cuda.is_available()

    device_name = (
        "cuda"
        if use_cuda
        else "cpu"
    )

    device = torch.device(
        device_name
    )

    set_random_seeds(
        seed=SEED,
        cuda=use_cuda,
    )

    print("=" * 76)
    print("Deep4Net Run-wise Internal Validation Experiment")
    print("=" * 76)

    print(f"Subject ID       : {subject_id}")
    print(f"Device           : {device_name}")
    print(f"Dropout          : {DROP_PROBABILITY}")
    print(f"Optimizer        : Adam")
    print(f"Learning rate    : {LEARNING_RATE}")
    print(f"Weight decay     : {WEIGHT_DECAY}")
    print(f"Batch size       : {BATCH_SIZE}")

    if subject_id == 1:
        print(
            "\nMethod warning: Subject 1's 1test was previously "
            "used during hyperparameter tuning. Its result is not "
            "a completely untouched test result."
        )

    # ========================================================
    # 10. 加载数据
    # ========================================================

    print("\n[1/6] Loading BNCI2014_001...")

    dataset = MOABBDataset(
        dataset_name="BNCI2014_001",
        subject_ids=[subject_id],
    )

    # ========================================================
    # 11. 数据预处理
    # ========================================================

    print("\n[2/6] Preprocessing EEG...")

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
    # 12. 创建 Cropped Windows
    # ========================================================

    print("\n[3/6] Creating cropped windows...")

    temporary_model = create_model()

    output_shape = (
        temporary_model
        .get_output_shape()
    )

    n_preds_per_input = int(
        output_shape[2]
    )

    trainable_parameters = (
        count_trainable_parameters(
            temporary_model
        )
    )

    del temporary_model

    print(f"Model output     : {output_shape}")
    print(f"Predictions/input: {n_preds_per_input}")
    print(f"Trainable params : {trainable_parameters:,}")

    windows_dataset = create_windows_from_events(
        dataset,

        trial_start_offset_samples=0,

        trial_stop_offset_samples=0,

        window_size_samples=N_TIMES,

        window_stride_samples=n_preds_per_input,

        drop_last_window=False,

        preload=True,
    )

    session_splits = windows_dataset.split(
        "session"
    )

    if "0train" not in session_splits:
        raise KeyError(
            "Session '0train' was not found."
        )

    if "1test" not in session_splits:
        raise KeyError(
            "Session '1test' was not found."
        )

    full_train_set = session_splits[
        "0train"
    ]

    held_out_test_set = session_splits[
        "1test"
    ]

    print(f"Complete 0train  : {len(full_train_set)}")
    print(f"Held-out 1test   : {len(held_out_test_set)}")

    # ========================================================
    # 13. 0train 内部划分
    # ========================================================

    print("\n[4/6] Creating internal run split...")

    (
        internal_train_set,
        internal_valid_set,
        internal_valid_run,
        train_metadata,
    ) = create_internal_run_split(
        full_train_set
    )

    run_order = (
        train_metadata["run"]
        .drop_duplicates()
        .tolist()
    )

    print(f"Available runs   : {run_order}")
    print(f"Internal val run : {internal_valid_run}")
    print(f"Internal train   : {len(internal_train_set)}")
    print(f"Internal valid   : {len(internal_valid_set)}")

    if "target" in train_metadata.columns:
        run_class_table = pd.crosstab(
            train_metadata["run"],
            train_metadata["target"],
        )

        print("\nRun/class distribution:")
        print(run_class_table)

    # ========================================================
    # 14. 第一阶段：选择最佳 Epoch
    # ========================================================

    print("\n[5/6] Selecting best epoch...")

    (
        best_epoch,
        selection_history_df,
    ) = run_model_selection(
        internal_train_set=internal_train_set,
        internal_valid_set=internal_valid_set,
        device_name=device_name,
        checkpoint_path=selection_checkpoint_path,
    )

    selection_history_df.to_csv(
        selection_history_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 15. 第二阶段：完整训练
    # ========================================================

    print("\n[6/6] Training final model...")

    (
        final_classifier,
        final_history_df,
    ) = run_final_training(
        full_train_set=full_train_set,
        best_epoch=best_epoch,
        device_name=device_name,
    )

    final_history_df.to_csv(
        final_history_path,
        index=False,
        encoding="utf-8-sig",
    )

    torch.save(
        final_classifier.module_.state_dict(),
        final_model_path,
    )

    # ========================================================
    # 16. 第三阶段：只在最后评价 1test
    # ========================================================

    print("\n" + "=" * 76)
    print("Stage 3: Final held-out 1test evaluation")
    print("=" * 76)

    y_true, y_pred, scores = collect_predictions(
        model=final_classifier.module_,
        dataset=held_out_test_set,
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

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(N_CLASSES)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    confusion = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(N_CLASSES)),
    )

    true_counts = np.bincount(
        y_true,
        minlength=N_CLASSES,
    )

    predicted_counts = np.bincount(
        y_pred,
        minlength=N_CLASSES,
    )

    print(f"Test samples      : {len(y_true)}")
    print(f"Test accuracy     : {accuracy:.4f}")
    print(
        f"Balanced accuracy : {balanced_accuracy:.4f}"
    )
    print(f"Macro-F1          : {macro_f1:.4f}")
    print(f"Weighted-F1       : {weighted_f1:.4f}")

    print("\nPredicted counts:")

    for class_name, count in zip(
        CLASS_NAMES,
        predicted_counts,
    ):
        print(
            f"  {class_name:<12}: {int(count)}"
        )

    print("\nClassification report:")
    print(report)

    print("Confusion matrix:")
    print(confusion)

    # ========================================================
    # 17. 保存测试结果
    # ========================================================

    metrics_df = pd.DataFrame(
        [
            {
                "subject": subject_id,
                "internal_validation_run": (
                    str(internal_valid_run)
                ),
                "best_epoch": best_epoch,
                "n_test_samples": len(y_true),
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

    metrics_df.to_csv(
        metrics_path,
        index=False,
        encoding="utf-8-sig",
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

    report_content = (
        f"Deep4Net Subject {subject_id} Held-out Test\n"
        f"{'=' * 50}\n"
        f"Internal validation run : {internal_valid_run}\n"
        f"Selected best epoch     : {best_epoch}\n"
        f"Test samples            : {len(y_true)}\n"
        f"Accuracy                : {accuracy:.4f}\n"
        f"Balanced accuracy       : {balanced_accuracy:.4f}\n"
        f"Macro-F1                : {macro_f1:.4f}\n"
        f"Weighted-F1             : {weighted_f1:.4f}\n\n"
        "True counts\n"
        "-----------\n"
        f"{dict(zip(CLASS_NAMES, true_counts.tolist()))}\n\n"
        "Predicted counts\n"
        "----------------\n"
        f"{dict(zip(CLASS_NAMES, predicted_counts.tolist()))}\n\n"
        "Classification report\n"
        "---------------------\n"
        f"{report}\n"
        "Confusion matrix\n"
        "----------------\n"
        f"{confusion}\n"
    )

    report_path.write_text(
        report_content,
        encoding="utf-8",
    )

    save_confusion_matrix(
        confusion=confusion,
        subject_id=subject_id,
        output_path=confusion_path,
    )

    selection_best_index = (
        selection_history_df[
            "valid_accuracy"
        ]
        .astype(float)
        .idxmax()
    )

    internal_best_accuracy = float(
        selection_history_df.loc[
            selection_best_index,
            "valid_accuracy",
        ]
    )

    final_train_accuracy = float(
        final_history_df.iloc[-1].get(
            "train_accuracy",
            np.nan,
        )
    )

    summary = (
        f"Deep4Net Subject {subject_id} Run-wise Protocol\n"
        f"{'=' * 52}\n"
        f"Internal training samples   : {len(internal_train_set)}\n"
        f"Internal validation samples : {len(internal_valid_set)}\n"
        f"Internal validation run     : {internal_valid_run}\n"
        f"Internal best accuracy      : {internal_best_accuracy:.4f}\n"
        f"Selected best epoch         : {best_epoch}\n"
        f"Complete 0train samples     : {len(full_train_set)}\n"
        f"Final train accuracy        : {final_train_accuracy:.4f}\n"
        f"Held-out 1test samples      : {len(held_out_test_set)}\n"
        f"Test accuracy               : {accuracy:.4f}\n"
        f"Test balanced accuracy      : {balanced_accuracy:.4f}\n"
        f"Test Macro-F1               : {macro_f1:.4f}\n"
        f"Test Weighted-F1            : {weighted_f1:.4f}\n"
        f"Final model                 : {final_model_path}\n"
    )

    summary_path.write_text(
        summary,
        encoding="utf-8",
    )

    print("\n" + "=" * 76)
    print("Run-wise experiment completed")
    print("=" * 76)

    print(f"Subject           : {subject_id}")
    print(f"Internal val run  : {internal_valid_run}")
    print(f"Selected epoch    : {best_epoch}")
    print(f"Internal best acc : {internal_best_accuracy:.4f}")
    print(f"Final train acc   : {final_train_accuracy:.4f}")
    print(f"Test accuracy     : {accuracy:.4f}")
    print(f"Test Macro-F1     : {macro_f1:.4f}")
    print(f"Predicted counts  : {predicted_counts.tolist()}")

    print(f"\nSelection history : {selection_history_path}")
    print(f"Final history     : {final_history_path}")
    print(f"Final model       : {final_model_path}")
    print(f"Test metrics      : {metrics_path}")
    print(f"Report            : {report_path}")
    print(f"Predictions       : {predictions_path}")
    print(f"Confusion matrix  : {confusion_path}")
    print(f"Summary           : {summary_path}")

    print("\nResult: PASS")


if __name__ == "__main__":
    main()