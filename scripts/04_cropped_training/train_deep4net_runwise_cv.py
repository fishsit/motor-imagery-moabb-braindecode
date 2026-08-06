"""
Deep4Net 单被试 Run-wise 六折交叉验证与最终测试。

实验流程
--------
阶段 1：0train 内部六折交叉验证

    BNCI2014_001 的 0train 包含 6 个 run。

    Fold 1：run 0 验证，其余 5 个 run 训练
    Fold 2：run 1 验证，其余 5 个 run 训练
    ...
    Fold 6：run 5 验证，其余 5 个 run 训练

    每一折都训练相同的固定轮数。
    然后计算每个 Epoch 在六折上的平均验证准确率，
    选择平均验证准确率最高的 Epoch。

阶段 2：完整 0train 训练

    创建一个全新的 Deep4Net；
    使用完整 0train；
    训练轮数固定为阶段 1 选出的最佳 Epoch。

阶段 3：最终测试

    最终模型训练完成后，才在 1test 上评价一次。
    1test 不参与 Epoch 选择、Early Stopping 或模型训练。

推荐首次运行：
    python scripts/04_cropped_training/train_deep4net_runwise_cv.py --subject 3

说明
----
Subject 1 和 Subject 2 的测试结果已经被查看。
因此本脚本首先在尚未评价的 Subject 3 上运行。

当前固定模型配置：
    Dropout = 0.25
    Optimizer = Adam
    Learning Rate = 0.001
    Weight Decay = 0
    Maximum CV Epochs = 35
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
# 1. 固定实验配置
# ============================================================

N_CHANS = 22
N_CLASSES = 4

# 250 Hz × 4 秒
N_TIMES = 1000

LOW_CUT_HZ = 4.0
HIGH_CUT_HZ = 38.0

FACTOR_NEW = 1e-3
INIT_BLOCK_SIZE = 1000

SEED = 20260806

# 六折都训练相同轮数，才能按 Epoch 求平均
MAX_CV_EPOCHS = 35

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
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


# ============================================================
# 2. 命令行参数
# ============================================================

def parse_arguments() -> argparse.Namespace:
    """读取被试编号。"""

    parser = argparse.ArgumentParser(
        description=(
            "Deep4Net run-wise cross-validation "
            "and held-out session evaluation."
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
    """创建当前固定配置的 Deep4Net。"""

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
    """将 skorch History 转换为普通 DataFrame。"""

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
    """删除本次实验的旧输出文件。"""

    for path in paths:
        if path.exists():
            path.unlink()
            print(f"Removed old file : {path}")


# ============================================================
# 4. 获取 Run 信息
# ============================================================

def get_run_information(
    train_session,
) -> tuple[pd.DataFrame, list[object]]:
    """读取 0train 的窗口元数据和 run 顺序。"""

    metadata = (
        train_session
        .get_metadata()
        .reset_index(drop=True)
    )

    if "run" not in metadata.columns:
        raise KeyError(
            "Window metadata does not contain a 'run' column. "
            f"Available columns: {metadata.columns.tolist()}"
        )

    run_order = (
        metadata["run"]
        .drop_duplicates()
        .tolist()
    )

    if len(run_order) < 2:
        raise RuntimeError(
            "At least two runs are required."
        )

    return metadata, run_order


def create_fold_datasets(
    full_train_set,
    metadata: pd.DataFrame,
    validation_run,
) -> tuple[Subset, Subset]:
    """
    创建一折的训练集与验证集。

    validation_run 对应的样本作为验证集；
    其余所有 run 作为训练集。
    """

    run_values = metadata["run"].to_numpy()

    valid_indices = np.flatnonzero(
        run_values == validation_run
    ).tolist()

    train_indices = np.flatnonzero(
        run_values != validation_run
    ).tolist()

    train_subset = Subset(
        full_train_set,
        train_indices,
    )

    valid_subset = Subset(
        full_train_set,
        valid_indices,
    )

    if len(train_subset) == 0:
        raise RuntimeError(
            f"Training subset is empty for run {validation_run}."
        )

    if len(valid_subset) == 0:
        raise RuntimeError(
            f"Validation subset is empty for run {validation_run}."
        )

    return train_subset, valid_subset


# ============================================================
# 5. 单折交叉验证
# ============================================================

def train_one_fold(
    fold_number: int,
    validation_run,
    train_subset,
    valid_subset,
    device_name: str,
) -> pd.DataFrame:
    """训练一折 Deep4Net，并返回每个 Epoch 的历史。"""

    fold_seed = SEED + fold_number

    set_random_seeds(
        seed=fold_seed,
        cuda=device_name == "cuda",
    )

    model = create_model()

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
            valid_subset
        ),

        batch_size=BATCH_SIZE,

        max_epochs=MAX_CV_EPOCHS,

        iterator_train__shuffle=True,

        iterator_train__drop_last=False,

        iterator_valid__drop_last=False,

        callbacks=[
            "accuracy",
        ],

        device=device_name,

        classes=list(range(N_CLASSES)),

        # 六折训练时减少大量逐 Epoch 输出
        verbose=0,
    )

    classifier.fit(
        train_subset,
        y=None,
    )

    history_df = history_to_dataframe(
        classifier
    )

    required_columns = {
        "epoch",
        "train_accuracy",
        "valid_accuracy",
        "train_loss",
        "valid_loss",
    }

    missing_columns = (
        required_columns
        - set(history_df.columns)
    )

    if missing_columns:
        raise KeyError(
            f"Fold {fold_number} history is missing: "
            f"{sorted(missing_columns)}"
        )

    history_df["fold"] = fold_number
    history_df["validation_run"] = str(
        validation_run
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

    best_train_accuracy = float(
        history_df.loc[
            best_index,
            "train_accuracy",
        ]
    )

    print(
        f"Fold {fold_number:>2} | "
        f"valid run={validation_run} | "
        f"train={len(train_subset):>3} | "
        f"valid={len(valid_subset):>3} | "
        f"best epoch={best_epoch:>2} | "
        f"train acc={best_train_accuracy:.4f} | "
        f"valid acc={best_valid_accuracy:.4f}"
    )

    return history_df


# ============================================================
# 6. 六折交叉验证
# ============================================================

def run_cross_validation(
    full_train_set,
    metadata: pd.DataFrame,
    run_order: list[object],
    device_name: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    int,
]:
    """执行全部 run-wise 交叉验证。"""

    fold_histories: list[pd.DataFrame] = []
    fold_best_rows: list[dict[str, object]] = []

    print("\n" + "=" * 80)
    print("Stage 1: Run-wise cross-validation inside 0train")
    print("=" * 80)

    for fold_number, validation_run in enumerate(
        run_order,
        start=1,
    ):
        train_subset, valid_subset = (
            create_fold_datasets(
                full_train_set=full_train_set,
                metadata=metadata,
                validation_run=validation_run,
            )
        )

        fold_history = train_one_fold(
            fold_number=fold_number,
            validation_run=validation_run,
            train_subset=train_subset,
            valid_subset=valid_subset,
            device_name=device_name,
        )

        fold_histories.append(
            fold_history
        )

        best_index = (
            fold_history["valid_accuracy"]
            .astype(float)
            .idxmax()
        )

        best_row = fold_history.loc[
            best_index
        ]

        fold_best_rows.append(
            {
                "fold": fold_number,
                "validation_run": str(validation_run),
                "best_epoch": int(
                    best_row["epoch"]
                ),
                "best_train_accuracy": float(
                    best_row["train_accuracy"]
                ),
                "best_valid_accuracy": float(
                    best_row["valid_accuracy"]
                ),
                "best_train_loss": float(
                    best_row["train_loss"]
                ),
                "best_valid_loss": float(
                    best_row["valid_loss"]
                ),
            }
        )

    all_history_df = pd.concat(
        fold_histories,
        ignore_index=True,
    )

    fold_best_df = pd.DataFrame(
        fold_best_rows
    )

    epoch_summary_df = (
        all_history_df
        .groupby("epoch", as_index=False)
        .agg(
            mean_train_accuracy=(
                "train_accuracy",
                "mean",
            ),
            std_train_accuracy=(
                "train_accuracy",
                "std",
            ),
            mean_valid_accuracy=(
                "valid_accuracy",
                "mean",
            ),
            std_valid_accuracy=(
                "valid_accuracy",
                "std",
            ),
            min_valid_accuracy=(
                "valid_accuracy",
                "min",
            ),
            max_valid_accuracy=(
                "valid_accuracy",
                "max",
            ),
            mean_train_loss=(
                "train_loss",
                "mean",
            ),
            mean_valid_loss=(
                "valid_loss",
                "mean",
            ),
        )
    )

    # idxmax 在并列时选择第一个，因此会优先选择较早 Epoch
    best_summary_index = (
        epoch_summary_df["mean_valid_accuracy"]
        .astype(float)
        .idxmax()
    )

    selected_epoch = int(
        epoch_summary_df.loc[
            best_summary_index,
            "epoch",
        ]
    )

    selected_mean_accuracy = float(
        epoch_summary_df.loc[
            best_summary_index,
            "mean_valid_accuracy",
        ]
    )

    selected_std_accuracy = float(
        epoch_summary_df.loc[
            best_summary_index,
            "std_valid_accuracy",
        ]
    )

    print("\nCross-validation selection completed")
    print(f"Selected epoch    : {selected_epoch}")
    print(
        "Mean valid acc    : "
        f"{selected_mean_accuracy:.4f}"
    )
    print(
        "Valid acc std     : "
        f"{selected_std_accuracy:.4f}"
    )

    return (
        all_history_df,
        fold_best_df,
        epoch_summary_df,
        selected_epoch,
    )


# ============================================================
# 7. 最终模型训练
# ============================================================

def train_final_model(
    full_train_set,
    selected_epoch: int,
    device_name: str,
) -> tuple[EEGClassifier, pd.DataFrame]:
    """
    使用完整 0train 训练最终模型。

    不使用外部验证集；
    不使用 Early Stopping；
    训练轮数固定为六折选择出的 Epoch。
    """

    print("\n" + "=" * 80)
    print("Stage 2: Final training on complete 0train")
    print("=" * 80)

    set_random_seeds(
        seed=SEED,
        cuda=device_name == "cuda",
    )

    model = create_model()

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

        train_split=None,

        batch_size=BATCH_SIZE,

        max_epochs=selected_epoch,

        iterator_train__shuffle=True,

        iterator_train__drop_last=False,

        callbacks=[
            "accuracy",
        ],

        device=device_name,

        classes=list(range(N_CLASSES)),

        verbose=1,
    )

    classifier.fit(
        full_train_set,
        y=None,
    )

    history_df = history_to_dataframe(
        classifier
    )

    if history_df.empty:
        raise RuntimeError(
            "Final training history is empty."
        )

    return classifier, history_df


# ============================================================
# 8. 最终模型预测
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
    """在最终测试集上生成预测。"""

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
            "No test predictions were generated."
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
# 9. 绘制交叉验证曲线
# ============================================================

def plot_cv_accuracy(
    epoch_summary_df: pd.DataFrame,
    selected_epoch: int,
    output_path: Path,
) -> None:
    """绘制六折平均训练与验证准确率。"""

    epochs = (
        epoch_summary_df["epoch"]
        .to_numpy(dtype=float)
    )

    mean_train = (
        epoch_summary_df[
            "mean_train_accuracy"
        ]
        .to_numpy(dtype=float)
    )

    mean_valid = (
        epoch_summary_df[
            "mean_valid_accuracy"
        ]
        .to_numpy(dtype=float)
    )

    std_valid = (
        epoch_summary_df[
            "std_valid_accuracy"
        ]
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    axis.plot(
        epochs,
        mean_train,
        marker="o",
        label="Mean train accuracy",
    )

    axis.plot(
        epochs,
        mean_valid,
        marker="o",
        label="Mean validation accuracy",
    )

    axis.fill_between(
        epochs,
        mean_valid - std_valid,
        mean_valid + std_valid,
        alpha=0.2,
        label="Validation ±1 std",
    )

    axis.axvline(
        selected_epoch,
        linestyle="--",
        label=f"Selected epoch = {selected_epoch}",
    )

    axis.axhline(
        0.25,
        linestyle=":",
        label="Chance level = 0.25",
    )

    axis.set_title(
        "Deep4Net Run-wise Cross-validation Accuracy"
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
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_confusion_matrix(
    confusion: np.ndarray,
    subject_id: int,
    output_path: Path,
) -> None:
    """保存最终 1test 混淆矩阵。"""

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
# 10. 主程序
# ============================================================

def main() -> None:
    """运行六折交叉验证和最终测试。"""

    arguments = parse_arguments()

    subject_id = arguments.subject

    prefix = (
        f"deep4net_subject{subject_id}_runwise_cv"
    )

    all_fold_history_path = (
        METRICS_DIR
        / f"{prefix}_all_fold_history.csv"
    )

    fold_best_path = (
        METRICS_DIR
        / f"{prefix}_fold_best.csv"
    )

    epoch_summary_path = (
        METRICS_DIR
        / f"{prefix}_epoch_summary.csv"
    )

    final_history_path = (
        METRICS_DIR
        / f"{prefix}_final_history.csv"
    )

    test_metrics_path = (
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

    final_model_path = (
        MODELS_DIR
        / f"{prefix}_final.pth"
    )

    cv_curve_path = (
        FIGURES_DIR
        / f"{prefix}_cv_accuracy.png"
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
        all_fold_history_path,
        fold_best_path,
        epoch_summary_path,
        final_history_path,
        test_metrics_path,
        report_path,
        predictions_path,
        summary_path,
        final_model_path,
        cv_curve_path,
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

    print("=" * 80)
    print("Deep4Net Run-wise Cross-validation Experiment")
    print("=" * 80)

    print(f"Subject ID       : {subject_id}")
    print(f"Device           : {device_name}")
    print(f"CV epochs        : {MAX_CV_EPOCHS}")
    print(f"Dropout          : {DROP_PROBABILITY}")
    print(f"Optimizer        : Adam")
    print(f"Learning rate    : {LEARNING_RATE}")
    print(f"Weight decay     : {WEIGHT_DECAY}")
    print(f"Batch size       : {BATCH_SIZE}")
    print(f"Random seed      : {SEED}")

    if subject_id in {1, 2}:
        print(
            "\nWarning: Subject 1 or Subject 2 test results "
            "have already been inspected during development."
        )

    # ========================================================
    # 11. 加载并预处理数据
    # ========================================================

    print("\n[1/7] Loading BNCI2014_001...")

    dataset = MOABBDataset(
        dataset_name="BNCI2014_001",
        subject_ids=[subject_id],
    )

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
    # 12. 创建窗口
    # ========================================================

    print("\n[3/7] Creating windows...")

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
    # 13. 读取 Run 元数据
    # ========================================================

    print("\n[4/7] Reading run metadata...")

    metadata, run_order = get_run_information(
        full_train_set
    )

    print(f"Run order        : {run_order}")
    print(f"Number of runs   : {len(run_order)}")

    if len(run_order) != 6:
        print(
            "Warning: BNCI2014_001 normally contains "
            f"6 runs, but {len(run_order)} were found."
        )

    if "target" in metadata.columns:
        print("\nRun/class distribution:")

        run_class_table = pd.crosstab(
            metadata["run"],
            metadata["target"],
        )

        print(run_class_table)

    # ========================================================
    # 14. 六折交叉验证
    # ========================================================

    print("\n[5/7] Running run-wise cross-validation...")

    (
        all_fold_history_df,
        fold_best_df,
        epoch_summary_df,
        selected_epoch,
    ) = run_cross_validation(
        full_train_set=full_train_set,
        metadata=metadata,
        run_order=run_order,
        device_name=device_name,
    )

    all_fold_history_df.to_csv(
        all_fold_history_path,
        index=False,
        encoding="utf-8-sig",
    )

    fold_best_df.to_csv(
        fold_best_path,
        index=False,
        encoding="utf-8-sig",
    )

    epoch_summary_df.to_csv(
        epoch_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    plot_cv_accuracy(
        epoch_summary_df=epoch_summary_df,
        selected_epoch=selected_epoch,
        output_path=cv_curve_path,
    )

    # ========================================================
    # 15. 使用完整 0train 训练最终模型
    # ========================================================

    print("\n[6/7] Training final model...")

    final_classifier, final_history_df = (
        train_final_model(
            full_train_set=full_train_set,
            selected_epoch=selected_epoch,
            device_name=device_name,
        )
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

    final_train_accuracy = float(
        final_history_df.iloc[-1].get(
            "train_accuracy",
            np.nan,
        )
    )

    # ========================================================
    # 16. 最终 1test 评价
    # ========================================================

    print("\n[7/7] Evaluating held-out 1test...")

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

    selected_row = (
        epoch_summary_df.loc[
            epoch_summary_df["epoch"]
            == selected_epoch
        ]
        .iloc[0]
    )

    cv_mean_accuracy = float(
        selected_row["mean_valid_accuracy"]
    )

    cv_std_accuracy = float(
        selected_row["std_valid_accuracy"]
    )

    print("\n" + "=" * 80)
    print("Final held-out 1test evaluation")
    print("=" * 80)

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
    # 17. 保存最终结果
    # ========================================================

    test_metrics_df = pd.DataFrame(
        [
            {
                "subject": subject_id,
                "n_runs": len(run_order),
                "max_cv_epochs": MAX_CV_EPOCHS,
                "selected_epoch": selected_epoch,
                "cv_mean_valid_accuracy": (
                    cv_mean_accuracy
                ),
                "cv_std_valid_accuracy": (
                    cv_std_accuracy
                ),
                "final_train_accuracy": (
                    final_train_accuracy
                ),
                "n_test_samples": len(y_true),
                "test_accuracy": accuracy,
                "test_balanced_accuracy": (
                    balanced_accuracy
                ),
                "test_macro_f1": macro_f1,
                "test_weighted_f1": weighted_f1,
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

    test_metrics_df.to_csv(
        test_metrics_path,
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
        f"Deep4Net Subject {subject_id} Run-wise CV Test\n"
        f"{'=' * 55}\n"
        f"Number of runs          : {len(run_order)}\n"
        f"Maximum CV epochs       : {MAX_CV_EPOCHS}\n"
        f"Selected epoch          : {selected_epoch}\n"
        f"CV mean valid accuracy  : {cv_mean_accuracy:.4f}\n"
        f"CV valid accuracy std   : {cv_std_accuracy:.4f}\n"
        f"Final train accuracy    : {final_train_accuracy:.4f}\n"
        f"Test samples            : {len(y_true)}\n"
        f"Test accuracy           : {accuracy:.4f}\n"
        f"Balanced accuracy       : {balanced_accuracy:.4f}\n"
        f"Macro-F1                : {macro_f1:.4f}\n"
        f"Weighted-F1             : {weighted_f1:.4f}\n\n"
        "True class counts\n"
        "-----------------\n"
        f"{dict(zip(CLASS_NAMES, true_counts.tolist()))}\n\n"
        "Predicted class counts\n"
        "----------------------\n"
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

    summary_content = (
        f"Deep4Net Subject {subject_id} Run-wise CV Summary\n"
        f"{'=' * 58}\n"
        f"Run order               : {run_order}\n"
        f"Number of folds         : {len(run_order)}\n"
        f"Maximum CV epochs       : {MAX_CV_EPOCHS}\n"
        f"Selected epoch          : {selected_epoch}\n"
        f"CV mean valid accuracy  : {cv_mean_accuracy:.4f}\n"
        f"CV valid accuracy std   : {cv_std_accuracy:.4f}\n"
        f"Final training samples  : {len(full_train_set)}\n"
        f"Final train accuracy    : {final_train_accuracy:.4f}\n"
        f"Held-out test samples   : {len(held_out_test_set)}\n"
        f"Test accuracy           : {accuracy:.4f}\n"
        f"Test balanced accuracy  : {balanced_accuracy:.4f}\n"
        f"Test Macro-F1           : {macro_f1:.4f}\n"
        f"Test Weighted-F1        : {weighted_f1:.4f}\n"
        f"Final model             : {final_model_path}\n"
    )

    summary_path.write_text(
        summary_content,
        encoding="utf-8",
    )

    save_confusion_matrix(
        confusion=confusion,
        subject_id=subject_id,
        output_path=confusion_path,
    )

    # ========================================================
    # 18. 最终终端摘要
    # ========================================================

    print("\n" + "=" * 80)
    print("Run-wise cross-validation experiment completed")
    print("=" * 80)

    print(f"Subject           : {subject_id}")
    print(f"Number of folds   : {len(run_order)}")
    print(f"Selected epoch    : {selected_epoch}")
    print(f"CV mean valid acc : {cv_mean_accuracy:.4f}")
    print(f"CV valid acc std  : {cv_std_accuracy:.4f}")
    print(f"Final train acc   : {final_train_accuracy:.4f}")
    print(f"Test accuracy     : {accuracy:.4f}")
    print(f"Test Macro-F1     : {macro_f1:.4f}")
    print(
        f"Predicted counts  : "
        f"{predicted_counts.tolist()}"
    )

    print(
        f"\nAll-fold history  : {all_fold_history_path}"
    )
    print(f"Fold best results : {fold_best_path}")
    print(f"Epoch summary     : {epoch_summary_path}")
    print(f"Final history     : {final_history_path}")
    print(f"Final model       : {final_model_path}")
    print(f"Test metrics      : {test_metrics_path}")
    print(f"Report            : {report_path}")
    print(f"Predictions       : {predictions_path}")
    print(f"CV curve          : {cv_curve_path}")
    print(f"Confusion matrix  : {confusion_path}")
    print(f"Summary           : {summary_path}")

    print("\nResult: PASS")


if __name__ == "__main__":
    main()