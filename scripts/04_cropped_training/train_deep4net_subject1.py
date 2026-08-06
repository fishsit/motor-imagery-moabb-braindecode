"""
Deep4Net 单被试跨会话 Cropped Training 正式实验。

实验协议
--------
数据集：
    BNCI2014_001

被试：
    Subject 1

训练集：
    0train 会话

验证集：
    1test 会话

预处理：
    1. 仅保留 22 个 EEG 通道
    2. EEG 从伏特转换为微伏
    3. 4～38 Hz 带通滤波
    4. Exponential Moving Standardization

模型：
    Deep4Net
    Dense Prediction
    CroppedLoss

训练策略：
    最大训练 40 个 epoch
    验证准确率连续 6 个 epoch 没有提升时提前停止
    自动保存验证准确率最高的模型
    同时保存训练结束时的最后一轮模型

输出文件：
    results/metrics/deep4net_subject1_history.csv
    results/metrics/deep4net_subject1_summary.txt
    models/deep4net_subject1_best.pth
    models/deep4net_subject1_final.pth
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from skorch.callbacks import Checkpoint, EarlyStopping
from skorch.helper import predefined_split

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
# 1. 实验参数
# ============================================================

SUBJECT_ID = 1

N_CHANS = 22
N_CLASSES = 4

# BNCI2014_001 的采样率为 250 Hz。
# 当前模型输入长度为 1000 点，即 4 秒。
N_TIMES = 1000

LOW_CUT_HZ = 4.0
HIGH_CUT_HZ = 38.0

# Exponential Moving Standardization 参数
FACTOR_NEW = 1e-3
INIT_BLOCK_SIZE = 1000

SEED = 20260806

# 最大训练轮数。
# EarlyStopping 可能会在达到 40 轮之前停止。
MAX_EPOCHS = 40

BATCH_SIZE = 32

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4

# EarlyStopping 参数
EARLY_STOPPING_PATIENCE = 6
EARLY_STOPPING_THRESHOLD = 1e-4

RESULTS_DIR = Path("results/metrics")
MODELS_DIR = Path("models")

HISTORY_PATH = (
    RESULTS_DIR
    / "deep4net_subject1_history.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "deep4net_subject1_summary.txt"
)

BEST_MODEL_PATH = (
    MODELS_DIR
    / "deep4net_subject1_best.pth"
)

FINAL_MODEL_PATH = (
    MODELS_DIR
    / "deep4net_subject1_final.pth"
)


# ============================================================
# 2. 辅助函数
# ============================================================

def scale_to_microvolts(
    data: np.ndarray,
) -> np.ndarray:
    """
    将 EEG 信号从伏特转换为微伏。

    MNE 默认以伏特保存 EEG 数据，乘以 1e6 后转换为微伏。
    """
    return data * 1e6


def count_trainable_parameters(
    model: torch.nn.Module,
) -> int:
    """统计模型中的可训练参数数量。"""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def history_to_dataframe(
    classifier: EEGClassifier,
) -> pd.DataFrame:
    """
    将 skorch 训练历史转换为 DataFrame。

    classifier.history 中包含 batch 级嵌套数据。
    保存 CSV 时仅保留 epoch 级指标。
    """
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


def get_float_value(
    row: pd.Series,
    key: str,
) -> float:
    """
    安全地从训练历史中读取浮点指标。

    指标不存在时返回 NaN。
    """
    value = row.get(key, np.nan)

    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


# ============================================================
# 3. 主程序
# ============================================================

def main() -> None:
    """运行 Deep4Net Subject 1 跨会话正式实验。"""

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    set_random_seeds(
        seed=SEED,
        cuda=use_cuda,
    )

    print("=" * 72)
    print("Deep4Net Subject 1 Cross-Session Cropped Training")
    print("=" * 72)

    print(f"Subject ID       : {SUBJECT_ID}")
    print(f"Device           : {device}")
    print(f"Maximum epochs   : {MAX_EPOCHS}")
    print(f"Early patience   : {EARLY_STOPPING_PATIENCE}")
    print(f"Input samples    : {N_TIMES}")
    print(f"Batch size       : {BATCH_SIZE}")
    print(f"Learning rate    : {LEARNING_RATE}")
    print(f"Weight decay     : {WEIGHT_DECAY}")
    print(f"Random seed      : {SEED}")

    # ========================================================
    # 4. 加载数据
    # ========================================================

    print("\n[1/8] Loading BNCI2014_001...")

    dataset = MOABBDataset(
        dataset_name="BNCI2014_001",
        subject_ids=[SUBJECT_ID],
    )

    print(dataset)

    # ========================================================
    # 5. 数据预处理
    # ========================================================

    print("\n[2/8] Preprocessing EEG...")

    preprocessors = [
        # 只保留 EEG 通道。
        # 相比 pick_types，这种写法不会产生 legacy 警告。
        Preprocessor(
            "pick",
            picks="eeg",
        ),

        # EEG 从伏特转换为微伏。
        Preprocessor(
            scale_to_microvolts,
        ),

        # 4～38 Hz 带通滤波。
        Preprocessor(
            "filter",
            l_freq=LOW_CUT_HZ,
            h_freq=HIGH_CUT_HZ,
        ),

        # 指数移动标准化。
        Preprocessor(
            exponential_moving_standardize,
            factor_new=FACTOR_NEW,
            init_block_size=INIT_BLOCK_SIZE,
        ),
    ]

    # Windows 环境下使用 n_jobs=1 较为稳定。
    preprocess(
        dataset,
        preprocessors,
        n_jobs=1,
    )

    sfreq = float(
        dataset.datasets[0].raw.info["sfreq"]
    )

    n_chans = int(
        dataset.datasets[0].raw.info["nchan"]
    )

    print(f"Sampling rate    : {sfreq} Hz")
    print(f"EEG channels     : {n_chans}")

    if n_chans != N_CHANS:
        raise RuntimeError(
            f"Expected {N_CHANS} EEG channels, "
            f"but received {n_chans}."
        )

    # ========================================================
    # 6. 创建 Deep4Net
    # ========================================================

    print("\n[3/8] Creating Deep4Net...")

    model = Deep4Net(
        n_chans=N_CHANS,
        n_outputs=N_CLASSES,
        n_times=N_TIMES,

        # 设为较小值，使模型可以输出多个时间位置的预测。
        final_conv_length=2,

        drop_prob=0.5,
    )

    # 将普通分类模型转换为密集预测模型。
    model.to_dense_prediction_model()

    output_shape = model.get_output_shape()

    n_preds_per_input = int(
        output_shape[2]
    )

    trainable_params = count_trainable_parameters(
        model
    )

    print(f"Model output     : {output_shape}")
    print(f"Predictions/input: {n_preds_per_input}")
    print(f"Trainable params : {trainable_params:,}")

    if n_preds_per_input <= 1:
        raise RuntimeError(
            "Deep4Net did not produce multiple dense predictions."
        )

    # ========================================================
    # 7. 创建 Cropped Windows
    # ========================================================

    print("\n[4/8] Creating cropped windows...")

    windows_dataset = create_windows_from_events(
        dataset,

        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,

        window_size_samples=N_TIMES,

        # 每个输入的密集预测数量作为窗口步长。
        window_stride_samples=n_preds_per_input,

        drop_last_window=False,
        preload=True,
    )

    print(windows_dataset)

    # ========================================================
    # 8. 按会话划分训练集和验证集
    # ========================================================

    print("\n[5/8] Splitting sessions...")

    split_datasets = windows_dataset.split(
        "session"
    )

    available_sessions = list(
        split_datasets.keys()
    )

    print(
        "Available sessions:",
        available_sessions,
    )

    if "0train" not in split_datasets:
        raise KeyError(
            "Session '0train' was not found."
        )

    if "1test" not in split_datasets:
        raise KeyError(
            "Session '1test' was not found."
        )

    train_set = split_datasets["0train"]
    valid_set = split_datasets["1test"]

    print(f"Train windows    : {len(train_set)}")
    print(f"Valid windows    : {len(valid_set)}")

    sample_x, sample_y, sample_ind = train_set[0]

    print(
        f"First input shape: {tuple(sample_x.shape)}"
    )

    print(
        f"First target     : {int(sample_y)}"
    )

    print(
        f"First index      : {sample_ind}"
    )

    expected_shape = (
        N_CHANS,
        N_TIMES,
    )

    if tuple(sample_x.shape) != expected_shape:
        raise RuntimeError(
            f"Expected input shape {expected_shape}, "
            f"but received {tuple(sample_x.shape)}."
        )

    # ========================================================
    # 9. 配置最佳模型保存
    # ========================================================

    print("\n[6/8] Configuring callbacks...")

    checkpoint = Checkpoint(
        # valid_accuracy_best 由 accuracy callback 生成。
        # 只有验证准确率达到当前最佳值时才保存模型。
        monitor="valid_accuracy_best",

        dirname=str(MODELS_DIR),

        f_params=BEST_MODEL_PATH.name,

        # 当前只保存模型参数，不额外保存优化器、损失函数和历史。
        f_optimizer=None,
        f_criterion=None,
        f_history=None,
    )

    early_stopping = EarlyStopping(
        monitor="valid_accuracy",

        patience=EARLY_STOPPING_PATIENCE,

        threshold=EARLY_STOPPING_THRESHOLD,

        threshold_mode="abs",

        # 验证准确率越高越好。
        lower_is_better=False,

        # 不自动把最佳参数加载回当前模型。
        # 这样可以分别保留 best 模型和 final 模型。
        load_best=False,
    )

    # ========================================================
    # 10. 创建 EEGClassifier
    # ========================================================

    print("\n[7/8] Creating EEGClassifier...")

    classifier = EEGClassifier(
        module=model,

        # 开启 Cropped Training。
        cropped=True,

        criterion=CroppedLoss,

        criterion__loss_function=(
            torch.nn.functional.cross_entropy
        ),

        optimizer=torch.optim.AdamW,

        optimizer__lr=LEARNING_RATE,

        optimizer__weight_decay=WEIGHT_DECAY,

        # 使用 Subject 1 的 1test 会话作为验证集。
        train_split=predefined_split(
            valid_set
        ),

        batch_size=BATCH_SIZE,

        max_epochs=MAX_EPOCHS,

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

        device=device,

        # 显式指定四个分类标签，避免 skorch 推断失败。
        classes=list(
            range(N_CLASSES)
        ),

        verbose=1,
    )

    # ========================================================
    # 11. 正式训练
    # ========================================================

    print("\n[8/8] Starting formal training...")

    classifier.fit(
        train_set,
        y=None,
    )

    # ========================================================
    # 12. 保存训练历史
    # ========================================================

    history_df = history_to_dataframe(
        classifier
    )

    if history_df.empty:
        raise RuntimeError(
            "Training history is empty."
        )

    history_df.to_csv(
        HISTORY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # 保存训练停止时的最后一轮模型。
    torch.save(
        classifier.module_.state_dict(),
        FINAL_MODEL_PATH,
    )

    if "valid_accuracy" not in history_df.columns:
        raise RuntimeError(
            "Training history does not contain valid_accuracy."
        )

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            "The best model checkpoint was not created: "
            f"{BEST_MODEL_PATH}"
        )

    # ========================================================
    # 13. 查找训练历史中的最佳结果
    # ========================================================

    best_row_index = (
        history_df["valid_accuracy"]
        .astype(float)
        .idxmax()
    )

    best_row = history_df.loc[
        best_row_index
    ]

    final_row = history_df.iloc[-1]

    best_epoch = int(
        best_row.get(
            "epoch",
            best_row_index + 1,
        )
    )

    completed_epochs = int(
        final_row.get(
            "epoch",
            len(history_df),
        )
    )

    best_train_accuracy = get_float_value(
        best_row,
        "train_accuracy",
    )

    best_valid_accuracy = get_float_value(
        best_row,
        "valid_accuracy",
    )

    best_train_loss = get_float_value(
        best_row,
        "train_loss",
    )

    best_valid_loss = get_float_value(
        best_row,
        "valid_loss",
    )

    final_train_accuracy = get_float_value(
        final_row,
        "train_accuracy",
    )

    final_valid_accuracy = get_float_value(
        final_row,
        "valid_accuracy",
    )

    final_train_loss = get_float_value(
        final_row,
        "train_loss",
    )

    final_valid_loss = get_float_value(
        final_row,
        "valid_loss",
    )

    stopped_early = (
        completed_epochs < MAX_EPOCHS
    )

    # ========================================================
    # 14. 保存实验摘要
    # ========================================================

    summary = (
        "Deep4Net Subject 1 Cross-Session Experiment\n"
        "===========================================\n"
        f"Subject ID            : {SUBJECT_ID}\n"
        "Train session         : 0train\n"
        "Validation session    : 1test\n"
        f"Maximum epochs        : {MAX_EPOCHS}\n"
        f"Completed epochs      : {completed_epochs}\n"
        f"Stopped early         : {stopped_early}\n"
        f"Early stop patience   : {EARLY_STOPPING_PATIENCE}\n"
        f"Learning rate         : {LEARNING_RATE}\n"
        f"Weight decay          : {WEIGHT_DECAY}\n"
        f"Batch size            : {BATCH_SIZE}\n"
        f"Trainable parameters  : {trainable_params}\n"
        f"Dense predictions     : {n_preds_per_input}\n"
        f"Best epoch            : {best_epoch}\n"
        f"Best train accuracy   : {best_train_accuracy:.4f}\n"
        f"Best valid accuracy   : {best_valid_accuracy:.4f}\n"
        f"Best train loss       : {best_train_loss:.4f}\n"
        f"Best valid loss       : {best_valid_loss:.4f}\n"
        f"Final train accuracy  : {final_train_accuracy:.4f}\n"
        f"Final valid accuracy  : {final_valid_accuracy:.4f}\n"
        f"Final train loss      : {final_train_loss:.4f}\n"
        f"Final valid loss      : {final_valid_loss:.4f}\n"
        f"Best model path       : {BEST_MODEL_PATH}\n"
        f"Final model path      : {FINAL_MODEL_PATH}\n"
    )

    SUMMARY_PATH.write_text(
        summary,
        encoding="utf-8",
    )

    # ========================================================
    # 15. 输出最终结果
    # ========================================================

    print("\n" + "=" * 72)
    print("Training completed")
    print("=" * 72)

    print(
        f"Completed epochs : {completed_epochs}"
    )

    print(
        f"Stopped early    : {stopped_early}"
    )

    print(
        f"History saved to : {HISTORY_PATH}"
    )

    print(
        f"Best model saved : {BEST_MODEL_PATH}"
    )

    print(
        f"Final model saved: {FINAL_MODEL_PATH}"
    )

    print(
        f"Summary saved to : {SUMMARY_PATH}"
    )

    print(
        f"Best epoch       : {best_epoch}"
    )

    print(
        f"Best train acc   : {best_train_accuracy:.4f}"
    )

    print(
        f"Best valid acc   : {best_valid_accuracy:.4f}"
    )

    print(
        f"Final train acc  : {final_train_accuracy:.4f}"
    )

    print(
        f"Final valid acc  : {final_valid_accuracy:.4f}"
    )

    print("\nResult: PASS")


if __name__ == "__main__":
    main()