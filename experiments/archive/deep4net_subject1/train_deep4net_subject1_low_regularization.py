"""
Deep4Net 单被试跨会话低正则化对比实验。

实验目的
--------
前一版 Deep4Net 使用：

    Dropout = 0.5
    AdamW
    Weight Decay = 5e-4

训练集准确率只有约 44%，并出现明显的类别预测偏置。

小样本过拟合测试证明：
    Deep4Net、Dense Prediction、CroppedLoss 和标签映射均正常。

因此，本实验降低正则化强度，观察：

1. 训练准确率能否明显提升；
2. 验证准确率能否超过 45.49%；
3. tongue 类别是否能够被正常识别；
4. 模型是否仍然偏向 right_hand。

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

低正则化配置：
    Dropout = 0.25
    Optimizer = Adam
    Weight Decay = 0
    Learning Rate = 0.001
    Maximum Epochs = 60
    Early Stopping Patience = 10

输出文件：
    results/metrics/
        deep4net_subject1_low_regularization_history.csv
        deep4net_subject1_low_regularization_summary.txt

    models/
        deep4net_subject1_low_regularization_best.pth
        deep4net_subject1_low_regularization_final.pth
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

# 250 Hz × 4 秒
N_TIMES = 1000

LOW_CUT_HZ = 4.0
HIGH_CUT_HZ = 38.0

# Exponential Moving Standardization 参数
FACTOR_NEW = 1e-3
INIT_BLOCK_SIZE = 1000

# 与原实验保持相同随机种子，便于公平比较
SEED = 20260806

# 低正则化实验允许训练更长时间
MAX_EPOCHS = 60

BATCH_SIZE = 32

LEARNING_RATE = 1e-3

# 本次实验关闭权重衰减
WEIGHT_DECAY = 0.0

# 本次实验降低 Dropout
DROP_PROBABILITY = 0.25

# Early Stopping 参数
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_THRESHOLD = 1e-4

RESULTS_DIR = Path("results/metrics")
MODELS_DIR = Path("models")

HISTORY_PATH = (
    RESULTS_DIR
    / "deep4net_subject1_low_regularization_history.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "deep4net_subject1_low_regularization_summary.txt"
)

BEST_MODEL_PATH = (
    MODELS_DIR
    / "deep4net_subject1_low_regularization_best.pth"
)

FINAL_MODEL_PATH = (
    MODELS_DIR
    / "deep4net_subject1_low_regularization_final.pth"
)


# ============================================================
# 2. 辅助函数
# ============================================================

def scale_to_microvolts(
    data: np.ndarray,
) -> np.ndarray:
    """
    将 EEG 信号从伏特转换为微伏。

    MNE 默认以伏特保存 EEG 数据。
    Braindecode 深度学习流程通常使用微伏尺度。
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

    batch 级历史是嵌套结构，不写入最终 CSV。
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
    """安全读取训练历史中的浮点指标。"""
    value = row.get(key, np.nan)

    try:
        return float(value)

    except (TypeError, ValueError):
        return float("nan")


def remove_old_output_files() -> None:
    """
    删除同名旧结果。

    防止本次训练异常中断后，程序误把上一次生成的模型
    当成本次实验结果。
    """
    output_paths = [
        HISTORY_PATH,
        SUMMARY_PATH,
        BEST_MODEL_PATH,
        FINAL_MODEL_PATH,
    ]

    for path in output_paths:
        if path.exists():
            path.unlink()
            print(f"Removed old file : {path}")


# ============================================================
# 3. 主程序
# ============================================================

def main() -> None:
    """运行 Deep4Net Subject 1 低正则化实验。"""

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    remove_old_output_files()

    use_cuda = torch.cuda.is_available()

    device = "cuda" if use_cuda else "cpu"

    set_random_seeds(
        seed=SEED,
        cuda=use_cuda,
    )

    print("=" * 76)
    print("Deep4Net Subject 1 Low-Regularization Experiment")
    print("=" * 76)

    print(f"Subject ID       : {SUBJECT_ID}")
    print(f"Device           : {device}")
    print(f"Maximum epochs   : {MAX_EPOCHS}")
    print(f"Early patience   : {EARLY_STOPPING_PATIENCE}")
    print(f"Input samples    : {N_TIMES}")
    print(f"Batch size       : {BATCH_SIZE}")
    print(f"Learning rate    : {LEARNING_RATE}")
    print(f"Weight decay     : {WEIGHT_DECAY}")
    print(f"Dropout          : {DROP_PROBABILITY}")
    print(f"Optimizer        : Adam")
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
        # 仅保留 EEG 通道
        Preprocessor(
            "pick",
            picks="eeg",
        ),

        # 伏特转换为微伏
        Preprocessor(
            scale_to_microvolts,
        ),

        # 4～38 Hz 带通滤波
        Preprocessor(
            "filter",
            l_freq=LOW_CUT_HZ,
            h_freq=HIGH_CUT_HZ,
        ),

        # 指数移动标准化
        Preprocessor(
            exponential_moving_standardize,
            factor_new=FACTOR_NEW,
            init_block_size=INIT_BLOCK_SIZE,
        ),
    ]

    # Windows 环境使用单进程较为稳定
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
    # 6. 创建低正则化 Deep4Net
    # ========================================================

    print("\n[3/8] Creating low-regularization Deep4Net...")

    model = Deep4Net(
        n_chans=N_CHANS,
        n_outputs=N_CLASSES,
        n_times=N_TIMES,

        # 保持 Dense Prediction 设置不变
        final_conv_length=2,

        # 从原来的 0.5 降低到 0.25
        drop_prob=DROP_PROBABILITY,
    )

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
    # 9. 配置最佳模型保存与提前停止
    # ========================================================

    print("\n[6/8] Configuring callbacks...")

    checkpoint = Checkpoint(
        # 只有验证准确率刷新最佳值时才保存
        monitor="valid_accuracy_best",

        dirname=str(MODELS_DIR),

        f_params=BEST_MODEL_PATH.name,

        f_optimizer=None,

        f_criterion=None,

        f_history=None,
    )

    early_stopping = EarlyStopping(
        monitor="valid_accuracy",

        patience=EARLY_STOPPING_PATIENCE,

        threshold=EARLY_STOPPING_THRESHOLD,

        threshold_mode="abs",

        # 准确率越高越好
        lower_is_better=False,

        # 保留 final 模型和 best 模型两个不同文件
        load_best=False,
    )

    # ========================================================
    # 10. 创建 EEGClassifier
    # ========================================================

    print("\n[7/8] Creating EEGClassifier...")

    classifier = EEGClassifier(
        module=model,

        cropped=True,

        criterion=CroppedLoss,

        criterion__loss_function=(
            torch.nn.functional.cross_entropy
        ),

        # 从 AdamW 改为 Adam
        optimizer=torch.optim.Adam,

        optimizer__lr=LEARNING_RATE,

        # 本次实验关闭权重衰减
        optimizer__weight_decay=WEIGHT_DECAY,

        # 0train 训练，1test 验证
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

        classes=list(
            range(N_CLASSES)
        ),

        verbose=1,
    )

    # ========================================================
    # 11. 正式训练
    # ========================================================

    print("\n[8/8] Starting low-regularization training...")

    classifier.fit(
        train_set,
        y=None,
    )

    # ========================================================
    # 12. 保存训练历史与最终模型
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

    # 保存 Early Stopping 触发时的最终模型
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
            "Best model checkpoint was not created: "
            f"{BEST_MODEL_PATH}"
        )

    # ========================================================
    # 13. 提取最佳结果
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
        "Deep4Net Subject 1 Low-Regularization Experiment\n"
        "================================================\n"
        f"Subject ID            : {SUBJECT_ID}\n"
        "Train session         : 0train\n"
        "Validation session    : 1test\n"
        f"Maximum epochs        : {MAX_EPOCHS}\n"
        f"Completed epochs      : {completed_epochs}\n"
        f"Stopped early         : {stopped_early}\n"
        f"Early stop patience   : {EARLY_STOPPING_PATIENCE}\n"
        f"Learning rate         : {LEARNING_RATE}\n"
        f"Optimizer             : Adam\n"
        f"Dropout               : {DROP_PROBABILITY}\n"
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

    print("\n" + "=" * 76)
    print("Low-regularization training completed")
    print("=" * 76)

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