"""
Deep4Net 单被试跨会话 Cropped Training 冒烟测试。

实验协议：
    Subject 1
    0train -> 训练集
    1test  -> 验证集

当前仅训练 1 个 epoch，用于验证：
1. 数据读取和预处理是否正常；
2. Deep4Net 能否处理真实 EEG；
3. CroppedLoss 和 EEGClassifier 是否能够正常训练；
4. 模型和训练历史是否能够保存。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
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

# 冒烟测试先运行 1 个 epoch
N_EPOCHS = 1

# CPU 环境下先使用较小 batch
BATCH_SIZE = 32

# Deep4Net 官方示例建议的训练量级
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.0005

RESULTS_DIR = Path("results/metrics")
MODELS_DIR = Path("models")

HISTORY_PATH = (
    RESULTS_DIR
    / "deep4net_subject1_smoke_history.csv"
)

MODEL_PATH = (
    MODELS_DIR
    / "deep4net_subject1_smoke.pth"
)


# ============================================================
# 2. 辅助函数
# ============================================================

def scale_to_microvolts(data: np.ndarray) -> np.ndarray:
    """
    将 MNE 中以伏特存储的 EEG 转换为微伏。

    Parameters
    ----------
    data:
        EEG 数组。

    Returns
    -------
    np.ndarray
        转换为微伏后的 EEG。
    """
    return data * 1e6


def count_trainable_parameters(
    model: torch.nn.Module,
) -> int:
    """统计模型中可训练参数数量。"""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# ============================================================
# 3. 主流程
# ============================================================

def main() -> None:
    """运行 Deep4Net 单被试冒烟测试。"""

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

    print("=" * 70)
    print("Deep4Net single-subject cropped smoke test")
    print("=" * 70)

    print(f"Subject ID       : {SUBJECT_ID}")
    print(f"Device           : {device}")
    print(f"Epochs           : {N_EPOCHS}")
    print(f"Input samples    : {N_TIMES}")
    print(f"Batch size       : {BATCH_SIZE}")

    # --------------------------------------------------------
    # 4. 加载 BNCI2014_001
    # --------------------------------------------------------

    print("\n[1/7] Loading BNCI2014_001...")

    dataset = MOABBDataset(
        dataset_name="BNCI2014_001",
        subject_ids=[SUBJECT_ID],
    )

    print(dataset)

    # --------------------------------------------------------
    # 5. EEG 预处理
    # --------------------------------------------------------

    print("\n[2/7] Preprocessing EEG...")

    preprocessors = [
        Preprocessor(
            "pick_types",
            eeg=True,
            meg=False,
            stim=False,
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

    # Windows 下使用 n_jobs=1 更稳定
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

    # --------------------------------------------------------
    # 6. 创建 Deep4Net
    # --------------------------------------------------------

    print("\n[3/7] Creating Deep4Net...")

    model = Deep4Net(
        n_chans=N_CHANS,
        n_outputs=N_CLASSES,
        n_times=N_TIMES,

        # 必须设为较小值，才能产生多个时间位置预测
        final_conv_length=2,

        drop_prob=0.5,
    )

    model.to_dense_prediction_model()

    output_shape = model.get_output_shape()
    n_preds_per_input = int(output_shape[2])

    trainable_params = count_trainable_parameters(
        model
    )

    print(f"Model output     : {output_shape}")
    print(f"Predictions/input: {n_preds_per_input}")
    print(f"Trainable params : {trainable_params:,}")

    if n_preds_per_input <= 1:
        raise RuntimeError(
            "Deep4Net did not produce dense predictions."
        )

    # --------------------------------------------------------
    # 7. 创建 Cropped Windows
    # --------------------------------------------------------

    print("\n[4/7] Creating cropped windows...")

    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        window_size_samples=N_TIMES,

        # 官方 cropped 流程使用每个输入的预测数量作为步长
        window_stride_samples=n_preds_per_input,

        drop_last_window=False,
        preload=True,
    )

    print(windows_dataset)

    # --------------------------------------------------------
    # 8. 按会话划分
    # --------------------------------------------------------

    print("\n[5/7] Splitting train and validation sessions...")

    split_datasets = windows_dataset.split(
        "session"
    )

    print(
        "Available sessions:",
        list(split_datasets.keys()),
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
        "First input shape:",
        tuple(sample_x.shape),
    )

    print(
        "First target:",
        int(sample_y),
    )

    print(
        "First window index:",
        sample_ind,
    )

    if tuple(sample_x.shape) != (
        N_CHANS,
        N_TIMES,
    ):
        raise RuntimeError(
            "Unexpected input shape: "
            f"{tuple(sample_x.shape)}"
        )

    # --------------------------------------------------------
    # 9. 创建 EEGClassifier
    # --------------------------------------------------------

    print("\n[6/7] Creating EEGClassifier...")

    classifier = EEGClassifier(
        module=model,

        cropped=True,

        criterion=CroppedLoss,
        criterion__loss_function=(
            torch.nn.functional.cross_entropy
        ),

        optimizer=torch.optim.AdamW,
        optimizer__lr=LEARNING_RATE,
        optimizer__weight_decay=WEIGHT_DECAY,

        train_split=predefined_split(valid_set),

        batch_size=BATCH_SIZE,
        max_epochs=N_EPOCHS,

        iterator_train__shuffle=True,
        iterator_train__drop_last=False,
        iterator_valid__drop_last=False,

        callbacks=[
            "accuracy",
        ],

        device=device,

        # 显式提供类别，避免 skorch 无法推断 classes_
        classes=list(range(N_CLASSES)),

        verbose=1,
    )

    # --------------------------------------------------------
    # 10. 冒烟训练
    # --------------------------------------------------------

    print("\n[7/7] Starting one-epoch smoke training...")

    classifier.fit(
        train_set,
        y=None,
    )

    # --------------------------------------------------------
    # 11. 保存训练历史
    # --------------------------------------------------------

    history_df = pd.DataFrame(
        classifier.history.to_list()
    )

    history_df.to_csv(
        HISTORY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    torch.save(
        classifier.module_.state_dict(),
        MODEL_PATH,
    )

    print("\n" + "=" * 70)
    print("Smoke test completed")
    print("=" * 70)

    print(f"History saved to : {HISTORY_PATH}")
    print(f"Model saved to   : {MODEL_PATH}")

    if len(classifier.history) > 0:
        last_record = classifier.history[-1]

        print(
            "Train loss       :",
            last_record.get("train_loss"),
        )

        print(
            "Valid loss       :",
            last_record.get("valid_loss"),
        )

        print(
            "Train accuracy   :",
            last_record.get("train_accuracy"),
        )

        print(
            "Valid accuracy   :",
            last_record.get("valid_accuracy"),
        )

    print("\nResult: PASS")


if __name__ == "__main__":
    main()