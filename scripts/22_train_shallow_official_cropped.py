"""使用 Braindecode 官方流程训练 Cropped ShallowFBCSPNet。"""

from pathlib import Path

import numpy as np
import torch
from skorch.callbacks import EarlyStopping, LRScheduler
from skorch.helper import predefined_split

from braindecode import EEGClassifier
from braindecode.datasets import MOABBDataset
from braindecode.models import ShallowFBCSPNet
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    exponential_moving_standardize,
    preprocess,
)
from braindecode.training import CroppedLoss
from braindecode.util import set_random_seeds


def main() -> None:
    print("=" * 70)
    print("官方 Cropped ShallowFBCSPNet")
    print("=" * 70)

    # 第一次先使用被试 1，便于验证官方流程。
    subject_id = 1
    seed = 20260805

    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"

    print(f"设备：{device}")
    print(f"被试：{subject_id}")

    set_random_seeds(seed=seed, cuda=cuda)

    # ============================================================
    # 1. 加载 MOABB 数据
    # ============================================================

    dataset = MOABBDataset(
        dataset_name="BNCI2014_001",
        subject_ids=[subject_id],
    )

    # ============================================================
    # 2. 官方预处理流程
    # ============================================================

    low_cut_hz = 4.0
    high_cut_hz = 38.0
    factor_new = 1e-3
    init_block_size = 1000
    volts_to_microvolts = 1e6

    preprocessors = [
        # 只保留 EEG，去除 EOG 和 STI。
        Preprocessor(
            "pick_types",
            eeg=True,
            meg=False,
            stim=False,
        ),

        # MNE 默认单位是 V，这里转换为 μV。
        Preprocessor(
            lambda data: np.multiply(data, volts_to_microvolts)
        ),

        # 官方示例采用 4～38 Hz。
        Preprocessor(
            "filter",
            l_freq=low_cut_hz,
            h_freq=high_cut_hz,
        ),

        # 指数移动标准化。
        Preprocessor(
            exponential_moving_standardize,
            factor_new=factor_new,
            init_block_size=init_block_size,
        ),
    ]

    print("\n开始预处理……")

    preprocess(
        dataset,
        preprocessors,
        n_jobs=1,
    )

    print("预处理完成。")
    print(dataset)

    # ============================================================
    # 3. 创建支持密集预测的模型
    # ============================================================

    n_chans = 22
    n_classes = 4
    classes = list(range(n_classes))

    # 输入计算窗口长度：1000 点，即 4 秒。
    n_times = 1000

    model = ShallowFBCSPNet(
        n_chans=n_chans,
        n_outputs=n_classes,
        n_times=n_times,
        final_conv_length=30,
    )

    # 将普通模型转换为密集预测模型。
    model.to_dense_prediction_model()

    # 每个计算窗口能够产生多少个 crop 预测。
    output_shape = model.get_output_shape()
    n_preds_per_input = output_shape[2]

    print("\n模型输出形状：", output_shape)
    print("每个输入窗口的预测数量：", n_preds_per_input)

    # ============================================================
    # 4. 从连续 Raw 中创建计算窗口
    # ============================================================

    sfreq = dataset.datasets[0].raw.info["sfreq"]

    assert all(
        ds.raw.info["sfreq"] == sfreq
        for ds in dataset.datasets
    )

    # 从提示前 0.5 秒开始截取，与官方教程一致。
    trial_start_offset_seconds = -0.5
    trial_start_offset_samples = int(
        trial_start_offset_seconds * sfreq
    )

    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=0,
        window_size_samples=n_times,
        window_stride_samples=n_preds_per_input,
        drop_last_window=False,
        preload=True,
    )

    print("\n窗口数据集：")
    print(windows_dataset)

    # ============================================================
    # 5. 按原始会话划分训练集和验证集
    # ============================================================

    splits = windows_dataset.split("session")

    print("\n可用会话：", list(splits.keys()))

    train_set = splits["0train"]
    valid_set = splits["1test"]

    print("训练窗口数量：", len(train_set))
    print("验证窗口数量：", len(valid_set))

    # ============================================================
    # 6. EEGClassifier + CroppedLoss
    # ============================================================

    learning_rate = 0.0625 * 0.01
    batch_size = 64
    n_epochs = 10

    classifier = EEGClassifier(
        model,
        cropped=True,
        criterion=CroppedLoss,
        criterion__loss_function=torch.nn.functional.cross_entropy,
        optimizer=torch.optim.AdamW,
        train_split=predefined_split(valid_set),
        optimizer__lr=learning_rate,
        optimizer__weight_decay=0,
        iterator_train__shuffle=True,
        iterator_train__drop_last=True,
        batch_size=batch_size,
        callbacks=[
            "accuracy",
            (
                "lr_scheduler",
                LRScheduler(
                    "CosineAnnealingLR",
                    T_max=max(1, n_epochs - 1),
                ),
            ),
            (
                "early_stopping",
                EarlyStopping(
                    patience=5,
                    load_best=True,
                ),
            ),
        ],
        device=device,
        classes=classes,
    )

    print("\n开始 Cropped Training……")

    classifier.fit(
        train_set,
        y=None,
        epochs=n_epochs,
    )

    # ============================================================
    # 7. 输出最佳结果并保存
    # ============================================================

    history = classifier.history.to_list()

    best_valid_accuracy = max(
        row.get("valid_accuracy", 0.0)
        for row in history
    )

    print("\n最佳验证准确率：")
    print(f"{best_valid_accuracy:.4f}")

    Path("models").mkdir(exist_ok=True)
    Path("results/metrics").mkdir(parents=True, exist_ok=True)

    classifier.save_params(
        f_params="models/shallow_official_cropped_params.pt",
        f_history="results/metrics/shallow_official_cropped_history.json",
    )

    print("\n模型参数已保存：")
    print("models/shallow_official_cropped_params.pt")

    print("训练历史已保存：")
    print("results/metrics/shallow_official_cropped_history.json")

    print("=" * 70)
    print("完成")
    print("=" * 70)


if __name__ == "__main__":
    main()