"""
LOSO 单折验证：以 Subject 1 为从未见过的测试被试

阶段1：
    Subject 2~9 的 0train -> 内部训练
    Subject 2~9 的 1test  -> 内部验证
    用于选择最佳 epoch

阶段2：
    Subject 2~9 的全部会话 -> 重新训练

阶段3：
    Subject 1 的全部会话 -> 最终测试
"""

from itertools import chain
from pathlib import Path

import numpy as np
import torch
import os

from sklearn.metrics import accuracy_score, confusion_matrix
from skorch.callbacks import EarlyStopping, LRScheduler
from skorch.helper import predefined_split

from braindecode import EEGClassifier
from braindecode.datasets import BaseConcatDataset, MOABBDataset
from braindecode.models import ShallowFBCSPNet
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    exponential_moving_standardize,
    preprocess,
)
from braindecode.training import CroppedLoss
from braindecode.util import set_random_seeds


# ============================================================
# 全局实验配置
# ============================================================

SEED = 20260805

SUBJECT_IDS = list(range(1, 10))
TEST_SUBJECT = int(
    os.environ.get("TEST_SUBJECT", "1")
)

N_CHANS = 22
N_CLASSES = 4
N_TIMES = 1000

MAX_EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.000625

CLASS_NAMES = [
    "feet",
    "left_hand",
    "right_hand",
    "tongue",
]


def merge_datasets(datasets: list[BaseConcatDataset]) -> BaseConcatDataset:
    """
    将多个 BaseConcatDataset 合并为一个 BaseConcatDataset。
    """

    internal_datasets = list(
        chain.from_iterable(
            dataset.datasets
            for dataset in datasets
        )
    )

    return BaseConcatDataset(internal_datasets)


def create_model() -> ShallowFBCSPNet:
    """
    创建新的密集预测 ShallowFBCSPNet。

    每次训练都必须创建全新的模型，不能复用上一阶段权重。
    """

    model = ShallowFBCSPNet(
        n_chans=N_CHANS,
        n_outputs=N_CLASSES,
        n_times=N_TIMES,
        final_conv_length=30,
    )

    model.to_dense_prediction_model()

    return model


def create_classifier(
    model: ShallowFBCSPNet,
    device: str,
    epochs: int,
    valid_set: BaseConcatDataset | None,
    use_early_stopping: bool,
) -> EEGClassifier:
    """
    创建 Cropped EEGClassifier。
    """

    callbacks = [
        (
            "lr_scheduler",
            LRScheduler(
                "CosineAnnealingLR",
                T_max=max(1, epochs - 1),
            ),
        )
    ]

    if valid_set is not None:
        callbacks.insert(0, "accuracy")

    if use_early_stopping:
        callbacks.append(
            (
                "early_stopping",
                EarlyStopping(
                    monitor="valid_accuracy",
                    lower_is_better=False,
                    patience=5,
                    load_best=True,
                ),
            )
        )

    train_split = (
        predefined_split(valid_set)
        if valid_set is not None
        else None
    )

    classifier = EEGClassifier(
        model,
        cropped=True,

        criterion=CroppedLoss,
        criterion__loss_function=(
            torch.nn.functional.cross_entropy
        ),

        optimizer=torch.optim.AdamW,
        optimizer__lr=LEARNING_RATE,
        optimizer__weight_decay=0.0,

        train_split=train_split,

        iterator_train__shuffle=True,
        iterator_train__drop_last=True,

        batch_size=BATCH_SIZE,
        callbacks=callbacks,

        device=device,
        classes=list(range(N_CLASSES)),
    )

    return classifier


def evaluate_trials(
    classifier: EEGClassifier,
    test_set: BaseConcatDataset,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    对 cropped 数据进行 trial-level 评价。

    一个原始 trial 可能对应多个 compute window；
    因此不能把每个 window 当作独立测试样本。

    predict_trials 返回：
        trial_predictions:
            (n_trials, n_classes, n_predictions)

        trial_targets:
            每个原始 trial 的真实标签
    """

    trial_predictions, trial_targets = classifier.predict_trials(
        test_set,
        return_targets=True,
    )

    trial_predictions = np.asarray(trial_predictions)
    y_true = np.asarray(trial_targets).reshape(-1)

    # 对同一个trial中的多个crop预测取平均
    mean_predictions = trial_predictions.mean(axis=-1)

    y_pred = mean_predictions.argmax(axis=1)

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(N_CLASSES)),
    )

    return accuracy, y_true, y_pred, matrix


def main() -> None:
    print("=" * 72)
    print("LOSO Single Fold")
    print(f"Held-out test subject: {TEST_SUBJECT}")
    print("=" * 72)

    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"

    print("Device:", device)

    set_random_seeds(
        seed=SEED,
        cuda=cuda,
    )

    # ============================================================
    # 1. 加载全部9名被试
    # ============================================================

    dataset = MOABBDataset(
        dataset_name="BNCI2014_001",
        subject_ids=SUBJECT_IDS,
    )

    print()
    print("Raw recordings:", len(dataset.datasets))

    # ============================================================
    # 2. 每段记录独立预处理
    # ============================================================

    preprocessors = [
        Preprocessor(
            "pick_types",
            eeg=True,
            meg=False,
            stim=False,
        ),

        # V -> μV
        Preprocessor(
            lambda data: np.multiply(data, 1e6)
        ),

        Preprocessor(
            "filter",
            l_freq=4.0,
            h_freq=38.0,
        ),

        Preprocessor(
            exponential_moving_standardize,
            factor_new=1e-3,
            init_block_size=1000,
        ),
    ]

    print()
    print("Preprocessing...")

    preprocess(
        dataset,
        preprocessors,
        n_jobs=1,
    )

    print("Preprocessing finished.")

    # ============================================================
    # 3. 获取 cropped window 参数
    # ============================================================

    temporary_model = create_model()

    output_shape = temporary_model.get_output_shape()
    n_preds_per_input = output_shape[2]

    print()
    print("Model output shape:", output_shape)
    print("Predictions per input:", n_preds_per_input)

    del temporary_model

    sfreq = dataset.datasets[0].raw.info["sfreq"]

    trial_start_offset_samples = int(
        -0.5 * sfreq
    )

    # ============================================================
    # 4. 创建全部被试的 compute windows
    # ============================================================

    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=(
            trial_start_offset_samples
        ),
        trial_stop_offset_samples=0,
        window_size_samples=N_TIMES,
        window_stride_samples=n_preds_per_input,
        drop_last_window=False,
        preload=True,
    )

    print()
    print("All windows:")
    print(windows_dataset)

    # ============================================================
    # 5. LOSO：按被试划分
    # ============================================================

    subject_splits = windows_dataset.split("subject")

    print()
    print("Subject split keys:", list(subject_splits.keys()))

    test_key = str(TEST_SUBJECT)

    if test_key not in subject_splits:
        raise KeyError(
            f"Cannot find subject key {test_key}. "
            f"Available keys: {list(subject_splits.keys())}"
        )

    test_set = subject_splits[test_key]

    train_subject_ids = [
        subject
        for subject in SUBJECT_IDS
        if subject != TEST_SUBJECT
    ]

    train_subject_sets = [
        subject_splits[str(subject)]
        for subject in train_subject_ids
    ]

    train_pool = merge_datasets(
        train_subject_sets
    )

    print()
    print("Training subjects:", train_subject_ids)
    print("Test subject:", TEST_SUBJECT)

    print("Train-pool windows:", len(train_pool))
    print("Test windows:", len(test_set))

    # ============================================================
    # 6. 阶段1：仅使用训练被试进行epoch选择
    # ============================================================

    session_splits = train_pool.split("session")

    print()
    print(
        "Training-subject session keys:",
        list(session_splits.keys()),
    )

    selection_train_set = session_splits["0train"]
    selection_valid_set = session_splits["1test"]

    print("Selection train windows:", len(selection_train_set))
    print("Selection valid windows:", len(selection_valid_set))

    set_random_seeds(
        seed=SEED,
        cuda=cuda,
    )

    selection_model = create_model()

    selection_classifier = create_classifier(
        model=selection_model,
        device=device,
        epochs=MAX_EPOCHS,
        valid_set=selection_valid_set,
        use_early_stopping=True,
    )

    print()
    print("=" * 72)
    print("Stage 1: select best epoch without test-subject data")
    print("=" * 72)

    selection_classifier.fit(
        selection_train_set,
        y=None,
        epochs=MAX_EPOCHS,
    )

    best_history_row = max(
        selection_classifier.history,
        key=lambda row: row["valid_accuracy"],
    )

    best_epoch = int(
        best_history_row["epoch"]
    )

    best_internal_accuracy = float(
        best_history_row["valid_accuracy"]
    )

    print()
    print("Best internal epoch:", best_epoch)
    print(
        "Best internal validation accuracy:",
        f"{best_internal_accuracy:.4f}",
    )

    # 删除阶段1模型，防止错误复用它的权重
    del selection_classifier
    del selection_model

    if cuda:
        torch.cuda.empty_cache()

    # ============================================================
    # 7. 阶段2：用8名训练被试的全部会话重新训练
    # ============================================================

    set_random_seeds(
        seed=SEED,
        cuda=cuda,
    )

    final_model = create_model()

    final_classifier = create_classifier(
        model=final_model,
        device=device,
        epochs=best_epoch,
        valid_set=None,
        use_early_stopping=False,
    )

    print()
    print("=" * 72)
    print(
        "Stage 2: retrain on all data from the 8 training subjects"
    )
    print("=" * 72)
    print("Final training epochs:", best_epoch)
    print("Final training windows:", len(train_pool))

    final_classifier.fit(
        train_pool,
        y=None,
        epochs=best_epoch,
    )

    # ============================================================
    # 8. 阶段3：只评价从未见过的Subject 1
    # ============================================================

    print()
    print("=" * 72)
    print("Stage 3: evaluate held-out subject")
    print("=" * 72)

    (
        test_accuracy,
        y_true,
        y_pred,
        matrix,
    ) = evaluate_trials(
        final_classifier,
        test_set,
    )

    print()
    print("Held-out subject:", TEST_SUBJECT)
    print("Number of test trials:", len(y_true))
    print("LOSO test accuracy:", f"{test_accuracy:.4f}")

    print()
    print("Confusion matrix")
    print("Class order:", CLASS_NAMES)
    print(matrix)

    # ============================================================
    # 9. 保存单折结果
    # ============================================================

    Path("models").mkdir(
        parents=True,
        exist_ok=True,
    )

    Path("results/metrics").mkdir(
        parents=True,
        exist_ok=True,
    )

    final_classifier.save_params(
        f_params=(
            f"models/loso_subject_{TEST_SUBJECT}_params.pt"
        ),
        f_history=(
            "results/metrics/"
            f"loso_subject_{TEST_SUBJECT}_history.json"
        ),
    )

    result_path = (
        "results/metrics/"
        f"loso_subject_{TEST_SUBJECT}_result.npz"
    )

    np.savez(
        result_path,
        test_subject=TEST_SUBJECT,
        train_subjects=np.asarray(train_subject_ids),
        best_epoch=best_epoch,
        internal_valid_accuracy=best_internal_accuracy,
        test_accuracy=test_accuracy,
        y_true=y_true,
        y_pred=y_pred,
        confusion_matrix=matrix,
        class_names=np.asarray(CLASS_NAMES),
    )

    print()
    print("Model saved:")
    print(
        f"models/loso_subject_{TEST_SUBJECT}_params.pt"
    )

    print("Result saved:")
    print(result_path)

    print("=" * 72)
    print("Single LOSO fold completed")
    print("=" * 72)


if __name__ == "__main__":
    main()