"""
顺序运行 BNCI2014_001 的完整9折LOSO实验。

特点：
1. 调用 27_loso_single_fold.py 完成每一折；
2. 已有结果自动跳过，支持中断后继续；
3. 汇总 Accuracy、Balanced Accuracy、Macro-F1；
4. 汇总9折混淆矩阵；
5. 保存 CSV 和 NPZ 结果。
"""

import csv
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
)


SUBJECT_IDS = list(range(1, 10))

SINGLE_FOLD_SCRIPT = Path(
    "scripts/27_loso_single_fold.py"
)

RESULT_DIR = Path(
    "results/metrics"
)

SUMMARY_CSV = RESULT_DIR / "loso_summary.csv"
SUMMARY_NPZ = RESULT_DIR / "loso_all_results.npz"


def run_missing_folds() -> None:
    """运行尚未完成的LOSO折。"""

    for subject in SUBJECT_IDS:
        result_path = (
            RESULT_DIR /
            f"loso_subject_{subject}_result.npz"
        )

        print()
        print("=" * 72)
        print(f"LOSO fold: held-out Subject {subject}")
        print("=" * 72)

        if result_path.exists():
            print("Result already exists, skipping:")
            print(result_path)
            continue

        environment = os.environ.copy()
        environment["TEST_SUBJECT"] = str(subject)
        environment["PYTHONUNBUFFERED"] = "1"

        subprocess.run(
            [
                sys.executable,
                str(SINGLE_FOLD_SCRIPT),
            ],
            env=environment,
            check=True,
        )


def load_and_summarize() -> None:
    """读取所有折并计算汇总指标。"""

    rows = []
    confusion_matrices = []

    for subject in SUBJECT_IDS:
        result_path = (
            RESULT_DIR /
            f"loso_subject_{subject}_result.npz"
        )

        if not result_path.exists():
            raise FileNotFoundError(
                f"Missing LOSO result: {result_path}"
            )

        with np.load(
            result_path,
            allow_pickle=False,
        ) as result:
            best_epoch = int(
                result["best_epoch"].item()
            )

            internal_valid_accuracy = float(
                result[
                    "internal_valid_accuracy"
                ].item()
            )

            test_accuracy = float(
                result["test_accuracy"].item()
            )

            y_true = result["y_true"]
            y_pred = result["y_pred"]

            matrix = result[
                "confusion_matrix"
            ].astype(np.int64)

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

        rows.append(
            {
                "subject": subject,
                "best_epoch": best_epoch,
                "internal_valid_accuracy": (
                    internal_valid_accuracy
                ),
                "test_accuracy": test_accuracy,
                "balanced_accuracy": (
                    balanced_accuracy
                ),
                "macro_f1": macro_f1,
            }
        )

        confusion_matrices.append(matrix)

    accuracies = np.asarray(
        [
            row["test_accuracy"]
            for row in rows
        ],
        dtype=np.float64,
    )

    balanced_accuracies = np.asarray(
        [
            row["balanced_accuracy"]
            for row in rows
        ],
        dtype=np.float64,
    )

    macro_f1_scores = np.asarray(
        [
            row["macro_f1"]
            for row in rows
        ],
        dtype=np.float64,
    )

    aggregate_confusion_matrix = np.sum(
        confusion_matrices,
        axis=0,
    )

    mean_accuracy = accuracies.mean()
    std_accuracy = accuracies.std(ddof=0)

    mean_balanced_accuracy = (
        balanced_accuracies.mean()
    )

    mean_macro_f1 = macro_f1_scores.mean()

    # 保存逐被试CSV
    with SUMMARY_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        fieldnames = [
            "subject",
            "best_epoch",
            "internal_valid_accuracy",
            "test_accuracy",
            "balanced_accuracy",
            "macro_f1",
        ]

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    # 保存便于后续绘图和统计的NPZ
    np.savez(
        SUMMARY_NPZ,
        subjects=np.asarray(SUBJECT_IDS),
        best_epochs=np.asarray(
            [
                row["best_epoch"]
                for row in rows
            ]
        ),
        internal_valid_accuracies=np.asarray(
            [
                row[
                    "internal_valid_accuracy"
                ]
                for row in rows
            ]
        ),
        test_accuracies=accuracies,
        balanced_accuracies=(
            balanced_accuracies
        ),
        macro_f1_scores=macro_f1_scores,
        mean_accuracy=mean_accuracy,
        std_accuracy=std_accuracy,
        mean_balanced_accuracy=(
            mean_balanced_accuracy
        ),
        mean_macro_f1=mean_macro_f1,
        aggregate_confusion_matrix=(
            aggregate_confusion_matrix
        ),
        class_names=np.asarray(
            [
                "feet",
                "left_hand",
                "right_hand",
                "tongue",
            ]
        ),
    )

    print()
    print("=" * 72)
    print("Complete 9-fold LOSO summary")
    print("=" * 72)

    for row in rows:
        print(
            f"Subject {row['subject']}: "
            f"Accuracy={row['test_accuracy']:.4f}, "
            f"Balanced={row['balanced_accuracy']:.4f}, "
            f"Macro-F1={row['macro_f1']:.4f}, "
            f"Epoch={row['best_epoch']}"
        )

    print()
    print(
        "Mean LOSO accuracy:",
        f"{mean_accuracy:.4f}",
    )

    print(
        "LOSO accuracy std:",
        f"{std_accuracy:.4f}",
    )

    print(
        "Mean balanced accuracy:",
        f"{mean_balanced_accuracy:.4f}",
    )

    print(
        "Mean macro-F1:",
        f"{mean_macro_f1:.4f}",
    )

    print()
    print("Aggregate confusion matrix:")
    print(aggregate_confusion_matrix)

    print()
    print("Summary CSV:")
    print(SUMMARY_CSV)

    print("Summary NPZ:")
    print(SUMMARY_NPZ)


def main() -> None:
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not SINGLE_FOLD_SCRIPT.exists():
        raise FileNotFoundError(
            f"Cannot find {SINGLE_FOLD_SCRIPT}"
        )

    run_missing_folds()
    load_and_summarize()


if __name__ == "__main__":
    main()