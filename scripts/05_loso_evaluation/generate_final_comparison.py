"""
生成运动想象 BCI 模型最终对比表。

本脚本将实验结果分成两部分：

1. 多被试总体实验
   - CSP + SVM：9被试 LOSO
   - ShallowFBCSPNet：9被试 LOSO
   - Deep4Net：Subject 3～9 跨会话测试

2. 单被试开发实验
   - ShallowFBCSPNet Subject 1
   - Deep4Net Subject 1 原始配置
   - Deep4Net Subject 1 低正则化配置

注意
----
不同模型的评价协议并不完全相同，因此不能只根据 Accuracy
直接得出某个模型绝对优于另一个模型的结论。

EEGNet 当前没有完成与其他模型一致的独立测试或 LOSO 评价，
因此只在说明文件中标记为待补充，不写入正式排名。

输入
----
results/metrics/deep4net_all_subject_summary.csv

输出
----
results/metrics/model_comparison_final.csv
results/metrics/model_comparison_multi_subject.csv
results/metrics/model_comparison_single_subject.csv
results/metrics/model_comparison_final.md
results/metrics/model_comparison_notes.txt

results/figures/model_comparison_accuracy.png
results/figures/model_comparison_macro_f1.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. 路径配置
# ============================================================

METRICS_DIR = Path("results/metrics")
FIGURES_DIR = Path("results/figures")

DEEP4NET_SUMMARY_PATH = (
    METRICS_DIR
    / "deep4net_all_subject_summary.csv"
)

FINAL_COMPARISON_PATH = (
    METRICS_DIR
    / "model_comparison_final.csv"
)

MULTI_SUBJECT_PATH = (
    METRICS_DIR
    / "model_comparison_multi_subject.csv"
)

SINGLE_SUBJECT_PATH = (
    METRICS_DIR
    / "model_comparison_single_subject.csv"
)

MARKDOWN_PATH = (
    METRICS_DIR
    / "model_comparison_final.md"
)

NOTES_PATH = (
    METRICS_DIR
    / "model_comparison_notes.txt"
)

ACCURACY_FIGURE_PATH = (
    FIGURES_DIR
    / "model_comparison_accuracy.png"
)

MACRO_F1_FIGURE_PATH = (
    FIGURES_DIR
    / "model_comparison_macro_f1.png"
)


# ============================================================
# 2. 已确认实验结果
# ============================================================

# CSP + SVM：
# 9被试 Leave-One-Subject-Out 结果
CSP_LOSO_MEAN_ACCURACY = 0.3889
CSP_LOSO_STD_ACCURACY = 0.1329

# 当前记录中没有完整的 CSP LOSO Macro-F1
CSP_LOSO_MACRO_F1 = np.nan
CSP_LOSO_STD_MACRO_F1 = np.nan


# ShallowFBCSPNet：
# 9被试 Leave-One-Subject-Out 结果
SHALLOW_LOSO_MEAN_ACCURACY = 0.4670
SHALLOW_LOSO_STD_ACCURACY = 0.1598
SHALLOW_LOSO_MEAN_MACRO_F1 = 0.4296

# 当前记录中没有保存 Macro-F1 标准差
SHALLOW_LOSO_STD_MACRO_F1 = np.nan


# Subject 1 单被试开发实验
SHALLOW_SUBJECT1_ACCURACY = 0.6493

DEEP4NET_SUBJECT1_BASELINE_ACCURACY = 0.4549
DEEP4NET_SUBJECT1_BASELINE_MACRO_F1 = 0.3780

DEEP4NET_SUBJECT1_LOW_REG_ACCURACY = 0.6389
DEEP4NET_SUBJECT1_LOW_REG_MACRO_F1 = 0.6310


# ============================================================
# 3. 通用辅助函数
# ============================================================

def format_percentage(
    value: float,
) -> str:
    """将小数转换为百分比字符串。"""

    if pd.isna(value):
        return "-"

    return f"{value * 100:.2f}%"


def load_deep4net_summary() -> dict[str, float | int | str]:
    """
    读取 Deep4Net Subject 3～9 汇总结果。

    结果来自：
        deep4net_all_subject_summary.csv
    """

    if not DEEP4NET_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            "Deep4Net summary file was not found: "
            f"{DEEP4NET_SUMMARY_PATH}"
        )

    dataframe = pd.read_csv(
        DEEP4NET_SUMMARY_PATH
    )

    required_columns = {
        "subject",
        "test_accuracy",
        "test_macro_f1",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise KeyError(
            "Deep4Net summary is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if dataframe.empty:
        raise RuntimeError(
            "Deep4Net summary file is empty."
        )

    mean_accuracy = float(
        dataframe["test_accuracy"].mean()
    )

    std_accuracy = float(
        dataframe["test_accuracy"].std()
    )

    mean_macro_f1 = float(
        dataframe["test_macro_f1"].mean()
    )

    std_macro_f1 = float(
        dataframe["test_macro_f1"].std()
    )

    subjects = (
        dataframe["subject"]
        .astype(int)
        .sort_values()
        .tolist()
    )

    return {
        "subjects": subjects,
        "n_subjects": len(subjects),
        "mean_accuracy": mean_accuracy,
        "std_accuracy": std_accuracy,
        "mean_macro_f1": mean_macro_f1,
        "std_macro_f1": std_macro_f1,
    }


# ============================================================
# 4. 创建多被试对比表
# ============================================================

def create_multi_subject_table(
    deep4net_result: dict[str, float | int | str],
) -> pd.DataFrame:
    """
    创建多被试总体实验表。

    注意：
        CSP和Shallow使用LOSO；
        Deep4Net使用被试内跨会话测试。

    因此本表用于汇总，不代表完全公平的模型排名。
    """

    rows = [
        {
            "model": "CSP + SVM",
            "model_type": "Traditional machine learning",
            "evaluation_protocol": (
                "9-subject LOSO"
            ),
            "subjects": "1-9",
            "n_subjects": 9,
            "mean_accuracy": (
                CSP_LOSO_MEAN_ACCURACY
            ),
            "std_accuracy": (
                CSP_LOSO_STD_ACCURACY
            ),
            "mean_macro_f1": (
                CSP_LOSO_MACRO_F1
            ),
            "std_macro_f1": (
                CSP_LOSO_STD_MACRO_F1
            ),
            "comparison_group": (
                "Cross-subject evaluation"
            ),
            "directly_comparable": False,
            "notes": (
                "Leave-one-subject-out cross-subject evaluation."
            ),
        },
        {
            "model": "ShallowFBCSPNet",
            "model_type": "Deep learning",
            "evaluation_protocol": (
                "9-subject LOSO"
            ),
            "subjects": "1-9",
            "n_subjects": 9,
            "mean_accuracy": (
                SHALLOW_LOSO_MEAN_ACCURACY
            ),
            "std_accuracy": (
                SHALLOW_LOSO_STD_ACCURACY
            ),
            "mean_macro_f1": (
                SHALLOW_LOSO_MEAN_MACRO_F1
            ),
            "std_macro_f1": (
                SHALLOW_LOSO_STD_MACRO_F1
            ),
            "comparison_group": (
                "Cross-subject evaluation"
            ),
            "directly_comparable": True,
            "notes": (
                "Same LOSO subject protocol as CSP + SVM."
            ),
        },
        {
            "model": "Deep4Net",
            "model_type": "Deep learning",
            "evaluation_protocol": (
                "Run-wise CV inside 0train, "
                "held-out 1test evaluation"
            ),
            "subjects": ",".join(
                str(subject)
                for subject in deep4net_result[
                    "subjects"
                ]
            ),
            "n_subjects": int(
                deep4net_result[
                    "n_subjects"
                ]
            ),
            "mean_accuracy": float(
                deep4net_result[
                    "mean_accuracy"
                ]
            ),
            "std_accuracy": float(
                deep4net_result[
                    "std_accuracy"
                ]
            ),
            "mean_macro_f1": float(
                deep4net_result[
                    "mean_macro_f1"
                ]
            ),
            "std_macro_f1": float(
                deep4net_result[
                    "std_macro_f1"
                ]
            ),
            "comparison_group": (
                "Within-subject cross-session evaluation"
            ),
            "directly_comparable": False,
            "notes": (
                "Subjects 3-9; evaluation protocol differs "
                "from LOSO."
            ),
        },
    ]

    return pd.DataFrame(rows)


# ============================================================
# 5. 创建单被试开发结果表
# ============================================================

def create_single_subject_table() -> pd.DataFrame:
    """
    创建 Subject 1 开发阶段结果表。

    这些结果用于展示模型调试与优化过程，
    不应与多被试总体结果直接混合排名。
    """

    rows = [
        {
            "model": "ShallowFBCSPNet + EMS + Cropped",
            "subject": 1,
            "evaluation_protocol": (
                "0train to 1test cross-session validation"
            ),
            "accuracy": (
                SHALLOW_SUBJECT1_ACCURACY
            ),
            "macro_f1": np.nan,
            "status": "Development result",
            "notes": (
                "Single-subject result; not LOSO."
            ),
        },
        {
            "model": "Deep4Net baseline",
            "subject": 1,
            "evaluation_protocol": (
                "0train to 1test checkpoint validation"
            ),
            "accuracy": (
                DEEP4NET_SUBJECT1_BASELINE_ACCURACY
            ),
            "macro_f1": (
                DEEP4NET_SUBJECT1_BASELINE_MACRO_F1
            ),
            "status": "Development result",
            "notes": (
                "Dropout=0.5, AdamW, weight decay=5e-4."
            ),
        },
        {
            "model": "Deep4Net low regularization",
            "subject": 1,
            "evaluation_protocol": (
                "0train to 1test checkpoint validation"
            ),
            "accuracy": (
                DEEP4NET_SUBJECT1_LOW_REG_ACCURACY
            ),
            "macro_f1": (
                DEEP4NET_SUBJECT1_LOW_REG_MACRO_F1
            ),
            "status": "Development result",
            "notes": (
                "Dropout=0.25, Adam, weight decay=0."
            ),
        },
        {
            "model": "EEGNet",
            "subject": 1,
            "evaluation_protocol": (
                "Comparable final evaluation not completed"
            ),
            "accuracy": np.nan,
            "macro_f1": np.nan,
            "status": "Pending",
            "notes": (
                "Existing EEGNet training progress is not "
                "sufficient for a protocol-matched final result."
            ),
        },
    ]

    return pd.DataFrame(rows)


# ============================================================
# 6. 生成统一结果表
# ============================================================

def create_combined_table(
    multi_subject_df: pd.DataFrame,
    single_subject_df: pd.DataFrame,
) -> pd.DataFrame:
    """将多被试与单被试结果转换成统一格式。"""

    combined_rows: list[dict[str, object]] = []

    for _, row in multi_subject_df.iterrows():
        combined_rows.append(
            {
                "result_category": (
                    "Multi-subject overall experiment"
                ),
                "model": row["model"],
                "subjects": row["subjects"],
                "evaluation_protocol": (
                    row["evaluation_protocol"]
                ),
                "accuracy": row[
                    "mean_accuracy"
                ],
                "accuracy_std": row[
                    "std_accuracy"
                ],
                "macro_f1": row[
                    "mean_macro_f1"
                ],
                "macro_f1_std": row[
                    "std_macro_f1"
                ],
                "status": "Completed",
                "notes": row["notes"],
            }
        )

    for _, row in single_subject_df.iterrows():
        combined_rows.append(
            {
                "result_category": (
                    "Single-subject development experiment"
                ),
                "model": row["model"],
                "subjects": str(
                    int(row["subject"])
                ),
                "evaluation_protocol": (
                    row["evaluation_protocol"]
                ),
                "accuracy": row["accuracy"],
                "accuracy_std": np.nan,
                "macro_f1": row["macro_f1"],
                "macro_f1_std": np.nan,
                "status": row["status"],
                "notes": row["notes"],
            }
        )

    return pd.DataFrame(
        combined_rows
    )


# ============================================================
# 7. 绘图
# ============================================================

def plot_multi_subject_accuracy(
    multi_subject_df: pd.DataFrame,
) -> None:
    """绘制多被试平均准确率图。"""

    plot_df = multi_subject_df.copy()

    labels = plot_df["model"].tolist()

    values = (
        plot_df["mean_accuracy"]
        .astype(float)
        .to_numpy()
    )

    errors = (
        plot_df["std_accuracy"]
        .astype(float)
        .fillna(0.0)
        .to_numpy()
    )

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    positions = np.arange(
        len(labels)
    )

    bars = axis.bar(
        positions,
        values,
        yerr=errors,
        capsize=5,
    )

    axis.axhline(
        0.25,
        linestyle="--",
        label="Chance level = 25%",
    )

    axis.set_xticks(
        positions,
        labels,
    )

    axis.set_ylim(
        0.0,
        0.8,
    )

    axis.set_ylabel(
        "Mean accuracy"
    )

    axis.set_title(
        "Motor Imagery Model Accuracy Summary"
    )

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    axis.legend()

    for bar, value in zip(
        bars,
        values,
    ):
        axis.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{value * 100:.2f}%",
            ha="center",
            va="bottom",
        )

    figure.tight_layout()

    figure.savefig(
        ACCURACY_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_multi_subject_macro_f1(
    multi_subject_df: pd.DataFrame,
) -> None:
    """绘制存在Macro-F1结果的模型。"""

    plot_df = (
        multi_subject_df
        .dropna(
            subset=["mean_macro_f1"]
        )
        .copy()
    )

    labels = plot_df["model"].tolist()

    values = (
        plot_df["mean_macro_f1"]
        .astype(float)
        .to_numpy()
    )

    figure, axis = plt.subplots(
        figsize=(8, 6)
    )

    positions = np.arange(
        len(labels)
    )

    bars = axis.bar(
        positions,
        values,
    )

    axis.set_xticks(
        positions,
        labels,
    )

    axis.set_ylim(
        0.0,
        0.8,
    )

    axis.set_ylabel(
        "Mean Macro-F1"
    )

    axis.set_title(
        "Motor Imagery Model Macro-F1 Summary"
    )

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    for bar, value in zip(
        bars,
        values,
    ):
        axis.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{value * 100:.2f}%",
            ha="center",
            va="bottom",
        )

    figure.tight_layout()

    figure.savefig(
        MACRO_F1_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# 8. Markdown报告
# ============================================================

def create_markdown_report(
    multi_subject_df: pd.DataFrame,
    single_subject_df: pd.DataFrame,
) -> str:
    """生成可直接用于README或实验报告的Markdown内容。"""

    lines: list[str] = []

    lines.append(
        "# Motor Imagery BCI Model Comparison"
    )

    lines.append("")

    lines.append(
        "## Multi-subject experiments"
    )

    lines.append("")

    lines.append(
        "| Model | Protocol | Subjects | "
        "Accuracy | Accuracy Std | Macro-F1 |"
    )

    lines.append(
        "|---|---|---:|---:|---:|---:|"
    )

    for _, row in multi_subject_df.iterrows():

        lines.append(
            "| "
            f"{row['model']} | "
            f"{row['evaluation_protocol']} | "
            f"{row['subjects']} | "
            f"{format_percentage(row['mean_accuracy'])} | "
            f"{format_percentage(row['std_accuracy'])} | "
            f"{format_percentage(row['mean_macro_f1'])} |"
        )

    lines.append("")

    lines.append(
        "> CSP + SVM and ShallowFBCSPNet use a "
        "leave-one-subject-out protocol. Deep4Net uses "
        "within-subject run-wise cross-validation and a "
        "held-out testing session. The results summarize "
        "different experimental questions and should not "
        "be treated as a completely controlled ranking."
    )

    lines.append("")

    lines.append(
        "## Subject 1 development experiments"
    )

    lines.append("")

    lines.append(
        "| Model | Accuracy | Macro-F1 | Status |"
    )

    lines.append(
        "|---|---:|---:|---|"
    )

    for _, row in single_subject_df.iterrows():

        lines.append(
            "| "
            f"{row['model']} | "
            f"{format_percentage(row['accuracy'])} | "
            f"{format_percentage(row['macro_f1'])} | "
            f"{row['status']} |"
        )

    lines.append("")

    lines.append(
        "## Main observations"
    )

    lines.append("")

    lines.append(
        "1. ShallowFBCSPNet improves the nine-subject "
        "LOSO mean accuracy from 38.89% for CSP + SVM "
        "to 46.70%."
    )

    lines.append("")

    lines.append(
        "2. Deep4Net obtains a mean held-out session "
        "accuracy of 46.23% and a mean Macro-F1 of "
        "41.80% across Subjects 3–9."
    )

    lines.append("")

    lines.append(
        "3. Deep4Net results vary strongly by subject, "
        "with test accuracy ranging from 28.13% to 63.89%."
    )

    lines.append("")

    lines.append(
        "4. EEGNet is not included in the formal comparison "
        "because a protocol-matched final evaluation has "
        "not yet been completed."
    )

    lines.append("")

    return "\n".join(lines)


# ============================================================
# 9. 说明文件
# ============================================================

def create_notes(
    deep4net_result: dict[str, float | int | str],
) -> str:
    """生成评价协议说明。"""

    return (
        "Model Comparison Notes\n"
        "======================\n\n"
        "1. CSP + SVM\n"
        "------------\n"
        "Protocol: Leave-One-Subject-Out across 9 subjects.\n"
        "Mean accuracy: 0.3889.\n"
        "Accuracy standard deviation: 0.1329.\n\n"
        "2. ShallowFBCSPNet\n"
        "-----------------\n"
        "Protocol: Leave-One-Subject-Out across 9 subjects.\n"
        "Mean accuracy: 0.4670.\n"
        "Accuracy standard deviation: 0.1598.\n"
        "Mean Macro-F1: 0.4296.\n\n"
        "3. Deep4Net\n"
        "-----------\n"
        "Protocol: Run-wise cross-validation inside 0train, "
        "then held-out 1test evaluation.\n"
        f"Subjects: {deep4net_result['subjects']}.\n"
        f"Mean accuracy: "
        f"{float(deep4net_result['mean_accuracy']):.4f}.\n"
        f"Accuracy standard deviation: "
        f"{float(deep4net_result['std_accuracy']):.4f}.\n"
        f"Mean Macro-F1: "
        f"{float(deep4net_result['mean_macro_f1']):.4f}.\n"
        f"Macro-F1 standard deviation: "
        f"{float(deep4net_result['std_macro_f1']):.4f}.\n\n"
        "4. EEGNet\n"
        "---------\n"
        "The existing EEGNet records contain training and "
        "development-stage results, but no final evaluation "
        "using a protocol that matches the other completed "
        "experiments. It is therefore marked as pending.\n\n"
        "5. Interpretation\n"
        "-----------------\n"
        "CSP + SVM and ShallowFBCSPNet can be compared more "
        "directly because both use the same LOSO protocol.\n"
        "Deep4Net answers a different question: "
        "within-subject generalization from 0train to 1test.\n"
        "Its accuracy should not be interpreted as directly "
        "higher or lower than the LOSO results without a "
        "protocol-matched experiment.\n"
    )


# ============================================================
# 10. 主程序
# ============================================================

def main() -> None:
    """生成最终模型对比结果。"""

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("Generating Final Model Comparison")
    print("=" * 80)

    print(
        f"Loading Deep4Net results: "
        f"{DEEP4NET_SUMMARY_PATH}"
    )

    deep4net_result = (
        load_deep4net_summary()
    )

    multi_subject_df = (
        create_multi_subject_table(
            deep4net_result
        )
    )

    single_subject_df = (
        create_single_subject_table()
    )

    combined_df = create_combined_table(
        multi_subject_df,
        single_subject_df,
    )

    multi_subject_df.to_csv(
        MULTI_SUBJECT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    single_subject_df.to_csv(
        SINGLE_SUBJECT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    combined_df.to_csv(
        FINAL_COMPARISON_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    markdown_content = (
        create_markdown_report(
            multi_subject_df,
            single_subject_df,
        )
    )

    MARKDOWN_PATH.write_text(
        markdown_content,
        encoding="utf-8",
    )

    notes_content = create_notes(
        deep4net_result
    )

    NOTES_PATH.write_text(
        notes_content,
        encoding="utf-8",
    )

    plot_multi_subject_accuracy(
        multi_subject_df
    )

    plot_multi_subject_macro_f1(
        multi_subject_df
    )

    print("\nMulti-subject summary:")
    print(
        multi_subject_df[
            [
                "model",
                "evaluation_protocol",
                "mean_accuracy",
                "std_accuracy",
                "mean_macro_f1",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nSingle-subject development summary:")
    print(
        single_subject_df[
            [
                "model",
                "accuracy",
                "macro_f1",
                "status",
            ]
        ].to_string(
            index=False
        )
    )

    print("\n" + "=" * 80)
    print("Final comparison generated")
    print("=" * 80)

    print(
        f"Combined CSV     : {FINAL_COMPARISON_PATH}"
    )

    print(
        f"Multi-subject CSV: {MULTI_SUBJECT_PATH}"
    )

    print(
        f"Single-subject CSV: {SINGLE_SUBJECT_PATH}"
    )

    print(
        f"Markdown report  : {MARKDOWN_PATH}"
    )

    print(
        f"Protocol notes   : {NOTES_PATH}"
    )

    print(
        f"Accuracy figure  : {ACCURACY_FIGURE_PATH}"
    )

    print(
        f"Macro-F1 figure  : {MACRO_F1_FIGURE_PATH}"
    )

    print("\nResult: PASS")


if __name__ == "__main__":
    main()