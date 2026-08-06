"""
Generate Deep4Net multi-subject summary.

Read:
    results/metrics/
    deep4net_subject*_runwise_cv_test_metrics.csv

Output:
    results/metrics/
        deep4net_all_subject_summary.csv
        deep4net_overall_summary.txt
"""


from pathlib import Path

import pandas as pd


# ============================================================
# Paths
# ============================================================

METRICS_DIR = Path(
    "results/metrics"
)


OUTPUT_CSV = (
    METRICS_DIR /
    "deep4net_all_subject_summary.csv"
)


OUTPUT_TXT = (
    METRICS_DIR /
    "deep4net_overall_summary.txt"
)


# ============================================================
# Main
# ============================================================


def main():

    print("=" * 80)
    print("Deep4Net Multi Subject Summary")
    print("=" * 80)


    files = sorted(
        METRICS_DIR.glob(
            "deep4net_subject*_runwise_cv_test_metrics.csv"
        )
    )


    if len(files) == 0:

        raise FileNotFoundError(
            "No Deep4Net subject result files found."
        )


    print(
        f"Found {len(files)} subject files"
    )


    results = []


    for file in files:

        print(
            f"Loading: {file.name}"
        )


        df = pd.read_csv(
            file
        )


        results.append(
            df
        )


    summary = pd.concat(
        results,
        ignore_index=True
    )


    # sort by subject id

    summary = summary.sort_values(
        by="subject"
    )


    # save detailed table

    summary.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig"
    )


    print()

    print(
        "Subject Results:"
    )

    print(
        summary[
            [
                "subject",
                "test_accuracy",
                "test_macro_f1",
                "selected_epoch",
                "cv_mean_valid_accuracy",
            ]
        ].to_string(
            index=False
        )
    )


    # ========================================================
    # Overall statistics
    # ========================================================


    mean_accuracy = (
        summary["test_accuracy"]
        .mean()
    )


    std_accuracy = (
        summary["test_accuracy"]
        .std()
    )


    mean_macro_f1 = (
        summary["test_macro_f1"]
        .mean()
    )


    std_macro_f1 = (
        summary["test_macro_f1"]
        .std()
    )


    mean_cv_accuracy = (
        summary["cv_mean_valid_accuracy"]
        .mean()
    )


    txt = f"""
Deep4Net Multi Subject Evaluation
========================================

Subjects evaluated:
{summary['subject'].tolist()}


Test Accuracy
----------------------------------------
Mean accuracy :
{mean_accuracy:.4f}

Std accuracy :
{std_accuracy:.4f}


Test Macro-F1
----------------------------------------
Mean Macro-F1 :
{mean_macro_f1:.4f}

Std Macro-F1 :
{std_macro_f1:.4f}


Cross Validation
----------------------------------------
Mean CV validation accuracy :
{mean_cv_accuracy:.4f}


Detailed results saved:
{OUTPUT_CSV}

"""


    OUTPUT_TXT.write_text(
        txt,
        encoding="utf-8"
    )


    print()

    print(
        "=" * 80
    )

    print(
        "Overall Summary"
    )

    print(
        "=" * 80
    )


    print(
        f"Mean test accuracy : {mean_accuracy:.4f}"
    )

    print(
        f"Std test accuracy  : {std_accuracy:.4f}"
    )

    print(
        f"Mean Macro-F1      : {mean_macro_f1:.4f}"
    )

    print(
        f"Mean CV accuracy   : {mean_cv_accuracy:.4f}"
    )


    print()

    print(
        f"CSV saved : {OUTPUT_CSV}"
    )

    print(
        f"TXT saved : {OUTPUT_TXT}"
    )

    print()

    print(
        "Result: PASS"
    )



if __name__ == "__main__":

    main()