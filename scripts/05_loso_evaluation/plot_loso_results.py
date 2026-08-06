"""
LOSO实验结果可视化

生成:
1. 每个subject准确率柱状图
2. 总混淆矩阵
3. 模型比较表
"""


from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay



RESULT_DIR = Path(
    "results"
)

METRIC_DIR = RESULT_DIR / "metrics"

FIG_DIR = RESULT_DIR / "figures"



def main():


    FIG_DIR.mkdir(
        exist_ok=True
    )


    print("="*70)

    print(
        "Plot LOSO Results"
    )

    print("="*70)



    # ==========================
    # 1. 读取LOSO结果
    # ==========================


    data=np.load(

        METRIC_DIR /
        "loso_all_results.npz",

        allow_pickle=True

    )


    subjects=data["subjects"]

    accuracies=data["test_accuracies"]


    mean_accuracy=float(
        data["mean_accuracy"]
    )


    std_accuracy=float(
        data["std_accuracy"]
    )


    confusion=data[
        "aggregate_confusion_matrix"
    ]



    print()

    print(
        "Mean Accuracy:",
        mean_accuracy
    )

    print(
        "STD:",
        std_accuracy
    )



    # ==========================
    # 2. Subject准确率
    # ==========================


    plt.figure(
        figsize=(8,5)
    )


    plt.bar(

        subjects,

        accuracies

    )


    plt.axhline(

        mean_accuracy,

        linestyle="--",

        label=
        f"Mean={mean_accuracy:.3f}"

    )


    plt.xlabel(
        "Subject"
    )


    plt.ylabel(
        "Accuracy"
    )


    plt.title(
        "LOSO Cross Subject Accuracy"
    )


    plt.xticks(
        subjects
    )


    plt.legend()


    plt.grid()



    plt.savefig(

        FIG_DIR /
        "loso_accuracy_subjects.png",

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()



    # ==========================
    # 3. 混淆矩阵
    # ==========================


    class_names=[

        "feet",

        "left_hand",

        "right_hand",

        "tongue"

    ]



    disp=ConfusionMatrixDisplay(

        confusion_matrix=confusion,

        display_labels=class_names

    )


    fig,ax=plt.subplots(

        figsize=(6,6)

    )


    disp.plot(

        ax=ax,

        cmap="Blues"

    )


    plt.title(

        "LOSO Aggregate Confusion Matrix"

    )


    plt.savefig(

        FIG_DIR /

        "loso_confusion_matrix.png",

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()



    # ==========================
    # 4. 模型比较表
    # ==========================


    comparison=pd.DataFrame(

        {

        "Model":[

            "CSP+SVM",

            "ShallowFBCSPNet+EMS+Cropped"

        ],


        "Accuracy":[

            0.3889,

            mean_accuracy

        ],


        "STD":[

            0.1329,

            std_accuracy

        ]

        }

    )



    comparison.to_csv(

        METRIC_DIR /

        "model_comparison.csv",

        index=False

    )



    print()

    print(
        "Generated:"
    )

    print(

        FIG_DIR /

        "loso_accuracy_subjects.png"

    )


    print(

        FIG_DIR /

        "loso_confusion_matrix.png"

    )


    print(

        METRIC_DIR /

        "model_comparison.csv"

    )


    print("="*70)



if __name__=="__main__":

    main()