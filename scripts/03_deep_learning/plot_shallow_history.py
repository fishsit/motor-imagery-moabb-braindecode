"""
绘制 ShallowFBCSPNet 训练曲线

读取:
results/metrics/shallowfbcspnet_history.csv

生成:
results/figures/shallowfbcspnet_training_curve.png
"""


import os

import pandas as pd

import matplotlib.pyplot as plt



def main():

    print("="*70)

    print("Plot ShallowFBCSPNet Training Curve")

    print("="*70)



    # 读取日志

    history = pd.read_csv(
        "results/metrics/shallowfbcspnet_history.csv"
    )


    print(history)



    # 创建保存目录

    os.makedirs(
        "results/figures",
        exist_ok=True
    )



    plt.figure(
        figsize=(10,4)
    )


    # =====================
    # Loss
    # =====================

    plt.subplot(
        1,
        2,
        1
    )


    plt.plot(
        history["epoch"],
        history["loss"],
        marker="o"
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Loss"
    )


    plt.title(
        "ShallowFBCSPNet Loss"
    )



    # =====================
    # Accuracy
    # =====================

    plt.subplot(
        1,
        2,
        2
    )


    plt.plot(
        history["epoch"],
        history["train_accuracy"],
        marker="o"
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Accuracy"
    )


    plt.title(
        "ShallowFBCSPNet Accuracy"
    )



    plt.tight_layout()



    plt.savefig(
        "results/figures/shallowfbcspnet_training_curve.png",
        dpi=300
    )


    print()

    print(
        "保存完成:"
    )

    print(
        "results/figures/shallowfbcspnet_training_curve.png"
    )


    print("="*70)



if __name__=="__main__":

    main()