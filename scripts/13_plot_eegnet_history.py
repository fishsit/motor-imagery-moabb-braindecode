"""
绘制 EEGNet 训练曲线

读取:
results/metrics/eegnet_history.csv

生成:
results/figures/eegnet_training_curve.png
"""


import os

import pandas as pd

import matplotlib.pyplot as plt



def main():

    print("="*70)

    print("Plot EEGNet Training Curve")

    print("="*70)



    # =====================
    # 读取训练记录
    # =====================

    history = pd.read_csv(
        "results/metrics/eegnet_history.csv"
    )


    print(history.head())



    # 创建图片目录

    os.makedirs(
        "results/figures",
        exist_ok=True
    )



    # =====================
    # 创建画布
    # =====================

    plt.figure(
        figsize=(10,4)
    )



    # Loss

    plt.subplot(
        1,
        2,
        1
    )


    plt.plot(
        history["epoch"],
        history["loss"]
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Loss"
    )


    plt.title(
        "EEGNet Training Loss"
    )



    # Accuracy

    plt.subplot(
        1,
        2,
        2
    )


    plt.plot(
        history["epoch"],
        history["train_accuracy"]
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Accuracy"
    )


    plt.title(
        "EEGNet Training Accuracy"
    )



    plt.tight_layout()



    # 保存

    plt.savefig(

        "results/figures/eegnet_training_curve.png",

        dpi=300

    )



    print()

    print(
        "保存完成:"
    )

    print(
        "results/figures/eegnet_training_curve.png"
    )



    print("="*70)



if __name__=="__main__":

    main()