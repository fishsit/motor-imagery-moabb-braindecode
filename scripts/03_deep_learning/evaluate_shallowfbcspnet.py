"""
评价 ShallowFBCSPNet

功能:
1. 加载训练好的模型
2. 测试数据预测
3. 分类报告
4. 混淆矩阵
"""


import os

import numpy as np

import torch

import matplotlib.pyplot as plt


from sklearn.metrics import (
    confusion_matrix,
    classification_report
)


from torch.utils.data import (
    TensorDataset,
    DataLoader,
    random_split
)


from braindecode.models import ShallowFBCSPNet




def main():


    print("="*70)

    print("Evaluate ShallowFBCSPNet")

    print("="*70)



    # =====================
    # 数据
    # =====================


    X=np.load(
        "data/X_eeg.npy"
    )


    y=np.load(
        "data/y_label.npy"
    )


    # 标签转换 1-4 -> 0-3

    y=y-1



    X=torch.tensor(
        X,
        dtype=torch.float32
    )


    y=torch.tensor(
        y,
        dtype=torch.long
    )



    dataset=TensorDataset(
        X,
        y
    )



    train_size=int(
        0.8*len(dataset)
    )


    test_size=len(dataset)-train_size



    _,test_dataset=random_split(
        dataset,
        [
            train_size,
            test_size
        ],
        generator=torch.Generator().manual_seed(42)
    )



    test_loader=DataLoader(
        test_dataset,
        batch_size=64
    )



    # =====================
    # 模型
    # =====================


    model=ShallowFBCSPNet(

        n_chans=22,

        n_outputs=4,

        n_times=1001

    )



    model.load_state_dict(
        torch.load(
            "models/shallowfbcspnet_bnci2014.pth"
        )
    )



    model.eval()



    # =====================
    # 预测
    # =====================


    y_true=[]

    y_pred=[]



    with torch.no_grad():


        for X_batch,y_batch in test_loader:


            output=model(
                X_batch
            )


            pred=torch.argmax(
                output,
                dim=1
            )


            y_true.extend(
                y_batch.numpy()
            )


            y_pred.extend(
                pred.numpy()
            )



    # =====================
    # 分类报告
    # =====================


    labels=[

        "left_hand",

        "right_hand",

        "feet",

        "tongue"

    ]



    print()

    print(
        "Classification Report:"
    )


    print(

        classification_report(

            y_true,

            y_pred,

            target_names=labels

        )

    )



    # =====================
    # 混淆矩阵
    # =====================


    cm=confusion_matrix(

        y_true,

        y_pred

    )


    print(
        "Confusion Matrix:"
    )

    print(cm)



    # =====================
    # 绘图
    # =====================


    os.makedirs(

        "results/figures",

        exist_ok=True

    )



    plt.figure(

        figsize=(6,5)

    )


    plt.imshow(
        cm
    )


    plt.colorbar()



    plt.xticks(

        range(4),

        labels,

        rotation=45

    )


    plt.yticks(

        range(4),

        labels

    )


    plt.xlabel(

        "Predicted"

    )


    plt.ylabel(

        "True"

    )


    plt.title(

        "ShallowFBCSPNet Confusion Matrix"

    )



    for i in range(4):

        for j in range(4):

            plt.text(

                j,

                i,

                cm[i,j],

                ha="center",

                va="center"

            )



    plt.tight_layout()



    plt.savefig(

        "results/figures/shallowfbcspnet_confusion_matrix.png",

        dpi=300

    )


    print()

    print(
        "保存完成:"
    )


    print(

        "results/figures/shallowfbcspnet_confusion_matrix.png"

    )


    print("="*70)



if __name__=="__main__":

    main()