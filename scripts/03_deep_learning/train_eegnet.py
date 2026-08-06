"""
EEGNet 深度学习训练

功能:
1. 加载EEG numpy数据
2. 创建DataLoader
3. 训练EEGNet
4. 输出准确率
"""


import numpy as np

import pandas as pd

from sklearn.preprocessing import StandardScaler

import torch

from torch.utils.data import (
    TensorDataset,
    DataLoader,
    random_split
)


from braindecode.models import EEGNet


from torch.nn import CrossEntropyLoss


from torch.optim import Adam



def main():


    print("="*70)

    print("Train EEGNet")

    print("="*70)



    # =====================
    # 1. 读取数据
    # =====================

    X = np.load(
        "data/X_eeg.npy"
    )

    y = np.load(
        "data/y_label.npy"
    )

    print(
        "Before normalization:",
        X.shape
    )

    # =====================
    # EEG标准化
    # =====================

    scaler = StandardScaler()

    # 对每个通道进行标准化

    # =====================
    # EEG标准化
    # =====================

    scaler = StandardScaler()

    # 转换:
    # (samples, channels, time)
    # ->
    # (samples, time, channels)

    X = X.transpose(
        0,
        2,
        1
    )

    # 将每个通道作为特征进行标准化

    X = scaler.fit_transform(
        X.reshape(
            -1,
            X.shape[-1]
        )
    ).reshape(
        X.shape
    )

    # 转回:
    # (samples, channels, time)

    X = X.transpose(
        0,
        2,
        1
    )

    print(
        "After normalization:",
        X.shape
    )
    # EEG分类标签转换为0-3

    y = y - 1

    print(
        "Data:",
        X.shape
    )



    # =====================
    # 2. numpy -> tensor
    # =====================


    X=torch.tensor(

        X,

        dtype=torch.float32

    )


    y=torch.tensor(

        y,

        dtype=torch.long

    )



    # CNN输入:

    # X=X.unsqueeze(1)
    # EEGNet直接接受:
    # (samples, channels, time)

    # 不需要unsqueeze


    # =====================
    # 3. Dataset
    # =====================


    dataset=TensorDataset(

        X,

        y

    )



    # 划分训练测试

    train_size=int(
        0.8*len(dataset)
    )


    test_size=len(dataset)-train_size



    train_dataset,test_dataset=random_split(

        dataset,

        [
            train_size,
            test_size
        ]

    )



    train_loader=DataLoader(

        train_dataset,

        batch_size=64,

        shuffle=True

    )


    test_loader=DataLoader(

        test_dataset,

        batch_size=64

    )



    # =====================
    # 4. 创建EEGNet
    # =====================


    model=EEGNet(

        n_chans=22,

        n_outputs=4,

        n_times=1001

    )



    print(model)



    # =====================
    # 5. 损失函数
    # =====================


    loss_fn=CrossEntropyLoss()



    optimizer=Adam(

        model.parameters(),

        lr=0.001

    )



    # =====================
    # 6.训练
    # =====================


    epochs=20

    history = []

    for epoch in range(epochs):


        model.train()


        total_loss=0


        correct=0

        total=0



        for batch_X,batch_y in train_loader:


            optimizer.zero_grad()



            output=model(
                batch_X
            )



            loss=loss_fn(

                output,

                batch_y

            )



            loss.backward()


            optimizer.step()



            total_loss+=loss.item()



            pred=torch.argmax(

                output,

                dim=1

            )



            correct+=(pred==batch_y).sum().item()


            total+=batch_y.size(0)



        train_acc=correct/total



        print(

            f"Epoch {epoch+1}/{epochs}",

            f"Loss:{total_loss:.4f}",

            f"Train Acc:{train_acc:.4f}"

        )

        history.append(
            {
                "epoch": epoch + 1,
                "loss": total_loss,
                "train_accuracy": train_acc
            }
        )
    # =====================
    # 7.测试
    # =====================


    model.eval()


    correct=0

    total=0



    with torch.no_grad():


        for X_test,y_test in test_loader:


            output=model(
                X_test
            )


            pred=torch.argmax(

                output,

                dim=1

            )


            correct+=(pred==y_test).sum().item()


            total+=y_test.size(0)



    test_acc=correct/total



    print()

    print(
        "Test Accuracy:",
        test_acc
    )

    # 保存训练历史

    history_df = pd.DataFrame(history)

    history_df.to_csv(
        "results/metrics/eegnet_history.csv",
        index=False
    )

    print(
        "训练日志已保存:"
    )

    print(
        "results/metrics/eegnet_history.csv"
    )
    # =====================
    # 保存模型
    # =====================

    torch.save(
        model.state_dict(),
        "models/eegnet_bnci2014.pth"
    )

    print(
        "模型已保存:"
    )

    print(
        "models/eegnet_bnci2014.pth"
    )

    print("="*70)

    print("完成")

    print("="*70)



if __name__=="__main__":

    main()