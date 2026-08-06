"""
ShallowFBCSPNet 运动想象分类

深度学习版本CSP
"""


import numpy as np
import pandas as pd

import torch

from torch.utils.data import (
    TensorDataset,
    DataLoader,
    random_split
)


from torch.nn import CrossEntropyLoss

from torch.optim import Adam


from braindecode.models import ShallowFBCSPNet




def main():


    print("="*70)

    print("Train ShallowFBCSPNet")

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


    # 标签0-3

    y=y-1



    X=torch.tensor(
        X,
        dtype=torch.float32
    )


    y=torch.tensor(
        y,
        dtype=torch.long
    )



    print(
        "X:",
        X.shape
    )


    print(
        "y:",
        y.shape
    )



    dataset=TensorDataset(
        X,
        y
    )



    train_size=int(
        0.8*len(dataset)
    )


    test_size=len(dataset)-train_size



    train_dataset,test_dataset=random_split(
        dataset,
        [
            train_size,
            test_size
        ],
        generator=torch.Generator().manual_seed(42)
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
    # 模型
    # =====================


    model=ShallowFBCSPNet(

        n_chans=22,

        n_outputs=4,

        n_times=1001

    )


    print(model)



    loss_fn=CrossEntropyLoss()


    optimizer=Adam(

        model.parameters(),

        lr=0.001

    )



    # =====================
    # 训练
    # =====================


    epochs=20
    history = []

    for epoch in range(epochs):


        model.train()


        total_loss=0

        correct=0

        total=0



        for X_batch,y_batch in train_loader:


            optimizer.zero_grad()


            output=model(
                X_batch
            )


            loss=loss_fn(
                output,
                y_batch
            )


            loss.backward()


            optimizer.step()



            total_loss+=loss.item()



            pred=torch.argmax(
                output,
                dim=1
            )


            correct+=(pred==y_batch).sum().item()


            total+=len(y_batch)



        acc=correct/total



        print(

            f"Epoch {epoch+1}/{epochs}",

            f"Loss:{total_loss:.4f}",

            f"Acc:{acc:.4f}"

        )

        history.append(
            {
                "epoch": epoch + 1,
                "loss": total_loss,
                "train_accuracy": acc
            }
        )

    # =====================
    # 测试
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


            total+=len(y_test)



    print()

    print(
        "Test Accuracy:",
        correct/total
    )

    # =====================
    # 保存训练历史
    # =====================

    history_df = pd.DataFrame(history)

    history_df.to_csv(
        "results/metrics/shallowfbcspnet_history.csv",
        index=False
    )

    print()

    print(
        "训练日志已保存:"
    )

    print(
        "results/metrics/shallowfbcspnet_history.csv"
    )

    # =====================
    # 保存模型
    # =====================

    torch.save(
        model.state_dict(),
        "models/shallowfbcspnet_bnci2014.pth"
    )

    print()

    print(
        "模型已保存:"
    )

    print(
        "models/shallowfbcspnet_bnci2014.pth"
    )

    print("="*70)

    print("完成")



if __name__=="__main__":

    main()