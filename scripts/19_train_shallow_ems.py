"""
ShallowFBCSPNet + ExponentialMovingStandardize

实验:
比较 EMS 对运动想象 EEG 分类的影响
"""


import os

import numpy as np

import torch

from torch.utils.data import (
    TensorDataset,
    DataLoader,
    random_split
)

from torch.nn import CrossEntropyLoss

from torch.optim import Adam


from braindecode.models import ShallowFBCSPNet


from braindecode.preprocessing import (
    exponential_moving_standardize
)



def main():


    print("="*70)

    print("Train ShallowFBCSPNet + EMS")

    print("="*70)



    # =====================
    # 加载数据
    # =====================


    X=np.load(
        "data/X_eeg.npy"
    )


    y=np.load(
        "data/y_label.npy"
    )


    print(
        "Original X:",
        X.shape
    )



    # =====================
    # EMS标准化
    # =====================


    print()

    print(
        "Applying ExponentialMovingStandardize..."
    )


    for i in range(len(X)):


        X[i]=exponential_moving_standardize(

            X[i],

            factor_new=0.001,

            init_block_size=1000

        )



    print(
        "EMS finished"
    )



    print(
        "After EMS mean:",
        X.mean()
    )


    print(
        "After EMS std:",
        X.std()
    )



    # 标签

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



    loss_fn=CrossEntropyLoss()


    optimizer=Adam(

        model.parameters(),

        lr=0.001

    )



    # =====================
    # training
    # =====================


    epochs=20


    history=[]



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



        history.append(

            {

                "epoch":epoch+1,

                "loss":total_loss,

                "train_accuracy":acc

            }

        )



        print(

            f"Epoch {epoch+1}/{epochs}",

            f"Loss:{total_loss:.4f}",

            f"Acc:{acc:.4f}"

        )



    # =====================
    # test
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



    acc=correct/total



    print()

    print(

        "Test Accuracy:",

        acc

    )



    # =====================
    # 保存
    # =====================


    os.makedirs(

        "models",

        exist_ok=True

    )


    torch.save(

        model.state_dict(),

        "models/shallowfbcspnet_ems_bnci2014.pth"

    )



    print()

    print(

        "Model saved:"

    )

    print(

        "models/shallowfbcspnet_ems_bnci2014.pth"

    )



    print("="*70)

    print("完成")



if __name__=="__main__":

    main()