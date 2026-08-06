"""
创建 PyTorch DataLoader

功能:
1. 读取 EEG numpy 数据
2. 转换 Tensor
3. 创建 Dataset
4. 创建 DataLoader
"""

import numpy as np

import torch

from torch.utils.data import TensorDataset, DataLoader



def main():

    print("="*70)

    print("Create EEG DataLoader")

    print("="*70)



    # =====================
    # 读取数据
    # =====================

    X = np.load(
        "data/X_eeg.npy"
    )


    y = np.load(
        "data/y_label.npy"
    )


    print(
        "Original X:",
        X.shape
    )

    print(
        "Original y:",
        y.shape
    )



    # =====================
    # numpy -> tensor
    # =====================


    X = torch.tensor(
        X,
        dtype=torch.float32
    )


    y = torch.tensor(
        y,
        dtype=torch.long
    )



    print(
        "\nTensor X:",
        X.shape
    )



    # =====================
    # 增加CNN维度
    # =====================

    X = X.unsqueeze(1)



    print(
        "CNN input X:",
        X.shape
    )



    # =====================
    # 创建Dataset
    # =====================


    dataset = TensorDataset(

        X,

        y

    )


    print(
        "\nDataset size:",
        len(dataset)
    )



    # =====================
    # DataLoader
    # =====================


    loader = DataLoader(

        dataset,

        batch_size=64,

        shuffle=True

    )



    # 取一个batch测试


    batch_X,batch_y = next(
        iter(loader)
    )


    print(
        "\nBatch EEG shape:"
    )

    print(
        batch_X.shape
    )


    print(
        "Batch label shape:"
    )

    print(
        batch_y.shape
    )



    print("="*70)

    print("完成")

    print("="*70)



if __name__=="__main__":

    main()