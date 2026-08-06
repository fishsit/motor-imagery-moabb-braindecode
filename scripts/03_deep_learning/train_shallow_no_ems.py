"""
测试 ExponentialMovingStandardize

用于 ShallowFBCSPNet 前的数据预处理
"""


import numpy as np


from braindecode.preprocessing import (
    exponential_moving_standardize
)



def main():

    print("="*70)

    print("Test ExponentialMovingStandardize")

    print("="*70)


    # 加载 EEG

    X = np.load(
        "data/X_eeg.npy"
    )


    print(
        "Before:",
        X.shape
    )


    print(
        "Before mean:",
        X.mean()
    )


    print(
        "Before std:",
        X.std()
    )


    # =====================
    # EMS标准化
    # =====================


    X_new = np.zeros_like(
        X
    )


    for i in range(
        len(X)
    ):


        X_new[i] = exponential_moving_standardize(
            X[i],
            factor_new=0.001,
            init_block_size=1000
        )



    print()

    print(
        "After:"
    )


    print(
        "After mean:",
        X_new.mean()
    )


    print(
        "After std:",
        X_new.std()
    )


    print("="*70)

    print("完成")



if __name__=="__main__":

    main()