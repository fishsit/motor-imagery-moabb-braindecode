"""
准备 Braindecode 深度学习数据集

功能:
1. 读取 BNCI2014_001
2. 合并9个被试
3. Epoch切片
4. EEG预处理
5. 保存numpy数据
"""

import os

import numpy as np
import mne

from braindecode.datasets import MOABBDataset



def load_subject(subject):

    print(
        f"Loading subject {subject}"
    )


    dataset = MOABBDataset(
        dataset_name="BNCI2014_001",
        subject_ids=[subject],
    )


    X_all = []
    y_all = []


    # 遍历所有run

    for record in dataset.datasets:


        raw = record.raw


        events, event_id = mne.events_from_annotations(
            raw
        )


        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=0,
            tmax=4,
            baseline=None,
            preload=True,
        )


        # 只保留EEG

        epochs.pick(
            "eeg"
        )


        # 运动想象频段

        epochs.filter(
            8,
            30
        )


        X = epochs.get_data()


        y = epochs.events[:, -1]


        X_all.append(X)

        y_all.append(y)



    X = np.concatenate(
        X_all,
        axis=0
    )


    y = np.concatenate(
        y_all,
        axis=0
    )


    return X, y




def main():


    print("="*70)

    print(
        "Prepare Deep Learning Dataset"
    )

    print("="*70)



    X_list=[]

    y_list=[]



    for subject in range(1,10):

        X,y = load_subject(
            subject
        )


        print(
            "subject data:",
            X.shape
        )


        X_list.append(X)

        y_list.append(y)



    # 合并所有被试


    X=np.concatenate(
        X_list,
        axis=0
    )


    y=np.concatenate(
        y_list,
        axis=0
    )


    print("\nFinal Dataset")

    print(
        "X:",
        X.shape
    )


    print(
        "y:",
        y.shape
    )


    # 创建保存目录

    os.makedirs(
        "data",
        exist_ok=True
    )


    np.save(
        "data/X_eeg.npy",
        X
    )


    np.save(
        "data/y_label.npy",
        y
    )


    print(
        "\n数据保存完成"
    )


    print(
        "data/X_eeg.npy"
    )

    print(
        "data/y_label.npy"
    )



if __name__=="__main__":

    main()