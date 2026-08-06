"""BNCI2014_001 CSP + SVM运动想象分类"""

import numpy as np

from braindecode.datasets import MOABBDataset

import mne

from mne.decoding import CSP

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():

    print("=" * 70)
    print("CSP + SVM 运动想象分类")
    print("=" * 70)


    # ==========================
    # 1. 加载数据
    # ==========================

    dataset = MOABBDataset(
        dataset_name="BNCI2014_001",
        subject_ids=[1],
    )


    raw = dataset.datasets[0].raw


    events, event_id = mne.events_from_annotations(raw)


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
    epochs.pick("eeg")


    # 滤波
    epochs.filter(
        8,
        30
    )


    X = epochs.get_data()

    y = epochs.events[:, -1]


    print("原始数据:")
    print(X.shape)



    # ==========================
    # 2. 只做左右手分类
    # ==========================

    labels = epochs.events[:, -1]


    left_id = event_id["left_hand"]
    right_id = event_id["right_hand"]


    mask = (
        (labels == left_id)
        |
        (labels == right_id)
    )


    X = X[mask]
    y = labels[mask]


    print("\n左右手数据:")
    print(X.shape)



    # 标签转换

    encoder = LabelEncoder()

    y = encoder.fit_transform(y)


    print(
        "类别:",
        encoder.classes_
    )


    # ==========================
    # 3. 划分训练测试
    # ==========================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )


    # ==========================
    # 4. CSP + SVM
    # ==========================

    clf = Pipeline(
        [
            (
                "CSP",
                CSP(
                    n_components=4,
                    log=True,
                    norm_trace=False
                )
            ),

            (
                "SVM",
                SVC(
                    kernel="linear"
                )
            )
        ]
    )


    clf.fit(
        X_train,
        y_train
    )


    y_pred = clf.predict(
        X_test
    )


    acc = accuracy_score(
        y_test,
        y_pred
    )


    print("\n测试准确率:")
    print(
        f"{acc:.4f}"
    )


    print("=" * 70)
    print("完成")
    print("=" * 70)



if __name__ == "__main__":
    main()