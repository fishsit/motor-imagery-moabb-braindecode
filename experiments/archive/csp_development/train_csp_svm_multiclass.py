"""
BNCI2014_001 四分类运动想象
CSP + SVM
"""

import mne
import numpy as np

from braindecode.datasets import MOABBDataset

from mne.decoding import CSP

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


def main():

    print("=" * 70)
    print("CSP + SVM 四分类运动想象")
    print("=" * 70)


    # =========================
    # 1. 加载数据
    # =========================

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


    # 删除EOG/STI
    epochs.pick("eeg")


    # 滤波
    epochs.filter(
        8,
        30
    )


    X = epochs.get_data()

    y = epochs.events[:, -1]


    print("\nEEG数据:")
    print(X.shape)


    print("\n类别:")
    print(event_id)



    # =========================
    # 2. 标签编码
    # =========================

    encoder = LabelEncoder()

    y = encoder.fit_transform(y)


    print("\n编码后类别:")
    print(
        encoder.classes_
    )


    # =========================
    # 3. 划分训练测试
    # =========================

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.25,

        random_state=42,

        stratify=y

    )


    print("\n训练样本:")
    print(len(X_train))

    print("测试样本:")
    print(len(X_test))


    # =========================
    # 4. CSP + SVM
    # =========================

    clf = Pipeline(

        [

            (
                "CSP",

                CSP(

                    n_components=8,

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


    # =========================
    # 5.训练
    # =========================

    print("\n开始训练...")


    clf.fit(

        X_train,

        y_train

    )


    # =========================
    # 6.测试
    # =========================

    y_pred = clf.predict(

        X_test

    )


    acc = accuracy_score(

        y_test,

        y_pred

    )


    print("\n======================")

    print(
        "测试准确率:"
    )

    print(
        f"{acc:.4f}"
    )


    print("\n混淆矩阵:")

    print(

        confusion_matrix(

            y_test,

            y_pred

        )

    )


    print("\n分类报告:")

    print(

        classification_report(

            y_test,

            y_pred,

            target_names=[
                "left_hand",
                "right_hand",
                "feet",
                "tongue"
            ]

        )

    )


    print("=" * 70)

    print("完成")

    print("=" * 70)



if __name__ == "__main__":

    main()