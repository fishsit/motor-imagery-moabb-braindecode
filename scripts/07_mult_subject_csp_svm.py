"""
BNCI2014_001 多被试 CSP + SVM 四分类
"""

import numpy as np
import mne

from braindecode.datasets import MOABBDataset

from mne.decoding import CSP

from sklearn.pipeline import Pipeline

from sklearn.svm import SVC

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix
)



def main():

    print("="*70)
    print("多被试 CSP + SVM 四分类")
    print("="*70)


    # 保存所有数据

    all_X = []
    all_y = []


    # ======================
    # 读取9个被试
    # ======================

    for subject in range(1,10):

        print(
            f"\n读取 Subject {subject}"
        )


        dataset = MOABBDataset(

            dataset_name="BNCI2014_001",

            subject_ids=[subject]

        )


        # 取第一个run
        raw = dataset.datasets[0].raw


        events,event_id = mne.events_from_annotations(
            raw
        )


        epochs = mne.Epochs(

            raw,

            events,

            event_id=event_id,

            tmin=0,

            tmax=4,

            baseline=None,

            preload=True

        )


        # 保留EEG

        epochs.pick("eeg")


        # 滤波

        epochs.filter(
            8,
            30
        )


        X = epochs.get_data()


        y = epochs.events[:,-1]


        print(
            "shape:",
            X.shape
        )


        all_X.append(X)

        all_y.append(y)



    # ======================
    # 合并数据
    # ======================


    X = np.concatenate(
        all_X,
        axis=0
    )


    y = np.concatenate(
        all_y,
        axis=0
    )


    print("\n======================")

    print("总数据:")

    print(X.shape)


    print("标签数量:")

    print(len(y))



    # 标签编码

    encoder = LabelEncoder()

    y = encoder.fit_transform(y)


    print(
        "类别:",
        encoder.classes_
    )



    # ======================
    # 划分训练测试
    # ======================


    X_train,X_test,y_train,y_test = train_test_split(

        X,

        y,

        test_size=0.25,

        random_state=42,

        stratify=y

    )


    print(
        "\n训练:",
        len(X_train)
    )

    print(
        "测试:",
        len(X_test)
    )



    # ======================
    # CSP + SVM
    # ======================


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



    print(
        "\n开始训练..."
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


    print(
        "\n准确率:"
    )

    print(
        f"{acc:.4f}"
    )


    print(
        "\n混淆矩阵:"
    )

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )


    print("="*70)

    print("完成")

    print("="*70)



if __name__=="__main__":

    main()