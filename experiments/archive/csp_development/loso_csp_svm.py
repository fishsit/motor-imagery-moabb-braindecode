"""
BNCI2014_001
Leave-One-Subject-Out
CSP + SVM
"""

import numpy as np
import mne

from braindecode.datasets import MOABBDataset

from mne.decoding import CSP

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score



def load_subject(subject):

    dataset = MOABBDataset(
        dataset_name="BNCI2014_001",
        subject_ids=[subject],
    )


    raw = dataset.datasets[0].raw


    events,event_id = mne.events_from_annotations(raw)


    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=0,
        tmax=4,
        baseline=None,
        preload=True,
    )


    epochs.pick("eeg")


    epochs.filter(
        8,
        30
    )


    X = epochs.get_data()

    y = epochs.events[:,-1]


    return X,y



def main():

    print("="*70)

    print(
        "LOSO跨被试 CSP + SVM"
    )

    print("="*70)



    subjects = range(1,10)


    all_data = {}



    # ======================
    # 加载所有被试
    # ======================

    for s in subjects:

        print(
            f"加载 Subject {s}"
        )

        all_data[s] = load_subject(s)



    results=[]



    # ======================
    # LOSO循环
    # ======================

    for test_subject in subjects:


        print("\n===================")

        print(
            f"测试 Subject {test_subject}"
        )


        X_test,y_test = all_data[test_subject]



        X_train_list=[]

        y_train_list=[]



        for s in subjects:

            if s != test_subject:

                X,y = all_data[s]


                X_train_list.append(X)

                y_train_list.append(y)



        X_train=np.concatenate(
            X_train_list,
            axis=0
        )


        y_train=np.concatenate(
            y_train_list,
            axis=0
        )



        # 标签编码

        encoder=LabelEncoder()


        y_train=encoder.fit_transform(
            y_train
        )


        y_test=encoder.transform(
            y_test
        )



        clf=Pipeline(

            [

                (
                    "CSP",

                    CSP(
                        n_components=8,
                        log=True
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


        pred=clf.predict(
            X_test
        )


        acc=accuracy_score(
            y_test,
            pred
        )


        print(
            f"Subject {test_subject} accuracy:"
            f"{acc:.4f}"
        )


        results.append(acc)



    print("\n===================")

    print(
        "LOSO平均准确率:"
    )

    print(
        np.mean(results)
    )


    print(
        "标准差:"
    )

    print(
        np.std(results)
    )


    print("="*70)



if __name__=="__main__":
    main()