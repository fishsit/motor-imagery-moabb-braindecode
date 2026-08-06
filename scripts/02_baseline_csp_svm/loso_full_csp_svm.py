"""
BNCI2014_001
完整Session+Run
LOSO CSP + SVM
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

    print(
        f"加载 Subject {subject}"
    )


    dataset = MOABBDataset(

        dataset_name="BNCI2014_001",

        subject_ids=[subject]

    )


    X_all=[]

    y_all=[]



    # 遍历所有run

    for record in dataset.datasets:


        raw = record.raw



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


        epochs.pick("eeg")


        epochs.filter(

            8,

            30

        )


        X = epochs.get_data()


        y = epochs.events[:,-1]



        X_all.append(X)

        y_all.append(y)



    X=np.concatenate(

        X_all,

        axis=0

    )


    y=np.concatenate(

        y_all,

        axis=0

    )


    print(

        "Subject shape:",

        X.shape

    )


    return X,y




def main():


    print("="*70)

    print(
        "完整BNCI2014 LOSO CSP+SVM"
    )

    print("="*70)



    subjects=range(1,10)



    data={}



    for s in subjects:

        data[s]=load_subject(s)



    results=[]



    for test_subject in subjects:


        print("\n----------------")

        print(
            "测试:",
            test_subject
        )



        X_test,y_test=data[test_subject]



        train_X=[]

        train_y=[]



        for s in subjects:


            if s != test_subject:


                X,y=data[s]


                train_X.append(X)

                train_y.append(y)



        X_train=np.concatenate(

            train_X,

            axis=0

        )


        y_train=np.concatenate(

            train_y,

            axis=0

        )



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

            "accuracy:",

            acc

        )


        results.append(acc)



    print("\n===================")

    print(
        "平均准确率:"
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


    print("===================")



if __name__=="__main__":

    main()