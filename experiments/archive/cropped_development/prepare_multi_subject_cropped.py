"""
准备多被试BNCI2014_001数据
"""

import mne
import numpy as np

from braindecode.datasets import MOABBDataset


def main():

    print("="*70)
    print("Prepare Multi Subject Dataset")
    print("="*70)


    all_epochs=[]


    for subject in range(1,10):

        print()

        print(
            f"Loading subject {subject}"
        )


        dataset = MOABBDataset(

            dataset_name="BNCI2014_001",

            subject_ids=[subject]

        )


        for ds in dataset.datasets:

            raw=ds.raw.copy()


            raw.pick(
                "eeg"
            )


            events,event_id=mne.events_from_annotations(
                raw
            )


            epochs=mne.Epochs(

                raw,

                events,

                event_id,

                tmin=0,

                tmax=4,

                baseline=None,

                preload=True

            )


            all_epochs.append(
                epochs
            )


    print()

    print(
        "Total recordings:",
        len(all_epochs)
    )


    total_trials=sum(
        len(ep)
        for ep in all_epochs
    )


    print(
        "Total trials:",
        total_trials
    )


    X=np.concatenate(

        [
            ep.get_data()
            for ep in all_epochs
        ],

        axis=0

    )


    y=np.concatenate(

        [
            ep.events[:,-1]
            for ep in all_epochs
        ]

    )


    print()

    print(
        "X shape:",
        X.shape
    )


    print(
        "y shape:",
        y.shape
    )


    print(
        "classes:",
        np.unique(y)
    )


    print("="*70)

    print("Done")



if __name__=="__main__":

    main()