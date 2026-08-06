"""
准备 Cropped Training 数据

流程:

MOABB
 ↓
Raw
 ↓
Epochs
 ↓
WindowsDataset
"""


import numpy as np

import mne


from braindecode.datasets import MOABBDataset



def main():

    print("="*70)

    print("Prepare Cropped Dataset")

    print("="*70)



    # =====================
    # 读取MOABB
    # =====================

    dataset = MOABBDataset(

        dataset_name="BNCI2014_001",

        subject_ids=[1]

    )


    print()

    print(
        "Records:",
        len(dataset.datasets)
    )



    # 取第一段record

    raw = dataset.datasets[0].raw


    print()

    print(raw)



    # =====================
    # 事件
    # =====================


    events, event_id = mne.events_from_annotations(
        raw
    )


    print()

    print(
        "Event id:"
    )

    print(event_id)



    # =====================
    # Epoch
    # =====================


    epochs = mne.Epochs(

        raw,

        events,

        event_id,

        tmin=0,

        tmax=4,

        baseline=None,

        preload=True

    )


    print()

    print(
        epochs
    )


    X = epochs.get_data()


    y = epochs.events[:, -1]


    print()

    print(
        "Epoch data:"
    )

    print(
        X.shape
    )


    print()

    print(
        "Labels:"
    )

    print(
        np.unique(y)
    )



    print("="*70)

    print("Done")



if __name__=="__main__":

    main()