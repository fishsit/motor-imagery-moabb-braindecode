"""
创建 Braindecode WindowsDataset

流程:

MOABB
 ↓
Epochs
 ↓
EMS
 ↓
WindowsDataset
"""


import numpy as np

import mne


from braindecode.datasets import (
    MOABBDataset,
    create_from_mne_epochs
)


from braindecode.preprocessing import (
    exponential_moving_standardize
)



def main():

    print("="*70)

    print("Create WindowsDataset")

    print("="*70)



    # =====================
    # 加载数据
    # =====================


    dataset = MOABBDataset(

        dataset_name="BNCI2014_001",

        subject_ids=[1]

    )


    raw = dataset.datasets[0].raw



    # =====================
    # 删除非EEG通道
    # =====================


    raw.pick(
        "eeg"
    )


    print()

    print(
        "EEG channels:"
    )

    print(
        len(raw.ch_names)
    )



    # =====================
    # 事件
    # =====================


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



    print()

    print(
        "Epoch:"
    )

    print(
        epochs
    )



    # =====================
    # EMS
    # =====================


    print()

    print(
        "Applying EMS..."
    )


    X=epochs.get_data()



    for i in range(len(X)):


        X[i]=exponential_moving_standardize(

            X[i],

            factor_new=0.001,

            init_block_size=1000

        )



    epochs._data=X



    print(
        "EMS finished"
    )



    # =====================
    # 创建WindowsDataset
    # =====================

    windows_dataset = create_from_mne_epochs(

        [
            epochs
        ],

        window_size_samples=250,

        window_stride_samples=125,

        drop_last_window=True

    )



    print()

    print(
        "WindowsDataset:"
    )


    print(
        windows_dataset
    )


    print()

    print(
        "Number of windows:"
    )


    print(
        len(windows_dataset)
    )



    print("="*70)

    print("Done")



if __name__=="__main__":

    main()