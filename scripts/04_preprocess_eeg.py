"""BNCI2014_001 EEG预处理"""

import numpy as np
import mne

from braindecode.datasets import MOABBDataset


def main():

    print("=" * 70)
    print("EEG预处理")
    print("=" * 70)


    # 读取数据
    dataset = MOABBDataset(
        dataset_name="BNCI2014_001",
        subject_ids=[1],
    )


    raw = dataset.datasets[0].raw


    # 提取事件
    events, event_id = mne.events_from_annotations(raw)


    # 创建Epoch
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=0,
        tmax=4,
        baseline=None,
        preload=True,
    )


    print("\n原始shape:")
    print(epochs.get_data().shape)



    # =====================
    # 1. 删除非EEG通道
    # =====================

    epochs.pick(
        picks="eeg"
    )


    print("\n删除EOG和Stim后:")
    print(epochs.get_data().shape)



    # =====================
    # 2. 8-30Hz滤波
    # =====================

    epochs.filter(
        l_freq=8,
        h_freq=30,
    )


    print("\n滤波完成")



    # =====================
    # 3. 标准化
    # =====================

    data = epochs.get_data()


    mean = np.mean(
        data,
        axis=(0,2),
        keepdims=True
    )

    std = np.std(
        data,
        axis=(0,2),
        keepdims=True
    )


    data = (data-mean)/std


    print("\n标准化完成")

    print("\n最终数据:")
    print(data.shape)


    print("=" * 70)
    print("预处理完成")
    print("=" * 70)



if __name__ == "__main__":
    main()