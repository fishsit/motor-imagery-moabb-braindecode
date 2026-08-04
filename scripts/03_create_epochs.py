"""创建 BNCI2014_001 运动想象 EEG Epoch。"""

import mne
from braindecode.datasets import MOABBDataset


def main():

    print("=" * 70)
    print("创建 EEG Epoch")
    print("=" * 70)

    dataset = MOABBDataset(
        dataset_name="BNCI2014_001",
        subject_ids=[1],
    )

    # 取第一段记录
    raw = dataset.datasets[0].raw

    print("\n原始 EEG:")
    print(raw)

    print("\n事件信息:")

    events, event_id = mne.events_from_annotations(raw)

    print(event_id)

    print("\n事件数量:")
    print(len(events))


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


    print("\nEpoch 创建完成")

    print(epochs)

    print("\nEpoch数量:")
    print(len(epochs))


    print("\nEpoch数据shape:")
    print(epochs.get_data().shape)


    print("=" * 70)
    print("完成")
    print("=" * 70)



if __name__ == "__main__":
    main()