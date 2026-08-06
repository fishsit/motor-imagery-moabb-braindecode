"""下载并检查 BNCI2014_001 运动想象脑电数据。"""
from collections import Counter
from braindecode.datasets import MOABBDataset
def main() -> None:
    """加载被试 1，并输出数据的基本结构。"""
dataset_name = "BNCI2014_001"
subject_id = 1

print("=" * 70)
print("BNCI2014_001 数据加载")
print("=" * 70)
print(f"数据集：{dataset_name}")
print(f"被试编号：{subject_id}")
print("第一次运行会自动下载数据，请保持网络连接。")

dataset = MOABBDataset(
    dataset_name=dataset_name,
    subject_ids=[subject_id],
)

print("\n数据加载成功。")
print(f"内部记录数量：{len(dataset.datasets)}")

print("\n数据描述：")
print(dataset.description.to_string(index=False))

first_recording = dataset.datasets[0]
raw = first_recording.raw

print("\n第一段记录的描述：")
print(first_recording.description)

print("\n第一段记录的基本信息：")
print(f"采样率：{raw.info['sfreq']} Hz")
print(f"通道数：{len(raw.ch_names)}")
print(f"采样点数：{raw.n_times}")

duration = raw.n_times / raw.info["sfreq"]
print(f"记录时长：{duration:.2f} 秒")

print("\n通道类型统计：")
channel_type_counts = Counter(raw.get_channel_types())
for channel_type, count in channel_type_counts.items():
    print(f"{channel_type}: {count}")

print("\n通道名称：")
print(raw.ch_names)

print("\n事件标注统计：")
annotation_counts = Counter(raw.annotations.description)
for event_name, count in annotation_counts.items():
    print(f"{event_name}: {count}")

print("=" * 70)
print("数据检查完成")
print("=" * 70)