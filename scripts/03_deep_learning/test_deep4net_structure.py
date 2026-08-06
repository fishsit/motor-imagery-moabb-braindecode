"""
检查 Deep4Net 的输入、输出和参数数量。

本脚本暂时不读取真实 EEG，也不进行训练。
目标是先确认：

1. Deep4Net 可以接收 (batch, channels, time) 格式；
2. Dense Prediction 转换可以正常完成；
3. 模型可以为一个输入窗口输出多个时间位置的预测。
"""

from __future__ import annotations

import torch
from braindecode.models import Deep4Net


# =========================
# 1. 基本参数
# =========================

N_CHANS = 22
N_CLASSES = 4

# 250 Hz × 4 秒
N_TIMES = 1000

# 使用两个虚拟 trial 组成一个 batch
BATCH_SIZE = 2


# =========================
# 2. 创建 Deep4Net
# =========================

model = Deep4Net(
    n_chans=N_CHANS,
    n_outputs=N_CLASSES,
    n_times=N_TIMES,

    # Cropped Training 需要最终卷积核不能占满整个时间维度，
    # 否则每个窗口只会产生一个预测。
    final_conv_length=2,
)


# =========================
# 3. 转换为密集预测模型
# =========================

model.to_dense_prediction_model()
model.eval()


# =========================
# 4. 构造虚拟 EEG
# =========================

dummy_eeg = torch.randn(
    BATCH_SIZE,
    N_CHANS,
    N_TIMES,
    dtype=torch.float32,
)


# =========================
# 5. 前向传播
# =========================

with torch.no_grad():
    output = model(dummy_eeg)


# =========================
# 6. 输出检查结果
# =========================

trainable_params = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("=" * 60)
print("Deep4Net structure check")
print("=" * 60)

print(f"Input shape       : {tuple(dummy_eeg.shape)}")
print(f"Output shape      : {tuple(output.shape)}")
print(f"Trainable params  : {trainable_params:,}")
print(f"Output dtype      : {output.dtype}")

if output.ndim == 3:
    print(f"Batch size        : {output.shape[0]}")
    print(f"Number of classes : {output.shape[1]}")
    print(f"Dense predictions : {output.shape[2]}")

    assert output.shape[0] == BATCH_SIZE
    assert output.shape[1] == N_CLASSES
    assert output.shape[2] > 1

    print("\nResult: PASS")
    print("Deep4Net can produce multiple temporal predictions.")
else:
    raise RuntimeError(
        "Deep4Net output should normally have shape "
        "(batch, classes, predictions) after dense conversion, "
        f"but received {tuple(output.shape)}."
    )