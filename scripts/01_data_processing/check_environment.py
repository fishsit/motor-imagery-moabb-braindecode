"""检查 MOABB + Braindecode 项目的基础运行环境。"""

import sys

import braindecode
import mne
import moabb
import numpy
import pandas
import sklearn
import torch


def main() -> None:
    """输出 Python 和主要依赖库的版本信息。"""

    print("=" * 60)
    print("MOABB + Braindecode 环境检查")
    print("=" * 60)

    print(f"Python       : {sys.version.split()[0]}")
    print(f"Python path  : {sys.executable}")
    print(f"PyTorch      : {torch.__version__}")
    print(f"MOABB        : {moabb.__version__}")
    print(f"Braindecode  : {braindecode.__version__}")
    print(f"MNE          : {mne.__version__}")
    print(f"NumPy        : {numpy.__version__}")
    print(f"Pandas       : {pandas.__version__}")
    print(f"Scikit-learn : {sklearn.__version__}")

    print("-" * 60)
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU name      : {torch.cuda.get_device_name(0)}")
        print(f"PyTorch CUDA  : {torch.version.cuda}")
    else:
        print("当前使用 CPU，暂时不影响项目复现。")

    print("=" * 60)
    print("环境检查通过。")
    print("=" * 60)


if __name__ == "__main__":
    main()