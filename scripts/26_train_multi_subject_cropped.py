"""
9被试 ShallowFBCSPNet + EMS + Cropped Training
"""


from pathlib import Path
import numpy as np
import torch

from skorch.callbacks import (
    EarlyStopping,
    LRScheduler
)

from skorch.helper import predefined_split


from braindecode import EEGClassifier


from braindecode.datasets import MOABBDataset


from braindecode.preprocessing import (
    Preprocessor,
    preprocess,
    exponential_moving_standardize,
    create_windows_from_events
)


from braindecode.models import ShallowFBCSPNet


from braindecode.training import CroppedLoss


from braindecode.util import set_random_seeds



def main():


    print("="*70)

    print(
        "Multi Subject Cropped Training"
    )

    print("="*70)



    seed=20260805

    cuda=torch.cuda.is_available()


    set_random_seeds(

        seed=seed,

        cuda=cuda

    )


    device="cuda" if cuda else "cpu"



    print(
        "Device:",
        device
    )



    # =====================
    # 1. 加载9个被试
    # =====================


    dataset=MOABBDataset(

        dataset_name="BNCI2014_001",

        subject_ids=list(
            range(1,10)
        )

    )


    print()

    print(
        "Dataset:"
    )

    print(dataset)



    # =====================
    # 2. 预处理
    # =====================

    preprocessors = [
        # 只保留22个EEG通道，删除EOG和刺激通道
        Preprocessor(
            "pick_types",
            eeg=True,
            meg=False,
            stim=False
        ),

        # MNE中的EEG单位为V，转换为μV
        Preprocessor(
            lambda data: np.multiply(data, 1e6)
        ),

        # 运动想象常用频段
        Preprocessor(
            "filter",
            l_freq=4.0,
            h_freq=38.0
        ),

        # 指数移动标准化
        Preprocessor(
            exponential_moving_standardize,
            factor_new=1e-3,
            init_block_size=1000
        )
    ]



    print()

    print(
        "Preprocessing..."
    )


    preprocess(

        dataset,

        preprocessors,

        n_jobs=1

    )


    print(
        "Preprocessing finished"
    )



    # =====================
    # 3. 创建Windows
    # =====================


    model=ShallowFBCSPNet(

        n_chans=22,

        n_outputs=4,

        n_times=1000,

        final_conv_length=30

    )


    model.to_dense_prediction_model()



    n_preds_per_input = (

        model.get_output_shape()[2]

    )


    print()

    print(
        "Predictions per input:",
        n_preds_per_input
    )



    windows_dataset=create_windows_from_events(

        dataset,

        trial_start_offset_samples=-125,

        trial_stop_offset_samples=0,

        window_size_samples=1000,

        window_stride_samples=n_preds_per_input,

        drop_last_window=False,

        preload=True

    )


    print()

    print(
        "Windows:"
    )

    print(
        windows_dataset
    )



    # =====================
    # 4. 数据划分
    # =====================


    split=windows_dataset.split(

        by="session"

    )


    print()

    print(
        split.keys()
    )



    train_set=split["0train"]


    valid_set=split["1test"]



    print(
        "Train:",
        len(train_set)
    )


    print(
        "Valid:",
        len(valid_set)
    )



    # =====================
    # 5. EEGClassifier
    # =====================

    clf = EEGClassifier(

        model,

        cropped=True,

        criterion=CroppedLoss,

        criterion__loss_function=torch.nn.functional.cross_entropy,

        classes=list(range(4)),

        train_split=predefined_split(

            valid_set

        ),


        optimizer=torch.optim.AdamW,


        optimizer__lr=0.000625,


        batch_size=64,


        iterator_train__shuffle=True,


        callbacks=[

            "accuracy",

            (

                "lr_scheduler",

                LRScheduler(

                    "CosineAnnealingLR",

                    T_max=20

                )

            ),

            (
                "early_stop",
                EarlyStopping(
                    monitor="valid_accuracy",
                    lower_is_better=False,
                    patience=5,
                    load_best=True
                )
            )

        ],


        device=device

    )

    criterion__loss_function = torch.nn.functional.cross_entropy

    print()

    print(
        "Start training..."
    )



    clf.fit(

        train_set,

        y=None,

        epochs=20

    )

    history = clf.history

    best_epoch = max(
        history,
        key=lambda row: row["valid_accuracy"]
    )

    best_accuracy = best_epoch["valid_accuracy"]
    best_epoch_number = best_epoch["epoch"]

    print()
    print("Best validation epoch:")
    print(best_epoch_number)

    print("Best validation accuracy:")
    print(f"{best_accuracy:.4f}")



    Path("models").mkdir(
        exist_ok=True
    )


    clf.save_params(

        f_params="models/multi_subject_cropped_params.pt",

        f_history="results/metrics/multi_subject_history.json"

    )


    print()

    print(
        "Saved"
    )


    print("="*70)



if __name__=="__main__":

    main()