"""
绘制官方Cropped Training训练曲线
"""


import json

import matplotlib.pyplot as plt



def main():

    print("="*70)

    print("Plot Cropped Training Curve")

    print("="*70)



    with open(
        "results/metrics/shallow_official_cropped_history.json",
        "r"
    ) as f:

        history=json.load(f)



    epochs=[]

    train_acc=[]

    valid_acc=[]

    train_loss=[]

    valid_loss=[]



    for item in history:

        epochs.append(
            item["epoch"]
        )

        train_acc.append(
            item["train_accuracy"]
        )

        valid_acc.append(
            item["valid_accuracy"]
        )

        train_loss.append(
            item["train_loss"]
        )

        valid_loss.append(
            item["valid_loss"]
        )



    # accuracy

    plt.figure(
        figsize=(7,5)
    )


    plt.plot(
        epochs,
        train_acc,
        marker="o",
        label="Train Accuracy"
    )


    plt.plot(
        epochs,
        valid_acc,
        marker="o",
        label="Valid Accuracy"
    )


    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy"
    )


    plt.title(
        "ShallowFBCSPNet Cropped Training Accuracy"
    )


    plt.legend()


    plt.grid()


    plt.savefig(
        "results/figures/shallow_cropped_accuracy.png",
        dpi=300
    )


    plt.close()



    # loss

    plt.figure(
        figsize=(7,5)
    )


    plt.plot(
        epochs,
        train_loss,
        marker="o",
        label="Train Loss"
    )


    plt.plot(
        epochs,
        valid_loss,
        marker="o",
        label="Valid Loss"
    )


    plt.xlabel(
        "Epoch"
    )


    plt.ylabel(
        "Loss"
    )


    plt.title(
        "ShallowFBCSPNet Cropped Training Loss"
    )


    plt.legend()


    plt.grid()


    plt.savefig(
        "results/figures/shallow_cropped_loss.png",
        dpi=300
    )


    print()

    print("保存完成:")

    print(
        "results/figures/"
    )


if __name__=="__main__":

    main()