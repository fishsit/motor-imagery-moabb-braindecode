# Motor Imagery BCI Model Comparison

## Multi-subject experiments

| Model | Protocol | Subjects | Accuracy | Accuracy Std | Macro-F1 |
|---|---|---:|---:|---:|---:|
| CSP + SVM | 9-subject LOSO | 1-9 | 38.89% | 13.29% | - |
| ShallowFBCSPNet | 9-subject LOSO | 1-9 | 46.70% | 15.98% | 42.96% |
| Deep4Net | Run-wise CV inside 0train, held-out 1test evaluation | 3,4,5,6,7,8,9 | 46.23% | 14.44% | 41.80% |

> CSP + SVM and ShallowFBCSPNet use a leave-one-subject-out protocol. Deep4Net uses within-subject run-wise cross-validation and a held-out testing session. The results summarize different experimental questions and should not be treated as a completely controlled ranking.

## Subject 1 development experiments

| Model | Accuracy | Macro-F1 | Status |
|---|---:|---:|---|
| ShallowFBCSPNet + EMS + Cropped | 64.93% | - | Development result |
| Deep4Net baseline | 45.49% | 37.80% | Development result |
| Deep4Net low regularization | 63.89% | 63.10% | Development result |
| EEGNet | - | - | Pending |

## Main observations

1. ShallowFBCSPNet improves the nine-subject LOSO mean accuracy from 38.89% for CSP + SVM to 46.70%.

2. Deep4Net obtains a mean held-out session accuracy of 46.23% and a mean Macro-F1 of 41.80% across Subjects 3–9.

3. Deep4Net results vary strongly by subject, with test accuracy ranging from 28.13% to 63.89%.

4. EEGNet is not included in the formal comparison because a protocol-matched final evaluation has not yet been completed.
