import pandas as pd


data = {

"Method":[

"CSP+SVM",

"EEGNet",

"ShallowFBCSPNet",

"ShallowFBCSPNet+EMS",

"ShallowFBCSPNet+EMS+Cropped"

],


"Accuracy":[

0.3659,

0.4175,

0.2951,

0.5323,

0.6493

]

}



df=pd.DataFrame(data)


df.to_csv(

"results/metrics/experiment_summary.csv",

index=False

)


print(df)

print()

print(
"Saved:"
)

print(
"results/metrics/experiment_summary.csv"
)