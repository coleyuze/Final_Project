from sklearn.metrics import classification_report
from sklearn.metrics import f1_score


def mean_f1(labels_df, predict_df):
    group_df = labels_df.groupby("Claim")
    all_f1 = []
    for claim, index in group_df.groups.items():
        label_claim = labels_df.iloc[index]
        predict_claim = predict_df.iloc[index]      
        report = classification_report(label_claim['Stance'],predict_claim['Stance'],output_dict=True)
        all_f1.append(report['macro avg']['f1-score'])
    return sum(all_f1)/len(all_f1)