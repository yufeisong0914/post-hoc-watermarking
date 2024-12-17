from sklearn.metrics import roc_curve, auc
import numpy as np


def cal_tpr_at_fpr(scores: list[float], labels: list[int], target_fpr: float = 0.01):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    print(f"AUC: {roc_auc}")
    fpr_np = np.array(fpr)
    tpr_np = np.array(tpr)
    if target_fpr < fpr_np[0]:
        print(f"TPR at {target_fpr * 100}% FPR: {tpr_np[0] * 100:5.1f}% (target FPR too low)")
        return
    if target_fpr > fpr_np[-1]:
        print(f"TPR at {target_fpr * 100}% FPR: {tpr_np[-1] * 100:5.1f}% (target FPR too high)")
        return
    idx = np.searchsorted(fpr_np, target_fpr, side='right')
    if fpr_np[idx - 1] == target_fpr:
        tpr_value = tpr_np[idx - 1]
    else:
        tpr_value = tpr_np[idx - 1] + (target_fpr - fpr_np[idx - 1]) * (tpr_np[idx] - tpr_np[idx - 1]) / (
                fpr_np[idx] - fpr_np[idx - 1])
    print(f"TPR at {target_fpr * 100}% FPR: {tpr_value * 100:5.1f}%")
