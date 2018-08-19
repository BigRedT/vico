import numpy as np
import sklearn.metrics as metrics


def compute_f1(
        pred_prob,
        gt_label,
        thresh=np.arange(0,1.2,0.1)):
    num_thresh = thresh.shape[0]
    scores_tuples = []
    gt_label = gt_label > 0.5
    for i in range(num_thresh):
        t = thresh[i]
        pred_label = pred_prob >= t
        pos_f1 = metrics.f1_score(gt_label,pred_label,pos_label=True)
        neg_f1 = metrics.f1_score(gt_label,pred_label,pos_label=False)
        if pos_f1 and neg_f1:
            avg_f1 = 0.5 * (pos_f1 + neg_f1)
        else:
            avg_f1 = 0
        acc = metrics.accuracy_score(gt_label,pred_label)
        scores_tuples.append((avg_f1,pos_f1,neg_f1,acc,t))
    best_scores_tuple = sorted(scores_tuples,key=lambda x: x[0],reverse=True)[0]
    return scores_tuples, best_scores_tuple
