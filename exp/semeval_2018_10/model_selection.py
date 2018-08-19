def select_best_concat_mlp(step_to_val_best_scores_tuple):
    best_step = ""
    best_avg_f1 = 0
    for step, scores_tuple in step_to_val_best_scores_tuple.items():
        avg_f1 = scores_tuple[0]
        if avg_f1 > best_avg_f1:
            best_step = step
            best_avg_f1 = avg_f1
    return int(best_step), step_to_val_best_scores_tuple[best_step]


def select_best_concat_svm(step_to_val_best_scores_tuple):
    best_step = ""
    best_avg_f1 = 0
    for step, scores_tuple in step_to_val_best_scores_tuple.items():
        avg_f1 = scores_tuple[0]
        if avg_f1 > best_avg_f1:
            best_step = step
            best_avg_f1 = avg_f1
    return int(best_step), step_to_val_best_scores_tuple[best_step]