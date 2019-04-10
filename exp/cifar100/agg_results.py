import os
import numpy as np

import utils.io as io


def print_header(runs):
    print_str = 'Embeddings'
    for run in runs:
        print_str += f' & Run {run}'
    print_str += ' & Mean & Std'
    print_str += ' \\\\'
    print(print_str)


def print_row(embed_name,acc_list):
    print_str = embed_name
    for acc in acc_list:
        print_str += f' & {acc}'
    mean = round(np.mean(acc_list),2)
    std = round(np.std(acc_list),2)
    print_str += f' & {mean} & {std} \\\\'
    print(print_str)


def main(exp_const):
    io.mkdir_if_not_exists(exp_const.exp_dir)
    
    results = {}
    for embed_name, exp_prefix in exp_const.prefix.items():
        results[embed_name] = []
        for run in exp_const.runs:
            results_json = f'{exp_const.runs_prefix}{run}/' + \
                f'{exp_prefix}{exp_const.held_out_classes}/' + \
                'selected_model_results.json'

            results[embed_name].append(
                io.load_json_object(results_json)['Unseen Acc'])
    
    print_header(exp_const.runs)
    for embed_name, exp_prefix in exp_const.prefix.items():
        print_row(embed_name,results[embed_name])
