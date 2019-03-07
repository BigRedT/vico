import os
import numpy as np
import plotly
import plotly.graph_objs as go

import utils.io as io


def plot_acc_vs_classes(results,metric_name,filename):
    embed_type_to_color = {
        'GloVe+ViCo(linear,100)': 'rgb(55, 128, 191)', 
        'GloVe+ViCo(linear,200)': 'rgb(55, 80, 191)',
        'GloVe+ViCo(select,200)': 'rgb(219, 64, 82)',
        'GloVe': 'rgb(44, 150, 44)',
        'ViCo(linear)': 'rgb(214, 39, 40)',
        'GloVe+random(100)': 'rgb(255, 200, 14)',
        'GloVe+random(200)': 'rgb(255, 127, 14)',
        'random(100)': 'rgb(200,200,200)',
        'chance': 'grey',
    }

    metric_name_to_ytitle = {
        'HM Acc': 'Harmonic Mean of Seen and Unseen Classes Accuracies)',
        'Seen Acc': 'Seen Classes Accuracy (%)',
        'Unseen Acc': 'Unseen Classes Accuracy (%)',
        'Step': 'Best model iterations',
    }

    num_held_out_classes = sorted(list(results['GloVe'].keys()))

    traces = []

    if metric_name != 'Step':
        y = []
        if metric_name == 'Unseen Acc':
            for num_held in num_held_out_classes:
                y.append(round(100/num_held,2))
        elif metric_name == 'Seen Acc':
            for num_held in num_held_out_classes:
                y.append(round(100/(100-num_held),2))
        elif metric_name == 'HM Acc':
            for num_held in num_held_out_classes:
                unseen_acc = 100/num_held
                seen_acc = 100/(100-num_held)
                hm_acc = 2*seen_acc*unseen_acc / (seen_acc + unseen_acc)
                y.append(round(hm_acc,2))
        else:
            assert(False), 'Not implemented'

        trace = go.Bar(
            x = [100-x_ for x_ in num_held_out_classes],
            y = y,
            text = y,
            textposition = 'auto',
            name = 'chance',
            marker = dict(color=embed_type_to_color['chance']),
            opacity=0.9,
        )
        traces.append(trace)

    for embed_type in results.keys():
        y = []
        err = []
        for num_held in num_held_out_classes:
            y.append(
                round(np.mean(results[embed_type][num_held][metric_name]),1))
            err.append(
                round(np.std(results[embed_type][num_held][metric_name]),1))

        trace = go.Bar(
            x = [100-x_ for x_ in num_held_out_classes],
            y = y,
            text = y,
            textposition = 'auto',
            name = embed_type,
            marker = dict(color=embed_type_to_color[embed_type]),
            opacity=0.9,
            #insidetextfont=dict(size=1),
            error_y=dict(
                type='data',
                array=err,
                visible=True,
                thickness=1.5,
                width=4,
                color='rgba(10,10,10,0.6)'
            )
        )
        traces.append(trace)

    xtitle = '#Trainable classes (= 100 - #Held out classes)'
    layout = go.Layout(
        #title = metric_name,
        xaxis = dict(title=xtitle),
        yaxis = dict(title=metric_name_to_ytitle[metric_name],dtick=5),
        hovermode = 'closest',
        width=1000,
        height=600,
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1)

    plotly.offline.plot(
        {'data': traces,'layout': layout},
        filename=filename,
        auto_open=False)


def main(exp_const):
    io.mkdir_if_not_exists(exp_const.exp_dir,recursive=True)
    
    print('Loading results ...')
    results = {
        #'chance': {},
        'random(100)': {},
        'GloVe+random(200)': {},
        'GloVe+random(100)': {},
        'GloVe': {}, 
        'GloVe+ViCo(linear,100)': {},
        'GloVe+ViCo(linear,200)': {},
        'GloVe+ViCo(select,200)': {}}
    for num_held_out_classes in exp_const.held_out_classes:
        print('Num held classes: ',num_held_out_classes)
        for embed_type in results.keys():
            print('embed type: ',embed_type)
            if embed_type=='chance':

                continue


            results[embed_type][num_held_out_classes] = {}
            num_runs = len(exp_const.runs)
            for run_dir in exp_const.runs:
                exp_dir = os.path.join(
                    run_dir,
                    exp_const.prefix[embed_type]+str(num_held_out_classes))
                results_json = os.path.join(
                    exp_dir,
                    'selected_model_results.json')
                results_ = io.load_json_object(results_json)
                for k,v in results_.items():
                    if k not in results[embed_type][num_held_out_classes]:
                        results[embed_type][num_held_out_classes][k] = []

                    results[embed_type][num_held_out_classes][k].append(v)


    for metric_name in results['GloVe'][exp_const.held_out_classes[0]].keys():
        filename = os.path.join(exp_const.exp_dir,f'{metric_name}.html')
        plot_acc_vs_classes(results,metric_name,filename)

    
    
        