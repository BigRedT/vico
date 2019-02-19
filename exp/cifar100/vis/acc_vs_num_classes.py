import os
import plotly
import plotly.graph_objs as go

import utils.io as io


def plot_acc_vs_classes(results,metric_name,filename):
    embed_type_to_color = {
        'glove+visual': 'rgb(31, 119, 180)',
        'glove+visual(xformed)': 'rgb(31, 119, 180)', 
        'glove': 'rgb(44, 150, 44)',
        'visual': 'rgb(214, 39, 40)',
        'visual(xformed)': 'rgb(214, 39, 40)',
        'glove+random': 'rgb(255, 127, 14)',
        'random': 'grey',
    }

    metric_name_to_ytitle = {
        'HM Acc': 'Harmonic Mean of Seen and Unseen Classes Accuracies)',
        'Seen Acc': 'Seen Classes Accuracy (%)',
        'Unseen Acc': 'Unseen Classes Accuracy (%)',
        'Step': 'Best model iterations',
    }

    num_held_out_classes = sorted(list(results['glove'].keys()))

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
            #mode = 'lines+markers',
            name = 'random',
            # line = dict(
            #     color=embed_type_to_color[embed_type],
            #     width=2,
            #     dash=None),
            marker = dict(color=embed_type_to_color['random']),
            opacity=0.9,
        )
        traces.append(trace)

    for embed_type in results.keys():
        y = []
        for num_held in num_held_out_classes:
            y.append(
                round(results[embed_type][num_held][metric_name],1))

        # trace = go.Scatter(
        #     x = [100-x_ for x_ in num_held_out_classes],
        #     y = y,
        #     mode = 'lines+markers',
        #     name = embed_type,
        #     line = dict(
        #         color=embed_type_to_color[embed_type],
        #         width=2,
        #         dash=None),
        #     marker = dict(size=9,symbol='circle'),
        # )
        trace = go.Bar(
            x = [100-x_ for x_ in num_held_out_classes],
            y = y,
            text = y,
            textposition = 'auto',
            #mode = 'lines+markers',
            name = embed_type,
            # line = dict(
            #     color=embed_type_to_color[embed_type],
            #     width=2,
            #     dash=None),
            marker = dict(color=embed_type_to_color[embed_type]),
            opacity=0.9,
        )
        traces.append(trace)

    xtitle = '#Trainable classes (= 100 - #Held out classes)'
    layout = go.Layout(
        #title = metric_name,
        xaxis = dict(title=xtitle),
        yaxis = dict(title=metric_name_to_ytitle[metric_name]),
        hovermode = 'closest',
        width=800,
        height=800)

    plotly.offline.plot(
        {'data': traces,'layout': layout},
        filename=filename,
        auto_open=False)


def main(exp_const):
    io.mkdir_if_not_exists(exp_const.exp_dir,recursive=True)
    
    print('Loading results ...')
    results = {'glove': {}, 'glove+visual': {}}
    for num_held_out_classes in exp_const.held_out_classes:
        glove_exp_dir = os.path.join(
            exp_const.out_base_dir,
            exp_const.glove_prefix+str(num_held_out_classes))
        glove_results_json = os.path.join(
            glove_exp_dir,
            'selected_model_results.json')
        results['glove'][num_held_out_classes] = \
                io.load_json_object(glove_results_json)

        visual_exp_dir = os.path.join(
            exp_const.out_base_dir,
            exp_const.visual_prefix+str(num_held_out_classes))
        visual_results_json = os.path.join(
            visual_exp_dir,
            'selected_model_results.json')
        results['glove+visual'][num_held_out_classes] = \
                io.load_json_object(visual_results_json)


    for metric_name in results['glove'][exp_const.held_out_classes[0]].keys():
        filename = os.path.join(exp_const.exp_dir,f'{metric_name}.html')
        plot_acc_vs_classes(results,metric_name,filename)

    
    
        