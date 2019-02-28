import os
import plotly
import plotly.graph_objs as go

import utils.io as io


def plot_acc_vs_classes(results,metric_name,filename):
    embed_type_to_color = {
        'GloVe+ViCo(linear)': 'rgb(55, 128, 191)',
        'GloVe+ViCo(xformed)': 'rgb(31, 119, 180)', 
        'GloVe+ViCo(select)': 'rgb(219, 64, 82)',
        'GloVe': 'rgb(44, 150, 44)',
        'ViCo(linear)': 'rgb(214, 39, 40)',
        'ViCo(xformed)': 'rgb(214, 39, 40)',
        'GloVe+random': 'rgb(255, 127, 14)',
        'random': 'grey',
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
            name = 'random',
            marker = dict(color=embed_type_to_color['random']),
            opacity=0.9,
        )
        traces.append(trace)

    for embed_type in results.keys():
        y = []
        for num_held in num_held_out_classes:
            y.append(
                round(results[embed_type][num_held][metric_name],1))

        trace = go.Bar(
            x = [100-x_ for x_ in num_held_out_classes],
            y = y,
            text = y,
            textposition = 'auto',
            name = embed_type,
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
        height=800,
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
        'GloVe': {}, 
        'GloVe+ViCo(linear)': {},
        'GloVe+ViCo(select)': {}}
    for num_held_out_classes in exp_const.held_out_classes:
        glove_exp_dir = os.path.join(
            exp_const.out_base_dir,
            exp_const.glove_prefix+str(num_held_out_classes))
        glove_results_json = os.path.join(
            glove_exp_dir,
            'selected_model_results.json')
        results['GloVe'][num_held_out_classes] = \
                io.load_json_object(glove_results_json)

        visual_exp_dir = os.path.join(
            exp_const.out_base_dir,
            exp_const.visual_prefix+str(num_held_out_classes))
        visual_results_json = os.path.join(
            visual_exp_dir,
            'selected_model_results.json')
        results['GloVe+ViCo(linear)'][num_held_out_classes] = \
                io.load_json_object(visual_results_json)

        visual_exp_dir = os.path.join(
            exp_const.out_base_dir,
            exp_const.visual_select_prefix+str(num_held_out_classes))
        visual_results_json = os.path.join(
            visual_exp_dir,
            'selected_model_results.json')
        results['GloVe+ViCo(select)'][num_held_out_classes] = \
                io.load_json_object(visual_results_json)
        


    for metric_name in results['GloVe'][exp_const.held_out_classes[0]].keys():
        filename = os.path.join(exp_const.exp_dir,f'{metric_name}.html')
        plot_acc_vs_classes(results,metric_name,filename)

    
    
        