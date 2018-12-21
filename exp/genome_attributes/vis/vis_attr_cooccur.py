import os
import numpy as np
import plotly
import plotly.graph_objs as go

import utils.io as io


def vis_cooccur(cooccur,words):
    trace = go.Heatmap(
        z=cooccur[::-1],
        x=words,
        y=words[::-1],
        showscale=True)
    data = [trace]

    layout = go.Layout(
        yaxis=dict(
            tickfont=dict(size=10),
        ),
        xaxis=dict(
            tickangle=-45,
            side='top',
            tickfont=dict(size=10),
        ),
        # height=800,
        # width=800,
        # autosize=False,
        showlegend=False,
        # margin=go.layout.Margin(
        #     l=150,
        #     r=100,
        #     b=100,
        #     t=100,
        # ),
    )
    
    plot_dict = {'data':[trace],'layout': layout}
    return plot_dict


def main():
    exp_dir = os.path.join(
        os.getcwd(),
        'symlinks/exp/genome_attributes/' + \
        'resnet_18_mean_adam_loss_balanced_bce_norm1_relu_attr_coocur/')
    
    num_imgs_per_class_npy = os.path.join(exp_dir,'num_imgs_per_class.npy')
    num_imgs_per_class = np.load(num_imgs_per_class_npy)

    cooccur_npy = os.path.join(exp_dir,'cooccur.npy')
    cooccur = np.load(cooccur_npy)
    cooccur = np.log(cooccur*num_imgs_per_class[:,None]+1)

    wnid_to_idx_json = os.path.join(exp_dir,'wnid_to_idx.json')
    wnid_to_idx = io.load_json_object(wnid_to_idx_json)

    words = []
    ids = []
    for wnid,idx in wnid_to_idx.items():
        if num_imgs_per_class[idx] > 1000:
            if wnid=='green.s.01':
                import pdb; pdb.set_trace()
            words.append(wnid)
            ids.append(idx)

    cooccur_ = cooccur[ids][:,ids]
    # cooccur_ = cooccur_ > 0.8
    # cooccur_ = cooccur_.astype(np.float32)

    plot_dict = vis_cooccur(cooccur_,words)

    plot_html = os.path.join(exp_dir,'cooccur_plot.html')
    plotly.offline.plot(
        plot_dict,
        filename=plot_html,
        auto_open=False)

if __name__=='__main__':
    main()