import plotly
import plotly.graph_objs as go
import numpy as np
import os

def create_confmat_heatmap(confmat,labels,filename):
    #confmat = confmat / np.sum(confmat,1,keepdims=True)
    confmat = np.maximum(0,np.log(confmat+1e-6))
    
    trace = go.Heatmap(
        z=confmat[::-1],
        x=labels,
        y=labels[::-1],
        showscale=True)
    data = [trace]

    layout = go.Layout(
        yaxis=dict(
            tickfont=dict(size=9),
        ),
        xaxis=dict(
            tickangle=-90, #-45,
            side='top',
            tickfont=dict(size=9),
        ),
        # height=800,
        # width=800,
        autosize=True,
        showlegend=False,
        # margin=go.Margin(
        #     l=150,
        #     r=100,
        #     b=100,
        #     t=100,
        # ),
    )

    plotly.offline.plot(
        {'data': [trace], 'layout': layout},
        filename=filename,
        auto_open=False)


def main(exp_const):
    confmat_npy = os.path.join(exp_const.exp_dir,'confmat.npy')
    confmat = np.load(confmat_npy)

    labels_npy = os.path.join(exp_const.exp_dir,'labels.npy')
    labels = np.load(labels_npy)

    confmat_html = os.path.join(exp_const.vis_dir,'confmat_log.html')
    create_confmat_heatmap(confmat,labels,confmat_html)
