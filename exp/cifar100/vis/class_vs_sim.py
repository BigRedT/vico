import os
import numpy as np
import plotly
import plotly.graph_objs as go
from scipy.stats import pearsonr as pearsoncorr

import utils.io as io


def create_scatter_plot(confmat,sim,labels,filename):
    text = []
    x = []
    y = []
    for i, label1 in enumerate(labels):
        for j, label2 in enumerate(labels):
            if i==j:
                continue

            # if confmat[i,j]==0:
            #     continue

            x.append(sim[i,j])
            y.append(confmat[i,j])
            text.append(f'{label1} / {label2}')

    trace = go.Scatter(
        x=x,
        y=y,
        marker=dict(
            size=11,
        ),
        mode='markers',
        text=text)

    layout = go.Layout(
        xaxis=dict(
            title='Embedding Similarity'
        ),
        yaxis=dict(
            title='Confusion'
        ),
        hovermode='closest'
    )

    plotly.offline.plot(
        {'data': [trace],'layout': layout},
        filename=filename,
        auto_open=False)

    return x,y,text

def main(exp_const,data_const):
    class_confmat = np.load(data_const.class_confmat_npy)
    visual_embed = np.load(data_const.visual_embed_npy)
    labels = np.load(data_const.labels_npy)

    glove_vecs = visual_embed[:,:data_const.glove_dim]
    visual_vecs = visual_embed[:,data_const.glove_dim:]

    visual_sim = np.matmul(visual_vecs,np.transpose(visual_vecs))
    glove_sim = np.matmul(glove_vecs,np.transpose(glove_vecs))

    corr_pvalue = {}
    filename = os.path.join(exp_const.vis_dir,'class_vs_glove_visual_sim.html')
    x,y,_ = create_scatter_plot(
        class_confmat,
        glove_sim+visual_sim,
        labels,
        filename)
    corr_pvalue['glove+visual'] = pearsoncorr(x,y)

    filename = os.path.join(exp_const.vis_dir,'class_vs_visual_sim.html')
    create_scatter_plot(class_confmat,visual_sim,labels,filename)
    x,y,_ = create_scatter_plot(
        class_confmat,
        visual_sim,
        labels,
        filename)
    corr_pvalue['visual'] = pearsoncorr(x,y)

    filename = os.path.join(exp_const.vis_dir,'class_vs_glove_sim.html')
    create_scatter_plot(class_confmat,glove_sim,labels,filename)
    x,y,_ = create_scatter_plot(
        class_confmat,
        glove_sim,
        labels,
        filename)
    corr_pvalue['glove'] = pearsoncorr(x,y)

    corr_pvalue_json = os.path.join(exp_const.exp_dir,'corr_pvalue.json')
    io.dump_json_object(corr_pvalue,corr_pvalue_json)
