import os
import numpy as np
import plotly
import plotly.graph_objs as go

import utils.io as io


def create_scatter_plot(del_confmat,visual_sim,glove_sim,labels,filename):
    mean_glove_sim = np.mean(glove_sim)
    std_glove_sim = np.std(glove_sim)
    text = []
    x = []
    y = []
    for i, label1 in enumerate(labels):
        for j, label2 in enumerate(labels):
            round_glove_sim = str(round(glove_sim[i,j],2))
            text.append(
                f'{label1} / {label2} (glove: {round_glove_sim})')
            x.append(visual_sim[i,j])
            y.append(del_confmat[i,j])

    trace = go.Scatter(
        x = x,
        y = y,
        mode = 'markers',
        text = text)

    layout = go.Layout(
        xaxis = dict(
            title = 'Visual Similarity'
        ),
        yaxis = dict(
            title = 'Conf Visual - Conf Glove'
        ),
        hovermode = 'closest'
    )

    plotly.offline.plot(
        {'data': [trace],'layout': layout},
        filename=filename,
        auto_open=False)
    


def main(exp_const,data_const):
    visual_confmat = np.load(data_const.visual_confmat_npy)
    glove_confmat = np.load(data_const.glove_confmat_npy)
    visual_embed = np.load(data_const.visual_embed_npy)
    labels = np.load(data_const.labels_npy)

    glove_vecs = visual_embed[:,:data_const.glove_dim]
    visual_vecs = visual_embed[:,data_const.glove_dim:]
    visual_sim = np.matmul(visual_vecs,np.transpose(visual_vecs))
    glove_sim = np.matmul(glove_vecs,np.transpose(glove_vecs))

    visual_confmat = np.maximum(0,np.log(visual_confmat+1e-6))
    glove_confmat = np.maximum(0,np.log(glove_confmat+1e-6))
    del_confmat = visual_confmat - glove_confmat

    filename = os.path.join(exp_const.vis_dir,'conf_vs_visual_sim.html')

    create_scatter_plot(del_confmat,visual_sim,glove_sim,labels,filename)


