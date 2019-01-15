import os
import numpy as np
import plotly
import plotly.graph_objs as go

import utils.io as io


def get_pair_counts(glove_sim,visual_sim,del_confmat,labels):
    pair_counts = {
        'g+v+': {'conf_red': 0, 'conf_inc': 0},
        'g+v-': {'conf_red': 0, 'conf_inc': 0},
        'g-v-': {'conf_red': 0, 'conf_inc': 0},
        'g-v+': {'conf_red': 0, 'conf_inc': 0},
    }
    for i, label1 in enumerate(labels):
        for j, label2 in enumerate(labels):
            if i==j:
                continue

            if abs(del_confmat[i,j]) <= 1e-6:
                continue

            quad = ''
            if glove_sim[i,j] > 0:
                quad += 'g+'
            else:
                quad += 'g-'
            
            if visual_sim[i,j] > 0:
                quad += 'v+'
            else:
                quad += 'v-'

            if del_confmat[i,j] < 0:
                pair_counts[quad]['conf_red'] += 1
            else:
                pair_counts[quad]['conf_inc'] += 1

    return pair_counts


def create_scatter_plot(del_confmat,visual_sim,glove_sim,labels,filename):
    text = []
    x = []
    y = []
    color = []
    for i, label1 in enumerate(labels):
        for j, label2 in enumerate(labels):
            if i==j:
                continue

            if abs(del_confmat[i,j]) < 1e-6:
                continue

            round_del_conf = str(-round(del_confmat[i,j],2))
            text.append(
                f'{label1} / {label2} (reduction in confusion: {round_del_conf})')
            x.append(visual_sim[i,j])
            y.append(glove_sim[i,j])
            color.append(-del_confmat[i,j])

    trace = go.Scatter(
        x=x,
        y=y,
        marker=dict(
            color=color,
            size=11,
            showscale=True
        ),
        mode='markers',
        text=text)

    layout = go.Layout(
        xaxis=dict(
            title='Visual Similarity'
        ),
        yaxis=dict(
            title='Glove Similarity'
        ),
        hovermode='closest'
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
    
    if exp_const.cosine==True:
        glove_vecs = glove_vecs / np.linalg.norm(glove_vecs,2,1,True)
        visual_vecs = visual_vecs / np.linalg.norm(visual_vecs,2,1,True)

    visual_sim = np.matmul(visual_vecs,np.transpose(visual_vecs))
    glove_sim = np.matmul(glove_vecs,np.transpose(glove_vecs))

    visual_sim = visual_sim - np.mean(visual_sim)
    glove_sim = glove_sim - np.mean(glove_sim)
    visual_sim = visual_sim / np.std(visual_sim)
    glove_sim = glove_sim / np.std(glove_sim)

    visual_confmat = np.maximum(0,np.log(visual_confmat+1e-6))
    glove_confmat = np.maximum(0,np.log(glove_confmat+1e-6))
    del_confmat = visual_confmat - glove_confmat

    if exp_const.cosine==True:
        name = 'conf_as_fun_of_cosine_sims.html'
    else:
        name = 'conf_as_fun_of_sims.html'

    filename = os.path.join(exp_const.vis_dir,name)
    create_scatter_plot(del_confmat,visual_sim,glove_sim,labels,filename)

    pair_counts = get_pair_counts(glove_sim,visual_sim,del_confmat,labels)
    pair_counts_json = os.path.join(exp_const.exp_dir,'conf_pair_counts.json')
    print(pair_counts_json)
    io.dump_json_object(pair_counts,pair_counts_json)

    print(pair_counts)

