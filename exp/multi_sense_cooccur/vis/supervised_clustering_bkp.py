import os
import numpy as np
from tqdm import tqdm
import plotly
import plotly.graph_objs as go
from sklearn.manifold import TSNE
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics as skmetrics

import utils.io as io
#from . import categories as C
#from . import fine_categories as C


def get_tsne_embeddings(embed,dim,nruns=3):
    tsne = TSNE(
        n_components=dim,
        perplexity=30,
        verbose=0, #3
        learning_rate=200,
        metric='cosine',
        n_iter=10000)

    best_kld = 10^6
    best_embed = None
    for i in range(nruns):
        tsne_embed = tsne.fit_transform(embed)
        if tsne.kl_divergence_ < best_kld:
            best_kld = tsne.kl_divergence_
            best_embed = tsne_embed
    
    print('Best KLD: ',best_kld)
    return best_embed


def get_word_feats(embed,dim=2,embed_type='tsne'):
    if embed_type=='original':
        return embed
    elif embed_type=='tsne':
        return get_tsne_embeddings(embed,dim)
    else:
        assert(False), f'embed_type {embed_type} not implemented'


def plot_metric_vs_depth(metric_name,metric,depth,filename):
    embed_type_to_color = {
        'GloVe+ViCo(linear)': 'rgb(31, 119, 180)',
        'GloVe+ViCo(xformed)': 'rgb(31, 119, 180)',
        'GloVe+ViCo(select)': 'rgb(31, 119, 180)', #'rgb(40,40,40)',
        'GloVe': 'rgb(44, 150, 44)',
        'ViCo(linear)': 'rgb(214, 39, 40)',
        'ViCo(select)': 'rgb(214, 39, 40)', #'rgb(135, 93, 183)',
        'ViCo(xformed)': 'rgb(214, 39, 40)',
        'GloVe+random': 'rgb(255, 127, 14)',
        'random': 'grey',
    }

    traces = []
    for embed_type in metric.keys():
        if 'xformed' in embed_type:
            dash = 'dot'
            symbol = 'circle'
        elif 'select' in embed_type:
            dash = 'dot'
            symbol = 'circle'
        else:
            dash = None
            symbol = 'circle'

        trace = go.Scatter(
            x = depth,
            y = metric[embed_type],
            mode = 'lines+markers',
            name = embed_type,
            line = dict(
                color=embed_type_to_color[embed_type],
                width=2,
                dash=dash),
            marker = dict(size=9,symbol=symbol)
        )
        traces.append(trace)
    
    layout = go.Layout(
        #title = metric_name,
        xaxis = dict(title='Decision Tree Depth'),
        yaxis = dict(title=metric_name),
        hovermode = 'closest',
        width=800,
        height=800)

    plotly.offline.plot(
        {'data': traces,'layout': layout},
        filename=filename,
        auto_open=False)


def main(exp_const,data_const):
    if exp_const.fine==True:
        from . import fine_categories as C
        vis_dir = os.path.join(exp_const.vis_dir,'supervised_clustering_fine')
    else:
        from . import categories as C
        vis_dir = os.path.join(exp_const.vis_dir,'supervised_clustering')
    
    io.mkdir_if_not_exists(vis_dir,recursive=True)

    print('Reading words and categories ...')
    categories = sorted([c for c in dir(C) if '__' not in c and c!='C'])
    #import pdb; pdb.set_trace()
    categories_to_idx = {l:i for i,l in enumerate(categories)}
    all_words = set()
    word_to_label = {}
    for category in categories:
        category_words = getattr(C,category)
        all_words.update(category_words)
        for word in category_words:
            word_to_label[word] = category
    
    print('Loading embeddings ...')
    embed = io.load_h5py_object(
        data_const.word_vecs_h5py)['embeddings'][()]
    xformed_embed = io.load_h5py_object(
        data_const.xformed_word_vecs_h5py)['embeddings'][()]
    select_embed = io.load_h5py_object(
        data_const.select_word_vecs_h5py)['embeddings'][()]
    word_to_idx = io.load_json_object(data_const.word_to_idx_json)

    print('Selecting words ...')
    words = [word for word in all_words if word in word_to_idx]
    labels = [word_to_label[word] for word in words]
    idxs = [word_to_idx[word] for word in words]
    embed = embed[idxs,:]
    visual_embed = embed[:,300:]
    xformed_embed = xformed_embed[idxs,:]
    xformed_visual_embed = xformed_embed[:,300:]
    select_embed = select_embed[idxs,:]
    select_visual_embed = select_embed[:,300:]
    glove_embed = embed[:,:300]
    random_embed = np.copy(embed)
    random_embed[:,300:] = 10*0.1*np.random.rand(
        embed.shape[0],
        visual_embed.shape[1])
    
    embed_type_to_embed = {
        'GloVe+ViCo(linear)': embed,
        #'GloVe+ViCo(xformed)': xformed_embed,
        'GloVe+ViCo(select)': select_embed,
        'ViCo(linear)': visual_embed,
        #'ViCo(xformed)': xformed_visual_embed,
        'ViCo(select)': select_visual_embed,
        'GloVe': glove_embed,
        'GloVe+random': random_embed,
        'random': random_embed[:,300:]
    }

    entropy = {}
    accuracy = {}
    homogeneity = {}
    completeness = {}
    v_measure = {}
    ari = {}
    for embed_type in embed_type_to_embed.keys():
        print(f'Computing word features ({embed_type}) ...')
        word_feats = get_word_feats(
            embed_type_to_embed[embed_type],
            dim=2,
            embed_type='original')

        print(f'Learn Decision Tree ({embed_type}) ...')
        entropy[embed_type] = []
        accuracy[embed_type] = []
        homogeneity[embed_type] = []
        completeness[embed_type] = []
        v_measure[embed_type] = []
        ari[embed_type] = []

        #depths = [1,2,4,6,8,10,12,14,16]
        if exp_const.fine==True:
            depths = [1,4,8,12,16,20,24,28,32,36,42,48]
        else:
            depths = [1,4,8,12,16,20,24] # 28,32

        for depth in depths:
            dt = DecisionTreeClassifier(
                criterion='gini',
                max_depth=depth,
                min_samples_leaf=2,
            )
            dt.fit(word_feats,labels)
            prob = dt.predict_proba(word_feats)
            pred_labels = dt.predict(word_feats)

            confmat = np.zeros([len(categories),len(categories)])
            counts = np.zeros([len(categories),1])
            for i,label in enumerate(labels):
                r = categories_to_idx[label]
                confmat[r] += prob[i]
                counts[r,0] += 1

            confmat = confmat / counts
            ce = -np.mean(np.sum(confmat*np.log(confmat+1e-6),1))

            acc = 0
            for gt_l,pred_l in zip(labels,pred_labels):
                acc += gt_l==pred_l
            acc = acc / len(labels)
            
            homo_score,comp_score,v_measure_score = \
                skmetrics.homogeneity_completeness_v_measure(
                    labels,
                    pred_labels)
            ari_score = skmetrics.adjusted_rand_score(labels,pred_labels)

            entropy[embed_type].append(ce)
            accuracy[embed_type].append(acc)
            homogeneity[embed_type].append(homo_score)
            completeness[embed_type].append(comp_score)
            v_measure[embed_type].append(v_measure_score)
            ari[embed_type].append(ari_score)
        
        print('*'*80)
        print(embed_type)
        print('*'*80)
        print('entropy',entropy[embed_type])
        print('accuracy',accuracy[embed_type])
        print('homogeneity',homogeneity[embed_type])
        print('completeness',completeness[embed_type])
        print('v_measure',v_measure[embed_type])
        print('ari',ari[embed_type])
        
        plot_metric_vs_depth(
            'Accuracy',
            accuracy,
            depths,
            os.path.join(vis_dir,'accuracy.html'))

        plot_metric_vs_depth(
            'Homogeneity',
            homogeneity,
            depths,
            os.path.join(vis_dir,'homogeneity.html'))

        plot_metric_vs_depth(
            'Completeness',
            completeness,
            depths,
            os.path.join(vis_dir,'completeness.html'))

        plot_metric_vs_depth(
            'V-Measure',
            v_measure,
            depths,
            os.path.join(vis_dir,'v_measure.html'))

        plot_metric_vs_depth(
            'Adjusted Rand Index',
            ari,
            depths,
            os.path.join(vis_dir,'ari.html'))
