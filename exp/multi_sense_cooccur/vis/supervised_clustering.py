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
        'ViCo(linear,50)': 'rgb(55, 128, 191,0.6)', 
        'ViCo(linear,100)': 'rgba(55, 128, 191,0.6)',
        'ViCo(linear,200)': 'rgba(55, 80, 191,0.6)',
        'ViCo(select,200)': 'rgba(219, 64, 82,0.6)',
        'GloVe+ViCo(linear,100, w/o WordNet)': 'rgb(139,0,139)',
        'GloVe+ViCo(linear,100)': 'rgb(55, 128, 191)',
        'GloVe+ViCo(linear,50)': 'rgb(55, 128, 191)', 
        'GloVe+ViCo(linear,200)': 'rgb(55, 80, 191)',
        'GloVe+ViCo(select,200)': 'rgb(219, 64, 82)',
        'GloVe': 'rgb(44, 150, 44)',
        'ViCo(linear)': 'rgb(214, 39, 40)',
        'GloVe+random(100)': 'rgb(255, 200, 14)',
        'GloVe+random(200)': 'rgb(255, 127, 14)',
        'random(100)': 'grey',
        'random(300)': 'rgb(96,96,96)',
        'vis-w2v': 'black',
        'GloVe+vis-w2v': 'black',
    }

    traces = []
    for embed_type in metric.keys():
        if embed_type=='GloVe' or 'random' in embed_type:
            dash = 'dot'
        else:
            dash = None

        if 'GloVe' in embed_type or 'random' in embed_type:
            symbol = 'circle'
        else:
            symbol = 'circle'#'square'
            dash = 'dash'

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
        yaxis = dict(title=metric_name,range=[0,0.98]),
        hovermode = 'closest',
        width=1100,
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
        vis_dir = os.path.join(exp_const.vis_dir,'supervised_clustering_coarse')
    
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

    entropy = {}
    accuracy = {}
    homogeneity = {}
    completeness = {}
    v_measure = {}
    ari = {}
    for embed_type, embed_info in data_const.embed_info.items():
        print('Loading embeddings ...')
        embed_ = io.load_h5py_object(embed_info.word_vecs_h5py)['embeddings']
        word_to_idx = io.load_json_object(embed_info.word_to_idx_json)
        
        print('Selecting words ...')
        words = [word for word in all_words if word in word_to_idx]
        labels = [word_to_label[word] for word in words]
        idxs = [word_to_idx[word] for word in words]

        embed = np.zeros([len(idxs),embed_.shape[1]])
        for i,j in enumerate(idxs):
            embed[i] = embed_[j]
        embed = embed_info.get_embedding(embed)

        print(f'Computing word features ({embed_type}) ...')
        word_feats = get_word_feats(
            embed,
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
            depths = [1,4,8,12,16,20,24,28,32,36,42,48,54,60,66,72,78]
        else:
            depths = [1,4,8,12,16,20,24,28,32,36,42,48,54,60,66,72,78] #,32,36,42,48] # 28,32

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


    print('*'*80)
    print('Aggregate stats across all clusters')
    print('*'*80)
    metrics = ['v_measure','ari','accuracy']

    metric_str = ' '
    for metric in metrics:
        metric_str += ' & '
        metric_str += metric
    print(metric_str)

    for embed_type in data_const.embed_info.keys():
        metric_str = embed_type
        for metric in metrics:
            metric_str += ' & '
            metric_value = round(np.mean(locals()[metric][embed_type]),2)
            metric_str += '{:.2f}'.format(metric_value)
        print(metric_str)