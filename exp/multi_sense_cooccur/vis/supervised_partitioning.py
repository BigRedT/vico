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
    traces = []
    for embed_type in metric.keys():
        if embed_type=='GloVe' or 'random' in embed_type: # GloVe, random or their concatenation
            dash = 'dot'
        elif 'GloVe' not in embed_type and 'random' not in embed_type:
            dash = 'dash'
        else:
            dash = None # line

        trace = go.Scatter(
            x = depth,
            y = metric[embed_type],
            mode = 'lines+markers',
            name = embed_type,
            line = dict(
                width=2,
                dash=dash),
            marker = dict(size=9,symbol='circle')
        )
        traces.append(trace)
    
    layout = go.Layout(
        xaxis = dict(title='Decision Tree Depth'),
        yaxis = dict(title=metric_name),
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
        vis_dir = os.path.join(exp_const.exp_dir,'fine')
        print('*'*80)
        print('Fine Categories')
        print('*'*80)
    else:
        from . import categories as C
        vis_dir = os.path.join(exp_const.exp_dir,'coarse')
        print('*'*80)
        print('Coarse Categories')
        print('*'*80)

    io.mkdir_if_not_exists(vis_dir,recursive=True)

    #print('Reading words and categories ...')
    categories = sorted([c for c in dir(C) if '__' not in c and c!='C'])
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
        print(f'- {embed_type}',end=' ',flush=True)

        #print('Loading embeddings ...')
        embed_ = io.load_h5py_object(embed_info.word_vecs_h5py)['embeddings']
        word_to_idx = io.load_json_object(embed_info.word_to_idx_json)
        
        #print('Selecting words ...')
        words = [word for word in all_words if word in word_to_idx]
        labels = [word_to_label[word] for word in words]
        idxs = [word_to_idx[word] for word in words]

        embed = np.zeros([len(idxs),embed_.shape[1]])
        for i,j in enumerate(idxs):
            embed[i] = embed_[j]
        embed = embed_info.get_embedding(embed)

        #print(f'Computing word features ({embed_type}) ...')
        word_feats = get_word_feats(
            embed,
            dim=2,
            embed_type='original')

        #print(f'Learn Decision Tree ({embed_type}) ...')
        entropy[embed_type] = []
        accuracy[embed_type] = []
        homogeneity[embed_type] = []
        completeness[embed_type] = []
        v_measure[embed_type] = []
        ari[embed_type] = []

        if exp_const.fine==True:
            depths = [1,4,8,12,16,20,24,28,32,36,42,48,54,60,66,72,78]
        else:
            depths = [1,4,8,12,16,20,24,28,32,36,42,48,54,60,66,72,78]

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
        
        print('[Done]')
        
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


    print('')
    print('Aggregate performance across different tree depths (Copy to your latex table/spreadsheet)')
    metrics = ['v_measure','ari','accuracy']

    print('')

    print('-'*40)
    metric_str = 'Embedding'
    for metric in metrics:
        metric_str += ' & '
        metric_str += metric
    print(metric_str)
    print('-'*40)

    for embed_type in data_const.embed_info.keys():
        metric_str = embed_type
        for metric in metrics:
            metric_str += ' & '
            metric_value = round(np.mean(locals()[metric][embed_type]),2)
            metric_str += '{:.2f}'.format(metric_value)
        
        metric_str += ' \\\\'
        print(metric_str)
    print('-'*40)

    print('')