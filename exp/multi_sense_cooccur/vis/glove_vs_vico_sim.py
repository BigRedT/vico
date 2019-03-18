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


def plot_scatter(glove_scores,vico_scores,pairs,filename):
    trace = go.Scatter(
        x = glove_scores,
        y = vico_scores,
        text=pairs,
        mode = 'markers',
        name = 'ViCo vs. GloVe',
        #marker = dict(size=9,symbol=symbol)
    )
    
    layout = go.Layout(
        #title = metric_name,
        xaxis = dict(title='GloVe Similarity'),
        yaxis = dict(title='ViCo Similarity',range=[0,1]),
        hovermode = 'closest',
        width=1100,
        height=1100)

    plotly.offline.plot(
        {'data': [trace],'layout': layout},
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

    sim = {}
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
        word_feats = \
            word_feats / np.linalg.norm(word_feats,ord=2,axis=1,keepdims=True)
        simmat = np.matmul(word_feats,np.transpose(word_feats))
        for i,word1 in enumerate(words):
            for j,word2 in enumerate(words):
                if j <= i:
                    continue

                pair = f'{word1} / {word2}'
                if pair not in sim:
                    sim[pair] = {}
                sim[pair][embed_type] = round(simmat[i,j],2)

    pairs = []
    glove_scores = []
    vico_scores = []
    selected_pairs = []
    for pair,scores in sim.items():
        if 'GloVe' in scores and 'ViCo' in scores:
            # if abs(scores['GloVe']-scores['ViCo']) < 0.3:
            #     continue
            # if max(scores['GloVe'],scores['ViCo']) < 0.5:
            #     continue

            pairs.append(pair)
            glove_scores.append(scores['GloVe'])
            vico_scores.append(scores['ViCo'])
            selected_pairs.append(
                [
                    pair,
                    scores['GloVe'],
                    scores['ViCo'],
                    scores['obj_attr'],
                    scores['attr_attr'],
                    scores['obj_hyp'],
                    scores['context']
                ]
            )
            
    header = [
        'Pair',
        'GloVe',
        'ViCo',
        'obj_attr',
        'attr_attr',
        'obj_hyp',
        'context'
    ]
    for i in range(1,7):
        sorted_selected_pairs = sorted(
            selected_pairs,
            key=lambda x: x[i],
            reverse=True)
        sorted_selected_pairs = [header] + sorted_selected_pairs

        select_pairs_txt = os.path.join(
            vis_dir,
            f'selected_pairs_{header[i]}.txt')
        f = open(select_pairs_txt,'w')
        for p in sorted_selected_pairs:
            print(p)
            f.write('\t'.join([str(k).ljust(25," ") for k in p] + ['\n']))
        
        #io.dump_json_object(sorted_selected_pairs,select_pairs_json)

    # filename = os.path.join(vis_dir,'glove_vs_vico_sim.html')
    # plot_scatter(glove_scores,vico_scores,pairs,filename)
    # import pdb; pdb.set_trace()