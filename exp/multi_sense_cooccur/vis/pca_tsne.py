import os
import numpy as np
from tqdm import tqdm
import plotly
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


import utils.io as io
from .categories import *


def get_tsne_embeddings(embed,tsne,nruns=3):
    best_kld = 10^6
    best_embed = None
    for i in range(nruns):
        tsne_embed = tsne.fit_transform(embed)
        if tsne.kl_divergence_ < best_kld:
            best_kld = tsne.kl_divergence_
            best_embed = tsne_embed
    
    print('Best KLD: ',best_kld)
    return best_embed


def scatter_plot(
        embed,
        words,
        colors,
        symbols,
        category_to_symbol,
        sizes,
        title,
        filename):
    symbol_to_category = {s:c for c,s in category_to_symbol.items()}
    symbol_to_category['circle'] = 'OTHER'
    unique_symbols = set(symbols)
    traces = [None]*len(unique_symbols)
    for i,symbol in enumerate(unique_symbols):
        idxs = [k for k,s in enumerate(symbols) if s==symbol]
        traces[i] = go.Scatter(
            x = embed[idxs,0].tolist(),
            y = embed[idxs,1].tolist(),
            mode='markers',
            marker=dict(
                color=[-colors[k] for k in idxs],
                symbol=[symbols[k] for k in idxs],
                size=[sizes[k] for k in idxs],
                line=dict(
                    color='rgb(0,0,0)',
                    width=1),
                showscale=False,
                colorscale='RdBu'),
            text=[words[k] for k in idxs],
            name=symbol_to_category[symbol].lower())
    layout = go.Layout(
        title = title,
        xaxis = dict(title='dim 1'),
        yaxis = dict(title='dim 2'),
        hovermode = 'closest')
    plotly.offline.plot(
        {'data': traces,'layout': layout},
        filename=filename,
        auto_open=False)


def main(exp_const,data_const):
    io.mkdir_if_not_exists(exp_const.vis_dir)

    category_to_symbol = {
        'FOOD': 'triangle-up',
        'ANIMALS': 'square',
        'COLORS': 'star',
        'APPLIANCES': 'hexagon',
        'UTENSILS': 'square-dot',
        'TRANSPORT': 'hexagram',
        'HUMANS': 'hourglass',
        'CLOTHES': 'pentagon',
        'ELECTRONICS': 'triangle-down-dot',
        'NUMBERS': 'circle-cross',
        'BUILDINGS': 'circle-dot',
        'ACTIONS': 'star-diamond',
        'BODYPARTS': 'bowtie',
    }

    category_words = set()
    for k in category_to_symbol.keys():
        for word in globals()[k]:
            category_words.add(word)


    print('Loading embeddings ...')
    embed = io.load_h5py_object(
        data_const.word_vecs_h5py)['embeddings'][()]
    word_to_idx = io.load_json_object(data_const.word_to_idx_json)
    object_freqs = io.load_json_object(data_const.object_freqs_json)
    attribute_freqs = io.load_json_object(data_const.attribute_freqs_json)

    print('Selecting words ...')
    visual_words = set()
    for word,freq in object_freqs.items():
        if freq > 100:
            if exp_const.category_words_only==True:
                if word in category_words:
                    visual_words.add(word)
            else:
                visual_words.add(word)

    for word,freq in attribute_freqs.items():
        if freq > 100:
            if exp_const.category_words_only==True:
                if word in category_words:
                    visual_words.add(word)
            else:
                visual_words.add(word)

    words = [word for word in visual_words if word in word_to_idx]
    idxs = [word_to_idx[word] for word in words]
    embed = embed[idxs,:]
    visual_embed = embed[:,300:]
    glove_embed = embed[:,:300]
    print('Selected words', visual_embed.shape[0])

    print('Assigning marker attributes ...')
    colors = []
    symbols = []
    sizes = []
    for word in words:
        word_type = 0
        freq = 1#0
        if word in object_freqs:
            word_type += 1
            freq *= object_freqs[word]
        
        if word in attribute_freqs:
            word_type += -1
            freq /= attribute_freqs[word]

        size = 9
        symbol = 'circle'
        for category in category_to_symbol.keys():
            if word in globals()[category]:
                symbol = category_to_symbol[category]
                size = 18
        
        sizes.append(size)
        symbols.append(symbol)
        #colors.append(word_type)
        colors.append(np.log(freq))

    if len(words) > 10000:
        import pdb; pdb.set_trace()

    print('Performing PCA ...')
    pca = PCA(n_components=2)
    visual_embed_pca = pca.fit_transform(visual_embed)

    print('Plotting PCA ...')
    filename = os.path.join(
            exp_const.vis_dir,
            f'pca.html')
    scatter_plot(
        visual_embed_pca,
        words,
        colors,
        symbols,
        category_to_symbol,
        sizes,
        'PCA',
        filename)

    for perplexity in [30]: #[5,30,50,100]:
        for lr in [200]: #[100,200,500,1000]:
            print(f'Performing TSNE (Perplexity: {perplexity} LR: {lr}) ...')
            tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                verbose=3,
                learning_rate=lr,
                metric='cosine',
                n_iter=10000)

            if exp_const.category_words_only==True:
                suffix = '_category_words_only'
            else:
                suffix = ''

            embed_types_to_embed = {
                'visual': visual_embed,
                'glove': glove_embed,
                'joint': embed,
            }

            for embed_type in embed_types_to_embed.keys():
                print(f'TSNE on {embed_type} embeddings ...')
                embed_tsne = get_tsne_embeddings(
                    embed_types_to_embed[embed_type],
                    tsne)

                print('Plotting TSNE ...')
                filename = os.path.join(
                    exp_const.vis_dir,
                    f'{embed_type}_tsne_perplexity_{perplexity}_lr_{lr}' + \
                    f'{suffix}.html')
                scatter_plot(
                    embed_tsne,
                    words,
                    colors,
                    symbols,
                    category_to_symbol,
                    sizes,
                    'TSNE',
                    filename)

                embed_tsne_npy = os.path.join(
                    exp_const.exp_dir,
                    f'{embed_type}_tsne_perplexity_{perplexity}_lr_{lr}' + \
                    f'{suffix}.npy')
                np.save(embed_tsne_npy,embed_tsne)

                embed_npy = os.path.join(
                    exp_const.vis_dir,
                    f'{embed_type}_tsne_embed{suffix}.npy')
                io.dump_json_object(
                    embed_types_to_embed[embed_type],
                    embed_npy)

    words_json = os.path.join(exp_const.vis_dir,f'tsne_words{suffix}.json')
    io.dump_json_object(words,words_json)