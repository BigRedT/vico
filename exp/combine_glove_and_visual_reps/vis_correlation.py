import os
import h5py
from tqdm import tqdm
import numpy as np
import plotly
import plotly.graph_objs as go

import utils.io as io


def vis(visual_words,glove_sim_mat,visual_sim_mat):
    num_words = len(visual_words)
    words = []
    glove_sim = []
    visual_sim = []
    for i in range(0,num_words,10):
        for j in range(i+1,num_words,10):
            words.append(visual_words[i] + ' / ' + visual_words[j])
            glove_sim.append(glove_sim_mat[i,j])
            visual_sim.append(visual_sim_mat[i,j])
    
    trace = go.Scatter(
        x = glove_sim,
        y = visual_sim,
        mode = 'markers',
        text=words)

    layout = go.Layout(
        xaxis={'title': 'Glove Similarity'},
        yaxis={'title': 'Visual Similarity'},
        hovermode='closest')
    
    plot_dict = {'data':[trace],'layout': layout}
    return plot_dict

def main(
        plot_html,
        visual_word_vecs_h5py,
        visual_word_vecs_idx_json,
        visual_words_json,
        visual_freq_json,
        glove_dim=300):
    visual_words = io.load_json_object(visual_words_json)
    visual_freq = io.load_json_object(visual_freq_json)
    visual_word_vecs_idx = io.load_json_object(visual_word_vecs_idx_json)
    visual_word_vecs = io.load_h5py_object(
        visual_word_vecs_h5py)['embeddings']
    visual_dim = visual_word_vecs.shape[1] - glove_dim
    num_words = len(visual_words)
    
    print('Subsampling ...')
    visual_words_ = []
    for word in visual_words:
        if visual_freq[word] > 10:
            visual_words_.append(word)
    #visual_words_ = visual_words[::10]
    num_words_ = len(visual_words_)
    print('Num words selected: ',num_words_)

    print('Loading vectors ...')
    glove = np.zeros([num_words_,glove_dim])
    visual = np.zeros([num_words_,visual_dim])
    for i,word in enumerate(tqdm(visual_words_)):
        idx = visual_word_vecs_idx[word]
        vec = visual_word_vecs[idx]
        glove[i] = vec[:glove_dim]
        visual[i] = vec[glove_dim:]

    print('Computing similarity ...')
    glove_norm = np.linalg.norm(glove,2,1,keepdims=True)
    glove = glove / (glove_norm+1e-6)
    glove_sim_mat = np.matmul(glove,glove.transpose())
    
    visual_norm = np.linalg.norm(visual,2,1,keepdims=True)
    visual = visual / (visual_norm+1e-6)
    visual_sim_mat = np.matmul(visual,visual.transpose())

    print('Visualizing ...')
    plot_dict = vis(visual_words_,glove_sim_mat,visual_sim_mat)
    plotly.offline.plot(
        plot_dict,
        filename=plot_html,
        auto_open=False)


if __name__=='__main__':
    plot_html = visual_words_json = os.path.join(
        os.getcwd(),
        'symlinks/exp/combine_glove_visual_reps/' + \
        'concat_glove_visual_avg_reps_balanced_bce_norm1/' + \
        'visual_glove_correlation.html')
    visual_words_json = os.path.join(
        os.getcwd(),
        'symlinks/exp/combine_glove_visual_reps/' + \
        'concat_glove_visual_avg_reps_balanced_bce_norm1/' + \
        'visual_words.json')
    visual_freq_json = os.path.join(
        os.getcwd(),
        'symlinks/exp/combine_glove_visual_reps/' + \
        'concat_glove_visual_avg_reps_balanced_bce_norm1/' + \
        'visual_freq.json')
    visual_word_vecs_idx_json = os.path.join(
        os.getcwd(),
        'symlinks/exp/combine_glove_visual_reps/' + \
        'concat_glove_visual_avg_reps_balanced_bce_norm1/' + \
        'visual_word_vecs_idx.json')
    visual_word_vecs_h5py = os.path.join(
        os.getcwd(),
        'symlinks/exp/combine_glove_visual_reps/' + \
        'concat_glove_visual_avg_reps_balanced_bce_norm1/' + \
        'visual_word_vecs.h5py')
    
    main(
        plot_html,
        visual_word_vecs_h5py,
        visual_word_vecs_idx_json,
        visual_words_json,
        visual_freq_json)
