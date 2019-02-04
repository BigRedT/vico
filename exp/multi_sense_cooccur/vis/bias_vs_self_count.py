import os
import numpy as np
import torch
from tqdm import tqdm
import plotly
import plotly.graph_objs as go


import utils.io as io
from utils.model import Model
from utils.constants import save_constants
import utils.pytorch_layers as pytorch_layers
from ..models.logbilinear import LogBilinear
from ..dataset import MultiSenseCooccurDataset


def get_poly(coeffs):
    def poly(x):
        p = 0
        for c in coeffs:
            p = c + x*p
        return p
    return poly

def get_polypoints(X,coeffs):
    poly = get_poly(coeffs)
    return [poly(x) for x in X]

def grad(x,coeffs):
    d = len(coeffs)-1
    g = 0
    for i in range(d):
        g += (d-i)*coeffs[i]*x**(d-i-1)
    return g

def linearize(x,coeffs):
    poly = get_poly(coeffs)
    g = grad(x,coeffs)
    c = poly(x) - g*x
    g = str(round(g,2))
    c = str(round(c,2))
    x = str(x)
    f = f'b = {g}*logX + {c} around {x}'
    return f

def main(exp_const,data_const,model_const):
    print('Creating network ...')
    model = Model()
    model.const = model_const
    model.net = LogBilinear(model.const.net)
    if model.const.model_num is not None:
        model.net.load_state_dict(torch.load(model.const.net_path))
    b = model.net.embed1.b.weight.data.numpy()
    cooccur_type_to_idx = model.net.const.cooccur_type_to_idx

    print('Loading dataset ...')
    dataset = MultiSenseCooccurDataset(data_const)
    cooccur_types = ['attr_attr','obj_attr','obj_hyp','context']
    df = dataset.df
    df = df.loc[df['word1']==df['word2']]
    
    print('Creating word to row map ...')
    word_to_row = {}
    for i,row in tqdm(df.iterrows()):
        word = row['word1']
        word_to_row[word] = row     

    print('Collecting data points ...')
    counts = {t:[] for t in cooccur_types}
    bias = {t:[] for t in cooccur_types}
    words = {t:[] for t in cooccur_types}
    for word,idx in tqdm(dataset.word_to_idx.items()):
        if word not in word_to_row:
            continue

        row = word_to_row[word]
        for cooccur_type in cooccur_types:
            X = row[cooccur_type]

            if X < 1:
                continue

            counts[cooccur_type].append(np.log(X))
            i = cooccur_type_to_idx[cooccur_type]
            bias[cooccur_type].append(b[idx,i])
            words[cooccur_type].append(word)

    for cooccur_type in counts.keys():
        print(f'Creating plot for {cooccur_type}')
        trace = go.Scatter(
            x = counts[cooccur_type],
            y = bias[cooccur_type],
            mode='markers',
            text=words[cooccur_type])

        print('Fitting polynomial ...')
        coeffs = np.polyfit(counts[cooccur_type],bias[cooccur_type],deg=4)
        print(coeffs)

        trace_poly = go.Scatter(
            x = np.arange(0,12,0.1),
            y = get_polypoints(np.arange(0,12,0.1),coeffs),
            mode='lines')

        layout = go.Layout(
            title = linearize(10,coeffs),
            xaxis = dict(title = 'log(X)'),
            yaxis = dict(title = 'b'),
            hovermode = 'closest')

        filename = os.path.join(
            exp_const.vis_dir,
            f'bias_vs_logX_{cooccur_type}.html')
        plotly.offline.plot(
            {'data': [trace, trace_poly],'layout': layout},
            filename=filename,
            auto_open=False)
