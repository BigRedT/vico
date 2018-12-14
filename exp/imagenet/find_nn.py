import numpy as np

import utils.io as io
from utils.html_writer import HtmlWriter


def main(exp_const,data_const):
    print('Loading reps ...')
    reps = np.load(data_const.reps_npy)
    num_imgs_per_class = np.load(data_const.num_imgs_per_class_npy)
    wnid_to_idx = io.load_json_object(data_const.wnid_to_idx_json)
    wnid_to_words = io.load_json_object(data_const.wnid_to_words_json)

    print('Select wnids with high frequency ...')
    select_wnids = [wnid for wnid,idx in wnid_to_idx.items() \
        if num_imgs_per_class[idx] > exp_const.min_freq]
    select_wnid_idxs = [wnid_to_idx[wnid] for wnid in select_wnids]
    select_reps = reps[select_wnid_idxs]
    wnid_to_select_idx = {wnid:i for i,wnid in enumerate(select_wnids)}

    print('Computing similarity ...')
    # reps: num_classes x dim
    sim = np.matmul(select_reps,np.transpose(select_reps))
    nn_idx = np.argsort(sim,1)[:,-2:-exp_const.num_nbrs-2:-1]
    nbrs = {}
    for i in range(sim.shape[0]):
        query_wnid = select_wnids[i]
        nbrs[query_wnid] = {
            'wnids': [select_wnids[k] for k in nn_idx[i]],
            'sim': [sim[i,k] for k in nn_idx[i]]
        }

    html_writer = HtmlWriter(data_const.knn_html)
    col_dict = {
        0: 'Query',
        1: 'Nbrs'
    }
    html_writer.add_element(col_dict)
    for query_wnid in sorted(nbrs.keys()):
        nbr_str = ''
        for wnid,score in zip(nbrs[query_wnid]['wnids'],nbrs[query_wnid]['sim']):
            nbr_str += '{}({:.2f})'.format(wnid_to_words[wnid][0],score)
            nbr_str += '&nbsp;'*4

        col_dict = {
            0: wnid_to_words[query_wnid][0],
            1: nbr_str
        }
        html_writer.add_element(col_dict)

    html_writer.close()

