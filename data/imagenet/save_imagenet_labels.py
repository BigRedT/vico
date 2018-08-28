import os
import pickle
import urllib.request as urlrequest

import utils.io as io


def main():
    url = 'https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/' + \
        'raw/d133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee/' + \
        'imagenet1000_clsid_to_human.pkl'
    outdir = os.path.join(
        os.getcwd(),
        'symlinks/data/imagenet/proc')
    io.mkdir_if_not_exists(outdir,recursive=True)
    labels_json = os.path.join(outdir,'labels.json')
    labels_dict = pickle.load(urlrequest.urlopen(url))
    labels = []
    for i in range(len(labels_dict)):
        labels.append(labels_dict[i])
    io.dump_json_object(labels,labels_json)


if __name__=='__main__':
    main()