import os
import csv

import utils.io as io
from data.semeval_2018_10.constants import SemEval201810Constants


def read_txt(txt_file):
    with open(txt_file,'r') as file:
        reader = csv.reader(file)
        data = []
        for row in reader:
            data.append(row)
    return data


def main():
    const = SemEval201810Constants()
    io.mkdir_if_not_exists(const.proc_dir)
    
    subset_txt_file = {
        'train': const.train_txt,
        'val': const.val_txt,
        'test': const.test_txt,
        'truth': const.truth_txt
    }
    
    for subset, txt_file in subset_txt_file.items():
        print(f'Converting {subset}.txt file to json ...')
        data = read_txt(txt_file)
        print(f'Number of samples: {len(data)}')
        io.dump_json_object(
            data,
            os.path.join(const.proc_dir,f'{subset}.json'))


if __name__=='__main__':
    main()

