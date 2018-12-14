import os
import re

import utils.io as io
from .constants import SquadConstants

def get_words_from_str(string):
    string = re.sub(r'[^A-Za-z0-9 ]','',string)
    return [word.lower() for word in string.split(' ')]

def create_subset_vocab(data_json):
    data = io.load_json_object(data_json)['data']
    vocab = {
        'question': set(),
        'answer': set(),
        'context': set(),
        'total': set(),
    }
    for i in range(len(data)):
        for para_data in data[i]['paragraphs']:
            context = para_data['context']
            context_words = get_words_from_str(context)
            vocab['context'].update(context_words)
            for qa in para_data['qas']:
                q_words = get_words_from_str(qa['question'])
                vocab['question'].update(q_words)
                for a in qa['answers']:
                    a_words = get_words_from_str(a['text'])
                    vocab['answer'].update(a_words)

    for k in ['question','answer','context']:
        vocab['total'].update(vocab[k])

    vocab = {k:list(v) for k,v in vocab.items()}

    return vocab


if __name__=='__main__':
    const = SquadConstants()
    dev_vocab = create_subset_vocab(const.dev_json)
    dev_vocab_json = os.path.join(const.raw_dir,'dev_vocab.json')
    io.dump_json_object(dev_vocab,dev_vocab_json)
    print('-'*80)
    print('Dev vocab size')
    print('-'*80)
    for k,v in dev_vocab.items():
        print(f'{k} : {len(v)}')

    train_vocab = create_subset_vocab(const.train_json)
    train_vocab_json = os.path.join(const.raw_dir,'train_vocab.json')
    io.dump_json_object(train_vocab,train_vocab_json)
    print('-'*80)
    print('Train vocab size')
    print('-'*80)
    for k,v in train_vocab.items():
        print(f'{k} : {len(v)}')