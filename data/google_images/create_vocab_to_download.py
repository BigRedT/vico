import os
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tag.stanford import StanfordPOSTagger
from tqdm import tqdm
import spacy

import utils.io as io


def main():
    vocab_json = os.path.join(
        os.getcwd(),
        'symlinks/data/glove/proc/glove_6B_300d_word_to_idx.json')
    vocab = io.load_json_object(vocab_json)

    r = re.compile('^[a-zA-Z]{1,}$')
    filtered_words = list(filter(r.match,vocab.keys()))
    filtered_vocab = {w:vocab[w] for w in filtered_words}
    print('Original vocab size: ' + str(len(vocab)))
    print('Vocab size with character words: ' + str(len(filtered_vocab)))

    # nlp = spacy.load('en_core_web_lg')
    # word_to_token = {}
    # for word in tqdm(filtered_words):
    #     word_to_token[word] = nlp(word)[0]

    #nltk.download('wordnet')
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatized_vocab = {}
    word_to_lemma = {}
    for word,freq in tqdm(filtered_vocab.items()):
        lemma = wordnet_lemmatizer.lemmatize(word)
        word_to_lemma[word] = lemma
        if lemma not in lemmatized_vocab:
            lemmatized_vocab[lemma] = 0
        lemmatized_vocab[lemma] += freq
    print('Lemmatized vocab size: ' + str(len(lemmatized_vocab)))

    #nltk.download('averaged_perceptron_tagger')
    nouns = 0
    adjectives = 0
    verbs = 0
    nn = set()
    jj = set()
    vb = set()
    rest = set()
    for word in tqdm(list(lemmatized_vocab.keys())):
        tag = nltk.pos_tag([word])[0][1]
        if tag=='NN':
            nouns += 1
            nn.add(word)
        elif tag=='JJ':
            adjectives += 1
            jj.add(word)
        elif tag=='VB':
            verbs += 1
            vb.add(word)
        else:
            rest.add(word)
            lemmatized_vocab.pop(word)
    print('Vocab size with nouns, adjectives and verbs: ' + str(len(lemmatized_vocab)))
    print(f'Nouns: {str(nouns)}')
    print(f'Adjectives: {str(adjectives)}')
    print(f'Verbs: {str(verbs)}')
    
    lemmatized_vocab_json = os.path.join(
        os.getcwd(),
        'symlinks/data/glove/proc/glove_6B_300d_lemmatized_word_to_idx.json')
    io.dump_json_object(lemmatized_vocab,lemmatized_vocab_json)
    word_to_lemma_json = os.path.join(
        os.getcwd(),
        'symlinks/data/glove/proc/glove_6B_300d_word_to_lemma.json')
    io.dump_json_object(word_to_lemma,word_to_lemma_json)

    
if __name__=='__main__':
    main()