import nltk
from nltk.stem import WordNetLemmatizer

class Lemmatizer():
    def __init__(self):
        self.lemmatize_ = WordNetLemmatizer().lemmatize
        self.pos_tag_ = nltk.pos_tag
        self.word_to_lemma = {}

    def lemmatize(self,word):
        if word in self.word_to_lemma:
            return self.word_to_lemma[word]
        
        pos_tag = self.get_pos(word)
        if pos_tag is None:
            self.word_to_lemma[word] = word
            return word
        
        lemma = self.lemmatize_(word,pos=pos_tag)
        self.word_to_lemma[word] = lemma

        return lemma

    def get_pos(self,word):
        pos_tag = self.pos_tag_([word])[0][1]
        if pos_tag in ['NN','NNS','NNP','NNPS']:
            return 'n'
        elif pos_tag in ['JJ','JJR','JJS']:
            return 'a'
        elif pos_tag in ['VB','VBZ','VBG','VBD','VBN','VBP']:
            return 'v'
        else:
            return None