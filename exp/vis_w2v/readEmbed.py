# Script to obtain the vocabulary from the embeddings file
import sys
import time
import os

# Main script doing the processing
def main(embedPath):
    #------------------------------------------------------------
    #print 'Reading embeddings from %s and saving vocab...\n' % embedPath,
    startTime = time.time();

    print('Here')
    with open(embedPath, 'r', encoding='latin') as fileId:
        print('Here')
        # Read only the word, ignore feature vector
        lines = [line.split(' ', 1) for line in fileId.readlines()]; #[0]
        import pdb; pdb.set_trace()

    import pdb; pdb.set_trace()
    # Header has vocab size, double check for sanity
    vocabSize = int(lines.pop(0));
    assert vocabSize == len(lines), 'Vocab size and number of words dont match!'

    # Split the embedPath at the last dot and replace it with _vocab.txt 
    vocabPath = embedPath.rsplit('.', 1)[0] + '_vocab.txt';

    # Write the words in the file
    with open(vocabPath, 'w') as fileId:
        for word in lines:
            fileId.write(word + '\n');

    #print '...done in %f seconds' % (time.time() - startTime,);
    #------------------------------------------------------------

#######################################################################
if __name__ == '__main__':
    embedPath = sys.argv[1];

    if len(sys.argv) > 1:
        embedPath = sys.argv[1];
    else:
        print('Wrong usage: No embedding path specified!\n')
        print('Usage: python getVocabFromEmbedddings.py <path-to-embeddings>')

    if not os.path.exists(embedPath):
        print('File at %s does not exist!')
    else:
        main(embedPath);