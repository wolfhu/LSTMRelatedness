
import numpy as np

def load_pretrained_glove_vectors(glove_file):
    """
    This loads the pretrained glove vectors found at
    http://nlp.stanford.edu/projects/glove/ into a dictionary and
    vocabulary wordlist. returns the word list and a dictionary with
    indexes of the word related to the vectors

    :param glove_file: the filepath of the pretrained glove vecotrs
    """
    index = 0
    vocabulary = {}
    glove_vectors = {} #skip information on first line
    with open(glove_file, 'r') as fin:
        for line in fin:
            items = line.replace('\r','').replace('\n','').split(' ')
            # if len(items) < 10: continue
            word = items[0]
            if word in vocabulary:
                wordindex = vocabulary[word]
            else:
                wordindex = index
                vocabulary[word] = index
                index += 1
            vect = np.array([np.float32(i) for i in items[1:] if len(i) > 1])
            glove_vectors[word] = vect

    return glove_vectors, vocabulary
