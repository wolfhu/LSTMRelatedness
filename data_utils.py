
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
            glove_vectors[wordindex] = vect

    return glove_vectors, vocabulary

def convert_sentence_to_glove_vectors(sentence, vocab, glove_vectors, vector_size=300):
    # TODO: tokenize better
    word_vectors = []
    for word in sentence.split(" "):
        word_vectors.append(convert_word_to_glove(word, vocab, glove_vectors, vector_size))
    return np.array(word_vectors)

def convert_word_to_glove(word, vocab, glove_vectors, vector_size = 300):
    if word in vocab:
        return glove_vectors[vocab[word]]
    else:
        index = len(vocab) + 1
        vocab[word] = index
        zeroes = np.zeros(vector_size)
        glove_vectors[index] = zeroes
        return zeroes

def load_sick_data(sick_path, vocab, glove_vectors, vector_size = 300):
    #TODO: this can probably be a lot more pythonic and efficient
    l_sentences = []
    r_sentences = []
    relatedness = []
    with open(sick_path) as f:
        for line in f.readlines():
            cols = line.split('\t')
            # l_sentence =
