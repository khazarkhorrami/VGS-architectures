
"""
"""

import scipy.spatial as ss
from scipy import spatial
import numpy
from sent2vec.vectorizer import Vectorizer

vectorizer = Vectorizer()
#vectorizer = Vectorizer( pretrained_weights='bert-base-uncased')


#..............................................................................
#            use this function on the new version of sent2ve                  #
#..............................................................................

# def find_similar_pairs (list_of_sentences):
#     vectors = vectorizer.vectors
#     vectors = vectorizer.run(list_of_sentences)
#     vectors = numpy.array(vectors)
#     #dist = spatial.distance.cosine(vectors[0], vectors[1])
#     dist = ss.distance.cdist( vectors,vectors ,  'cosine')
#     best = {}
#     best['best_pairs'] = []
#     best['similarity'] = []
#     for row in dist:

#         print(row)
#         best_similarity = numpy.sort(row)[1]
#         best_pair = numpy.argsort(row)[1]
#         best['similarity'].append(best_similarity)
#         best['best_pairs'].append(best_pair)      
#     return best    

# list_of_sentences = ['a blue sky','A man is passing by car', 'educational courses are there ready to be tought at this semester','I love dogs']
# best = find_similar_pairs(list_of_sentences)



#..............................................................................
#            use this function on the old version of sent2ve                  #
#..............................................................................


def bert_similarity (list_of_sentences):

    vectorizer.bert(list_of_sentences)

    vectors_bert = vectorizer.vectors
    dist = ss.distance.cdist( vectors_bert, vectors_bert ,  'cosine')
    #max_similarities = 1 - spatial.distance.cosine(vectors_bert[0],vectors_bert[1])
    return dist

def find_similar_pairs (list_of_sentences):
    dist = bert_similarity (list_of_sentences)
    # vectors = vectorizer.vectors
    # vectors = vectorizer.bert(list_of_sentences[0],list_of_sentences[1])
    # vectors = numpy.array(vectors)
    #dist = spatial.distance.cosine(vectors[0], vectors[1])
    #dist = ss.distance.cdist( vectors,vectors ,  'cosine')
    best = {}
    best['best_pairs'] = []
    best['similarity'] = []
    for row in dist:

        #print(row)
        best_similarity = numpy.sort(row)[1]
        best_pair = numpy.argsort(row)[1]
        best['similarity'].append(best_similarity)
        best['best_pairs'].append(best_pair)      
    return best    


