import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np
import json
import regex as re

from gensim.models import Word2Vec

from language_change_methods.features import function_words


def get_top_vocab_and_vectors(model, n=10000):
    """
    Gets the top n words from the model's vocabulary and the vectors of these words.
    """
    top_vocab = sorted(model.wv.vocab.keys(), key=lambda x: model.wv.vocab[x].count, reverse=True)[:n]
    top_vectors = np.array([model.wv[t] for t in top_vocab])
    return top_vocab, top_vectors


def neighbors(query : str,
              embs: np.ndarray,
              vocab: list,
              K : int = 3) -> list:
    sims = np.dot(embs[vocab.index(query),],embs.T)
    output = []
    for sim_idx in sims.argsort()[::-1][1:(1+K)]:
        if sims[sim_idx] > 0:
            output.append(vocab[sim_idx])
    return output


def get_most_changey_words_with_models(model1, model2, n=100, k=1000, top_n=None):
    nn_scores = []
    
    top_vocab = sorted(model1.wv.vocab.keys(), key=lambda x: model1.wv.vocab[x].count, reverse=True)[:top_n]
    
    vocab1 = model1.wv.vocab
    vocab2 = model2.wv.vocab
    # Loop through all the words in the vocab
    for w in vocab1:
        if (w not in function_words 
                and w in vocab1 
                and w in vocab2 
                and vocab1[w].count > n 
                and vocab2[w].count > n 
                and w in top_vocab):
            neighbours1 = set([x[0] for x in model1.wv.most_similar(w, topn=k)])
            neighbours2 = set([x[0] for x in model2.wv.most_similar(w, topn=k)])
            nn_scores.append((len(neighbours1.intersection(neighbours2)), w))
            
    nn_scores_sorted = sorted(nn_scores)
    return nn_scores_sorted


def get_most_changey_words_with_vectors(vocab1, vocab2, vectors1, vectors2, n=20, k=1000):
    nn_scores = []
    # Loop through all the words in the vocab
    for w in vocab1:
        if w not in function_words and w in vocab1 and w in vocab2:
            neighbours1 = set(neighbors(w, vectors1, vocab1, k))
            neighbours2 = set(neighbors(w, vectors2, vocab2, k))
            nn_scores.append((len(neighbours1.intersection(neighbours2)), w))
            
    nn_scores_sorted = sorted(nn_scores)
    return nn_scores_sorted


def neighbours_over_time(search_term, time_models, top_n=10000):
    for window, curr_model in time_models.items():
        curr_vocab, curr_vectors = get_top_vocab_and_vectors(curr_model, top_n)
        print(window)
        if search_term in curr_vocab:
            print(neighbors(search_term, curr_vectors, curr_vocab, 12))


def get_changiest_words_per_window(time_models, top_n=10000, k=1000):
    out_dic = dict()
    windows = list(time_models.keys())
    for i in range(1, len(windows)):
        model_1 = time_models[windows[i-1]]
        model_2 = time_models[windows[i]]

        vocab_1, vectors_1 = get_top_vocab_and_vectors(model_1, top_n)
        vocab_2, vectors_2 = get_top_vocab_and_vectors(model_2, top_n)

        out_dic[windows[i]] = get_most_changey_words_with_vectors(vocab_1, vocab_2, vectors_1, vectors_2, k=k)

    return out_dic


def print_changiest_over_time(changiest_words_per_window, time_models, min_freq=0, remove_punc=True, remove_func=True, word_list=None):
    for window, changey_words in changiest_words_per_window.items():
        check_freq = lambda w, m: m.wv.vocab[w].count > min_freq
        queries = [w[1] for w in changey_words if check_freq(w[1],time_models[window])]
        if remove_punc:
            check_punc = lambda w: True if re.fullmatch(r"\p{P}+", w) else False
            queries = [w for w in queries if not check_punc(w)]

        if remove_func:
            queries = [w for w in queries if w not in function_words]

        if word_list is not None:
            queries = [w for w in queries if w in word_list]

        queries = queries[:20]

        print(window)
        print("{:20} {:20} {:20} {:20} {:20}".format(*queries[:5]))
        print("{:20} {:20} {:20} {:20} {:20}".format(*queries[5:10]))
        print("{:20} {:20} {:20} {:20} {:20}".format(*queries[10:15]))
        print("{:20} {:20} {:20} {:20} {:20}".format(*queries[15:20]))
        print("-----------------------------")