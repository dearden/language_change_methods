from typing import Iterable
import pandas as pd
import numpy as np
import itertools
import regex as re

from datetime import datetime
from collections import Counter
from typing import Iterable, Callable

from language_change_methods.features import function_words
from language_change_methods.utility_functions import merge_lists, get_data_windows


def calc_ac1(t1, t2):
    """
    Calculate AC1 agreement statistic for comparing lists of collocates
    """
    a = (t1 & t2).sum()
    b = t1.sum() - a
    c = t2.sum() - a
    d = (~(t1 | t2)).sum()
    f = 2 * ((a + b + a + c)/(2*(a + b + c + d))) * (1 - (a + b + a + c)/(2*(a + b + c + d)))
    agree = (a + d) / (a + b + c + d)
    ac1 = (agree - f) / (1 - f)
    return ac1


def coll_mi(n_toks, node_freq, coll_freq, node_coll_freq):
    """
    Mutual information association measure for collocates
    """
    return np.log2(node_coll_freq / ((node_freq * coll_freq) / n_toks))

def coll_lr(n_toks, node_freq, coll_freq, node_coll_freq):
    """
    Log-ratio association measure for collocates
    """
    return np.log2((node_coll_freq * (n_toks - node_freq)) / ((coll_freq - node_coll_freq) * node_freq))


def get_collocate_window(doc, idx):
    if idx < 5:
        before = doc[:idx]
    else:
        before = doc[idx-5:idx]
    if idx > len(doc) - 5:
        after = doc[idx+1:]
    else:
        after = doc[idx+1:idx+6]
    return list(before) + list(after)


def count_document_collocates(search_tok, doc):
    collocates = []
    doc = np.array(doc)
    search_indices = np.where(doc == search_tok)[0]
    for idx in search_indices:
        collocates += get_collocate_window(doc, idx)
    return Counter(collocates)


def count_collocates(search_tok, toks, window_size=5):
    collocates = Counter()
    for doc in toks:
        collocates.update(count_document_collocates(search_tok, doc))
    return collocates


def tok_okay(tok):
    okay = True
    okay = okay and not re.match(r"[\p{P}|\p{S}|\p{N}]+", tok)
    okay = okay and tok.lower() not in function_words
    return okay


def find_key_collocates(query: str, toks: Iterable, measure: Callable, association_threshold: float, 
                        min_freq: int, window_size:int=5, remove_stops:bool=False) -> pd.Series:
    """ Find key collocates for a given word in provided documents.

    Given a word and a list of tokenised documents, find the key collocates
    using the given association measure, threshold, and minimum frequency.

    query : str
        The word to find the collocates for.
    toks : Iterable
        The tokenised documents to search (list of tokens per document).
    measure : function
        The association measure to use.
    association_threshold : float
        The threshold to cutoff "key" collocates.
    min_freq : int
        The minimum frequency to consider for collocates.
    window_size: int
        The window size used for the collocates (default 5 words either side)
    """
    # Get counts
    counts = Counter(merge_lists(toks))
    
    # Get all collocates
    colls = count_collocates(query, toks, window_size=window_size)

    # Convert to a pandas series for easier processing
    colls = pd.Series(colls)

    if remove_stops:
        # Remove all stop words (this is optional)
        colls = colls[colls.index.to_series().apply(tok_okay)]

    if min_freq is not None:
        # Filter out collocates that are not common enough
        colls = colls[colls > min_freq]

    if measure is not None:
        # Calculate the association measure for each collocate
        calc_measure = lambda x: measure(sum(counts.values()), counts[query], counts[x], colls[x])
        scores = colls.index.to_series().apply(calc_measure)
        if association_threshold is not None:
            scores = scores[scores > association_threshold]
    else:
        scores = colls.apply(lambda x: None)

    out = pd.DataFrame(scores, columns=["score"])
    out["coll_freq"] = colls.loc[scores.index]
    out["raw_freq"] = out.index.to_series().apply(lambda x: counts[x])
    return out


def get_collocates_over_time(query, contributions, tokens, win_size, win_step, 
                             measure=coll_mi, threshold=3, min_freq=5, coll_win_size=5, 
                             group_idxs=None, time_column="time", include_counts=False, remove_stops=False):
    coll_over_time = dict()
    for win_time, win_posts in get_data_windows(contributions, win_size, win_step, time_column=time_column):
        if group_idxs is not None:
            win_posts = win_posts[win_posts.index.isin(group_idxs)]
        
        # Get the tokens for this window
        win_toks = tokens[tokens.index.isin(win_posts.index)]

        # Get key collocates for the current window
        colls = find_key_collocates(query, win_toks, measure=measure, association_threshold=threshold, 
                                    min_freq=min_freq, window_size=coll_win_size, remove_stops=remove_stops)
        
        # Add this to the dict
        if not include_counts:
            coll_over_time[win_time] = colls["score"]
        else:
            coll_over_time[win_time] = colls
        
    coll_over_time = pd.Series(coll_over_time)
    return coll_over_time