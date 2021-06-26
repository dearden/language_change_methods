import nltk
import spacy
from collections import Counter
from textstat import flesch_reading_ease
import regex as re
from nltk import ngrams as make_ngrams
from utility_functions import get_ll_and_lr
import pandas as pd
from typing import Iterable
from sklearn.feature_extraction import DictVectorizer


with open("C:/Users/Eddie/Documents/language_change_methods/word_lists/function_words.txt") as func_file:
    function_words = [line.strip().lower() for line in func_file.readlines()]


def get_wordcounts_multiple_texts(toks):
    count = Counter()
    for text in toks:
        count.update(text)
    return count


def get_normalised_wordcounts(toks, words_to_keep=None):
    # Get all wordcounts for window
    curr_counts = get_wordcounts_multiple_texts(toks)
    
    # Only keep the word counts for top words
    if words_to_keep is not None:
        curr_counts = {word: count for word, count in curr_counts.items() if word in words_to_keep}

    # Normalise the counts
    num_words = toks.apply(len).sum()
    curr_counts = {word: count/num_words for word, count in curr_counts.items()}
    return curr_counts


def get_binary_wordcounts(toks, words_to_keep=None):
    # Get all wordcounts for window
    curr_counts = get_wordcounts_multiple_texts(toks)
    curr_counts = {word: 1 for word in curr_counts.keys()}
    # Get keep only the words to keep
    if words_to_keep is not None:
        curr_counts = {word: count for word, count in curr_counts.items() if word in words_to_keep}
    return curr_counts


def get_tok_counts(toks, words_to_keep=None, binary=False):
    """
    Given a list of tokens, does a count, and only keeps words in the optionally provided words_to_keep.
    @toks: an iterator/list of tokens making up a text.
    @words_to_keep: an iterator/list of words to consider in the Count.
    """
    tok_counts = Counter(toks)
    if binary:
        tok_counts = {tok: 1 for tok in tok_counts.keys()}
    if words_to_keep is not None:
        tok_counts = Counter({w: c for w, c in tok_counts.items() if w in words_to_keep})
    return tok_counts


def get_ttr(toks):
    """
    Given a list of tokens, return the type token ratio (# unique / # tokens)
    @toks: a list of tokens.
    """
    unique = set(toks)
    return len(unique) / len(toks)


def get_complexity(text):
    """
    Define complexity as 1 - (Flesch Reading Ease / 121.22)
    This will usually be between 0 and 1 (0 being simple and 1 being complex), but can exceed 1 in special circumstances, with no upper limit.
    """
    reading_ease = flesch_reading_ease(text)
    readability = reading_ease / 121.22
    return 1 - readability


def count_regex_occurances(reg, text):
    """
    Counts the number of a given regex in a given text.
    This can be used, for example, for finding the counts of certain patterns (e.g. quotes or URLs).
    """
    return len(list(re.findall(reg, text)))


def ngram_okay(phrase):
    """
    Checks if an N-gram is okay.
    Primarily, this means seeing if it includes punctuation or white space.
    """
    okay = True
    for phrase_part in phrase.split("_"):
        okay = okay and not re.match(r"[\p{P}|\p{S}|\p{N}]+", phrase_part)
    return okay


def get_ngram_counts(toks: Iterable, n: int, join_char: str = "_"):
    """
    Gets the ngram counts for a given value of N and a given list of tokenised documents.
    @param toks: A dict-like structure of tokenised documents.
    @param n: The value of n for finding ngrams.
    """
    ngrams = pd.Series({i: list(make_ngrams(post, n)) for i, post in toks.items()})
    ngrams = ngrams.apply(lambda post: [join_char.join(words) for words in post])

    counts = get_wordcounts_multiple_texts(ngrams)
    counts = Counter({n: c for n, c in counts.items() if ngram_okay(n)})
    
    return counts


# Function for finding the LL and LR for the ngrams of a given sequence of tokens
def get_ngram_lr_and_ll(toks1, toks2, n, join_char="_"):
    """
    For a given value of n, calculates the log-likelihood and log-ratio between two different corpora.
    @param toks1: A dict-like structure of tokenised documents.
    @param toks2: A dict-like structure of tokenised documents.
    @param n: The value of n for finding ngrams.
    """
    counts1 = get_ngram_counts(toks1, n, join_char=join_char)
    counts2 = get_ngram_counts(toks2, n, join_char=join_char)

    key_ngrams = get_ll_and_lr(counts1, counts2)
    key_ngrams = pd.DataFrame(key_ngrams, columns=["ngram", "freq1", "freq2", "LR", "LL", "Len_Corpus1", "Len_Corpus2"]).set_index("ngram")
    
    return key_ngrams


def combine_counts(counts):
    total_counts = Counter()
    for row in counts:
        total_counts = total_counts + Counter(row)
    return [total_counts]


def make_feature_matrix(feat_counts, sparse):
    v = DictVectorizer(sparse=sparse)
    feats = v.fit_transform(feat_counts.values())
    feat_names = v.get_feature_names()
    return feats, feat_names


def get_top_n_toks(toks, n):
    counts = Counter()
    for row in toks:
        counts.update(row)
    return [x[0] for x in counts.most_common(n)]
