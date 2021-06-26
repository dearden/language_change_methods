import sys
import os

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
from datetime import date, timedelta
import regex as re
import json
import time
import unicodedata
import spacy
import itertools

from collections import Counter

nlp = spacy.load('en_core_web_sm')

round_down_month = lambda x: x.replace(day=1)

convert_to_bool = lambda x: True if x=="True" else False

merge_lists = lambda x: list(itertools.chain.from_iterable(x))


def log_ratio(x, y):
    ratio = x / y
    return np.log2(ratio)


def get_log_ratio(word, counts1, counts2, len1=None, len2=None):
    if len1 is None:
        len1 = sum(counts1.values())
    if len2 is None:
        len2 = sum(counts2.values())

    get_value = lambda w, d: d[w] if w in d else 0
    c1 = get_value(word, counts1)
    c2 = get_value(word, counts2)
    lr = log_ratio((c1+0.5)/len1, (c2+0.5)/len2)
    return lr


def get_log_ratios(counts1, counts2):
    out = []
    lenCorp1 = sum(counts1.values())
    lenCorp2 = sum(counts2.values())

    for word in set(list(counts1) + list(counts2)):
        get_value = lambda w, d: d[w] if w in d else 0
        c1 = get_value(word, counts1)
        c2 = get_value(word, counts2)

        # If either of the corpuses have no text, set lr to None.
        if lenCorp1 == 0 or lenCorp2 == 0:
            lr = None
        else:
            lr = log_ratio((c1+0.5)/lenCorp1, (c2+0.5)/lenCorp2)

        out.append((word, c1, c2, lr, lenCorp1, lenCorp2))
    return out


def log_likelihood(freq1, freq2, n1, n2):
    e1 = n1 * (freq1 + freq2) / (n1 + n2)
    e2 = n2 * (freq1 + freq2) / (n1 + n2)

    # This lambda performs the calculation needed for each corpus, but returns 0 if freq is 0, to avoid ln errors
    sub_function = lambda freq, e: freq * np.log(freq / e) if freq > 0 else 0
    g2 = 2 * (sub_function(freq1, e1) + sub_function(freq2, e2))
    return g2


def get_ll_and_lr(counts1, counts2):
    out = []
    lenCorp1 = sum(counts1.values())
    lenCorp2 = sum(counts2.values())

    for word in set(list(counts1) + list(counts2)):
        # Get the counts for the word in each corpus
        get_value = lambda w, d: d[w] if w in d else 0
        c1 = get_value(word, counts1)
        c2 = get_value(word, counts2)

        # If either of the corpuses have no text, set lr to None.
        if lenCorp1 == 0 or lenCorp2 == 0:
            lr = None
            ll = None
        else:
            lr = log_ratio((c1+0.5)/lenCorp1, (c2+0.5)/lenCorp2)
            ll = log_likelihood(c1, c2, lenCorp1, lenCorp2)

        out.append((word, c1, c2, lr, ll, lenCorp1, lenCorp2))
    return out


def count_tokens(texts):
    count = Counter()
    for text in texts:
        count.update(text)
    return count


def get_keywords_from_tokens(tokens, comparison, lr_threshold=1, count_threshold=10):
    tok_counts = dict(count_tokens(tokens.values))
    comp_counts = dict(count_tokens(comparison.values))

    # calculate the log ratios for the two count dictionaries.
    # This will create a dataframe  with the log-ratio for each word in the corpus w.r.t the comparison.
    log_ratios = pd.DataFrame(get_log_ratios(tok_counts, comp_counts),
                            columns=['word', 'count', 'comp_count', 'log-ratio', 'len', 'comp_len'])

    # Make the words the index.
    log_ratios.set_index("word", inplace=True)

    # If we have set a log-ratio threshold, only keep the words with an LR above that threshold.
    if lr_threshold is not None:
        out_kw = log_ratios.loc[log_ratios['count'] > count_threshold].loc[log_ratios['log-ratio'] > lr_threshold].sort_values("log-ratio", ascending=False)["log-ratio"].apply(float)
    # Otherwise, just keep all of it.
    else:
        out_kw = log_ratios

    return out_kw


# Adds key dates (provided in a pandas dataframe) to a given graph.
# The format of the dataframe must follow the template (THAT I NEED TO CLARIFY)
def add_key_dates(ax, key_dates):
    for date, event in key_dates.iterrows():
        ax.axvline(x=date, color="gray", alpha=event.transparency, zorder=1)


# Creates a generator which gives the beginning of each window in given range and step.
def dayrange(d_start, d_end, window=0, step=1):
    for n in range(0, int(((d_end - timedelta(days=window)) - d_start).days), step):
        yield d_start + timedelta(n)

# Get time windows beginning at the start and rolling a specified amount of time forward each window. (e.g. a number of days)
def get_time_windows(data, window_size, step, time_column="time", string_date=False):
    # lambda for checking a date is within two bounds
    check_in_range = lambda x, beg, end: beg <= x.date() <= end

    # Sort the values initially otherwise it won't work.
    data = data.sort_values(time_column, ascending=True)
    start_date = data.iloc[0][time_column]
    end_date = data.iloc[-1][time_column]

    for curr_day in dayrange(start_date, end_date, window=window_size, step=step):
        # create window
        win_beg = curr_day
        win_end = curr_day + timedelta(days=window_size-1)
        # get all posts for which time stamp is in window
        data_in_window = data[data[time_column].apply(check_in_range, args=(win_beg, win_end))]
        if string_date: 
            yield win_beg.strftime("%Y/%m/%d %H:%M:%S"), data_in_window
        else:
            yield win_beg, data_in_window


# Get rolling windows which move forward a number of contributions each time.
def get_data_windows(data, window_size, step, time_column="time", string_date=False):
    # Sort the values initially because otherwise it won't work.
    data = data.sort_values(time_column, ascending=True)
    # Go through the contributions with the step specified and return the specified size window.
    for i in range(0, len(data) - window_size, step):
        # Get the data items for the window.
        curr_window = data.iloc[i:i+window_size]
        # Get the start date for the window.
        if string_date:
            curr_win_date = data.iloc[i][time_column].strftime("%Y/%m/%d %H:%M:%S")
        else:
            curr_win_date = data.iloc[i][time_column]
        # Yield the date and window.
        yield curr_win_date, curr_window


def clean_text(text):
    text = text.lower()
    text = text.strip()

    # Replace quotes with standard tag.
    text = re.sub(r"\<quote.*\>", " QUOTE ", text)

    # Replace numbers with tag.
    text = re.sub(r"\d+(\.\d+)*", "NUMBER", text)

    # Replace urls with tag.
    text = re.sub(r"(?:https?://)?(?:[-\w]+\.)+[a-zA-Z]{2,9}[-\w/#~:;.?+=&%@~]*", " URL ", text)

    # Get rid of newlines.
    text = text.replace("\n", " ")

    # Honestly not sure what this achieves
    # text = re.sub(r"([\.\,\!\?\:\;]+)(?=\w)", r"\1 ", text)

    # in cases of multiple punctuation marks, keep the first
    text = re.sub(r"(\p{P})\p{P}*", r"\1 ", text)

    # # replace all punctuation with tags
    # text = re.sub(r"\p{P}+", " PUNCT ", text)

    text = unicodedata.normalize("NFKD", text)

    text = re.sub(r"[ ]+", r" ", text)

    return text


def basic_preprocessing(text):
    # Handle Quotes
    text = re.sub(r"\<quote.*?\>", "", text)
    # Handle URLS
    text = re.sub(r"(?:https?://)?(?:[-\w]+\.)+[a-zA-Z]{2,9}[-\w/#~:;.?+=&%@~]*", "", text)
    # Handle Encoding
    text = unicodedata.normalize("NFKD", text)
    # Remove repeated spaces
    text = re.sub(r"[ ]+", r" ", text)
    return text.strip()

def spacy_tokenise(text):
    doc = nlp.tokenizer(text)
    return [tok.text for tok in doc]


def tokenise(text):
    """
    Turns given text into tokens.
    """
    cleaned = clean_text(text)
    tokens = spacy_tokenise(cleaned)
    return tokens

def check_dir(dir_name):
    """
    Checks if a directory exists. Makes it if it doesn't.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        

def make_ngram_concordance(ngram, tokens, total_size=80, doc_labels=None):
    """
    Given an ngram, and a list of tokenised documents, creates the concordances

    ngram : List
        The ngram to search for in the given tokens.
    tokens : Iterable
        The tokenised documents to search (list of tokens per document).
    total_size: int
        Number of characters to show for each line of concordance
    doc_labels: Iterable
        Iterable of labels, one for each document (same length as tokens).
    """
    concordance_list = []
    tok_size = len(" ".join(ngram))
    # Sets the context size (half the full size - ngram size)
    context_size = int((total_size - tok_size) / 2)
    
    # Loop through all documents
    for doc_num, document in enumerate(tokens):
        for i in range(len(document)-(len(ngram)-1)):
            if document[i:i+len(ngram)] == ngram:
                # Gets context on left side of ngram
                left_context = " ".join(document[:i])
                left_context = " " * max(0, context_size-len(left_context)) + left_context
                left_context = left_context[-context_size:]
                # Gets context on right side of ngram
                right_context = " ".join(document[i+len(ngram):])
                right_context = right_context + " " * max(0, context_size-len(right_context))
                right_context = right_context[:context_size]
                
                # Either use the value of i as a label, or the specified labels.
                if doc_labels is not None:
                    label = doc_labels[doc_num]
                else:
                    label = doc_num
                
                out_tok = "\033[1m" + " ".join(ngram) + "\033[0m"
                concordance_list.append((label, " ".join([left_context, out_tok, right_context])))
    return concordance_list


def print_concordance(concordance_list, n_examples=25):
    """
    Prints the given concordance list
    """
    # Print the desired n examples.
    print(f"Showing {n_examples} out of {len(concordance_list)}:")
    for line in concordance_list[:n_examples]:
        print(f"{line[0]:<10}", line[1])


        
def get_ngram_example(ngram, tokens, total_size=80, n_examples=25, doc_labels=None):
    """
    Given an ngram, and a list of tokenised documents, prints concordances of this ngram.
    
    ngram : List
        The ngram to search for in the given tokens.
    tokens : Iterable
        The tokenised documents to search (list of tokens per document).
    total_size: int
        Number of characters to show for each line of concordance
    n_examples: int
        Number of concordance lines to show.
    doc_labels: Iterable
        Iterable of labels, one for each document (same length as tokens).
    """
    concordance_list = make_ngram_concordance(ngram, tokens, total_size=total_size, doc_labels=doc_labels)
    print_concordance(concordance_list, n_examples=n_examples)


# Concordancing functions
def get_example_chars(tok, tokens, total_size=80, n_examples=25, doc_labels=None):
    """
    Given an ngram, and a list of tokenised documents, prints concordances of this ngram.
    
    tok : str
        The token to search for in the given tokens.
    tokens : Iterable
        The tokenised documents to search (list of tokens per document).
    total_size: int
        Number of characters to show for each line of concordance
    n_examples: int
        Number of concordance lines to show.
    doc_labels: Iterable
        Iterable of labels, one for each document (same length as tokens).
    """
    get_ngram_example([tok], tokens, total_size=total_size, n_examples=n_examples, doc_labels=doc_labels)
    
    
def make_pos_concordance(ngram, pos, tokens, total_size=80, doc_labels=None):
    """
    Given a Part-of-Speech ngram, and a list of tokenised documents (PoS and Words), 
    creates concordances of words for this PoS ngram.
    
    ngram : List
        The PoS ngram to search for in the given tokens.
    pos : Iterable
        The PoS tagged documents to search (list of PoS tags per document).
    tokens : Iterable
        The tokenised documents to search (list of tokens per document - must correspond to PoS).
    total_size: int
        Number of characters to show for each line of concordance
    doc_labels: Iterable
        Iterable of labels, one for each document (same length as tokens).
    """
    concordance_list = []
    # Loop through all documents
    for doc_num, (pos_doc, tok_doc) in enumerate(zip(pos, tokens)):
        for i in range(len(pos_doc)-(len(ngram)-1)):
            if pos_doc[i:i+len(ngram)] == ngram:
                # Gets the current word ngram corresponding to PoS ngram
                curr_ngram = tok_doc[i:i+len(ngram)]
                tok_size = len(" ".join(curr_ngram))
                context_size = int((total_size - tok_size) / 2)
                
                # Gets context on left side of ngram
                left_context = " ".join(tok_doc[:i])
                left_context = " " * max(0, context_size-len(left_context)) + left_context
                left_context = left_context[-context_size:]
                # Gets context on right side of ngram
                right_context = " ".join(tok_doc[i+len(ngram):])
                right_context = right_context + " " * max(0, context_size-len(right_context))
                right_context = right_context[:context_size]
                
                # Either use the value of i as a label, or the specified labels.
                if doc_labels is not None:
                    label = doc_labels[doc_num]
                else:
                    label = doc_num
                
                out_tok = "\033[1m" + " ".join(curr_ngram) + "\033[0m"
                concordance_list.append((label, " ".join([left_context, out_tok, right_context])))
    return concordance_list


def get_text_example_of_pos(ngram, pos, tokens, total_size=80, n_examples=25, doc_labels=None):
    """
    Given a Part-of-Speech ngram, and a list of tokenised documents (PoS and Words), 
    prints concordances of this ngram. Matches PoS and prints corresponding words.
    
    ngram : List
        The PoS ngram to search for in the given tokens.
    pos : Iterable
        The PoS tagged documents to search (list of PoS tags per document).
    tokens : Iterable
        The tokenised documents to search (list of tokens per document - must correspond to PoS).
    total_size: int
        Number of characters to show for each line of concordance
    n_examples: int
        Number of concordance lines to show.
    doc_labels: Iterable
        Iterable of labels, one for each document (same length as tokens).
    """
    concordance_list = make_pos_concordance(ngram, pos, tokens, total_size=total_size, doc_labels=doc_labels)
    print_concordance(concordance_list, n_examples=n_examples)