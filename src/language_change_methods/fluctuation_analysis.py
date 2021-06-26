import json
import re
import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine as cosine_dist
from sklearn.metrics.pairwise import cosine_similarity
from pygam import GAM, s, f, LinearGAM
from itertools import combinations

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from language_change_methods.utility_functions import add_key_dates


cosine_sim = lambda x,y: 1 - cosine_dist(x,y)

colour_list = ["#d73027", "#4575b4", "#fdae61", "#abd9e9", "#fee090", "#f46d43", "#74add1", "#e0f3f8"]
GRAPH_BG_COL = "beige"


def fluct_anal(vectors_over_time, comparison_function):
    comparisons = dict()
    for i in range(1, vectors_over_time.shape[0]):
        win_time = vectors_over_time.index[i]
        curr = vectors_over_time.iloc[i]
        pre = vectors_over_time.iloc[i-1]
        comparisons[win_time] = comparison_function(curr, pre)

    return pd.Series(comparisons)


def comp_anal(m1, m2, comparison_function):
    comparisons = dict()
    for i in range(m1.shape[0]):
        win_time = m1.index[i]
        v1 = m1.iloc[i]
        v2 = m2.iloc[i]
        comparisons[win_time] = comparison_function(v1, v2)

    return pd.Series(comparisons)


def calc_ac1(t1, t2):
    a = (t1 & t2).sum()
    b = t1.sum()
    c = t2.sum()
    d = (~(t1 | t2)).sum()
    f = 2 * ((a + b + a + c)/(2*(a + b + c + d))) * (1 - (a + b + a + c)/(2*(a + b + c + d)))
    agree = (a + d) / (a + b + c + d)
    ac1 = (agree - f) / (1 - f)
    return ac1


def add_missing_columns(df1, df2, fill_value=0):
    """
    Updates df1 to have all the columns of df2
    """
    # Things that are in df2 but not df1
    for c in set(df2.columns) - set(df1.columns):
        df1[str(c)] = [fill_value] * len(df1)
                          
    # Sort the columns to be the same for both
    df1 = df1[sorted(df1.columns)]
    return df1


def make_dfs_comparable(df1, df2, fill_value=0):
    """
    Gives dataframes the same columns
    """    
    df1 = add_missing_columns(df1, df2, fill_value=fill_value)
    df2 = add_missing_columns(df2, df1, fill_value=fill_value)
    return df1, df2


def plot_gam(X, y, gam, dates, ax, xlabels=None, line_colour=None, area=True, dots=True, label=None):
    XX = pd.date_range(dates[0], dates[-1]).values
    
    gam.gridsearch(X, y)
    y_pred = gam.predict(XX.reshape(-1,1))
    
    ax.plot(XX, y_pred, label=label, color=line_colour, zorder=3)
    
    if area:
        intervals = gam.prediction_intervals(XX, width=.95)
        intervals[:,0]
        ax.fill_between(XX, intervals[:,0], intervals[:,1], color=line_colour, alpha=0.1, zorder=0)
    
    if dots:
        ax.plot(X, y, marker=".", linewidth=0, color=line_colour, alpha=0.75, zorder=2)
        
    plt.xticks(rotation=90)


def plot_stepped(X, y, ax, xlabels=None, line_colour=None, area=True, dots=True, label=None):
    ax.plot(dates, y, label=label, color=line_colour, zorder=3)
    
    if area:
        intervals = gam.prediction_intervals(XX, width=.95)
        intervals[:,0]
        ax.fill_between(XX, intervals[:,0], intervals[:,1], color=line_colour, alpha=0.1, zorder=0)
    
    if dots:
        ax.plot(X, y, marker=".", linewidth=0, color=line_colour, alpha=0.75, zorder=2)
        
    plt.xticks(rotation=90)


def make_similarity_over_time(m1, m2, key_dates=None):
    # Calculates the pairwise similarities and then keeps the diagonal elements 
    # (i.e. sim(list1[0], list2[0]), sim(list1[1], list2[1]), etc)
    sims = cosine_similarity(m1.values, m2.values).diagonal()

    sims = pd.Series(sims, index=m1.index)

    dates = sims.index.values

    # X = np.arange(0, len(sims), 1).reshape(-1, 1)
    X = dates.reshape(-1,1)
    # X = matrix.values[:-1, :]
    y = sims.values

    nice_dates = sims.index.to_series().apply(lambda x: x.strftime("%Y-%m-%d")).values

    clf = LinearGAM(n_splines=10)

    fig, ax = plt.subplots(figsize=(20, 10))
    # ax.set_facecolor(GRAPH_BG_COL)
    # plot_gam(X, y, clf, dates=dates, ax=ax, xlabels=nice_dates, line_colour="#8da0cb")
    ax.plot(dates, y, drawstyle="steps-post", color="#8da0cb")

    if key_dates is not None:
        add_key_dates(ax, key_dates)

    ax.grid(axis="y")
    return fig


def get_fluctuations(kw_matrix, sim_method=cosine_sim):
    all_windows = list(kw_matrix.index)

    sims = dict()
    for i in range(1, len(all_windows)):
        curr_sim = sim_method(kw_matrix.loc[all_windows[i]], kw_matrix.loc[all_windows[i-1]])
        sims[all_windows[i]] = curr_sim

    sims = pd.Series(sims)
    return sims


def make_groups_fluctuation_plot(group_matrices, group_names, key_dates=None, line_colours=None, area=True):
    fig, ax = plt.subplots(figsize=(20, 10))
    # ax.set_facecolor(GRAPH_BG_COL)
    if line_colours is None:
        line_colours = colour_list[:len(group_names)]

    for gname, colour in zip(group_names, line_colours):
        # Get the actual similarities.
        sims = get_fluctuations(group_matrices[gname], cosine_sim)

        dates = sims.index.values

        X = dates.reshape(-1,1)
        y = sims.values

        nice_dates = sims.index.to_series().apply(lambda x: x.strftime("%Y-%m-%d")).values

        clf = LinearGAM(n_splines=10)

        # plot_gam(X, y, clf, dates=dates, ax=ax, xlabels=nice_dates, label=gname, line_colour=colour, area=area)
        ax.plot(dates, y, label=gname, drawstyle="steps-post", color=colour)

    if key_dates is not None:
        add_key_dates(ax, key_dates)

    ax.grid(axis="y")

    ax.legend()
    return fig

def plot_gam_of_series(series, ax, line_colour="#8da0cb", label=None):
    dates = series.index.values

    X = dates.reshape(-1,1)
    y = series.values

    nice_dates = series.index.to_series().apply(lambda x: x.strftime("%Y-%m-%d")).values

    clf = LinearGAM(n_splines=10)
    
    plot_gam(X, y, clf, dates=dates, ax=ax, xlabels=nice_dates, line_colour=line_colour, label=label)


def plot_1_sim(m1, m2, ax, line_colour="#8da0cb", label=None):
    sims = cosine_similarity(m1.values, m2.values).diagonal()

    sims = pd.Series(sims, index=m1.index)

    plot_gam_of_series(sims, ax, line_colour, label)


def compare_similarity(matrices, mc, key_dates=None, colours=colour_list, labels=None):
    fig, ax = plt.subplots(figsize=(20, 10))

    # Set a default value of labels if one isn't provided
    if labels == None: labels = [None] * len(matrices)

    for i in range(len(matrices)):
        plot_1_sim(matrices[i], mc, ax, colours[i], label=labels[i])
    
    if key_dates is not None:
        for date, event in key_dates.iterrows():
            ax.axvline(x=date, color="gray", alpha=event.transparency, zorder=1)
            
    for xtick in ax.xaxis.get_ticklabels():
        xtick.set_size(20)

    for ytick in ax.yaxis.get_ticklabels():
        ytick.set_size(20)
        
    ax.set_xlabel("Time", fontsize=25)
    ax.set_ylabel("Cosine Similarity", fontsize=25)

    ax.grid(axis="y")
    ax.legend(fontsize=30)
    return fig


def plot_fluctuation_of_groups(group_w_freqs, out_fp, key_dates=None):
    interest_groups = list(group_w_freqs.keys())
    colours = colour_list[:len(interest_groups)]

    for gname, colour in zip(interest_groups, colours):
        fig = make_groups_fluctuation_plot(group_w_freqs, [gname], key_dates=key_dates, line_colours=[colour])
        fig.savefig(os.path.join(out_fp, "{0}-fluctuation.pdf".format(gname)))

    fig = make_groups_fluctuation_plot(group_w_freqs, interest_groups, key_dates=key_dates, area=False, line_colours=colours)
    fig.savefig(os.path.join(out_fp, "all-groups-fluctuation.pdf"))


def plot_similarity_of_groups(group_w_freqs, out_fp, key_dates=None):
    interest_groups = list(group_w_freqs.keys())

    for combo in list(combinations(interest_groups, 2)):
        m1 = group_w_freqs[combo[0]]
        m2 = group_w_freqs[combo[1]]
        fig = make_similarity_over_time(m1, m2, key_dates=key_dates)
        fig.savefig(os.path.join(out_fp, "{0}-{1}-similarity.pdf".format(combo[0], combo[1])))


def plot_similarity_of_groups_to_comparison(group_w_freqs, comparison, comparison_name, out_fp, key_dates=None):
    interest_groups = list(group_w_freqs.keys())

    for combo in list(combinations(interest_groups, 2)):
        m1 = group_w_freqs[combo[0]]
        m2 = group_w_freqs[combo[1]]
        fig = compare_similarity([m1, m2], comparison, key_dates=key_dates, labels=list(combo))
        fig.savefig(os.path.join(out_fp, "{0}-{1}-compared-to-{2}.pdf".format(combo[0], combo[1], comparison_name)))

    fig = compare_similarity(list(group_w_freqs.values()), comparison, key_dates=key_dates, labels=interest_groups)
    fig.savefig(os.path.join(out_fp, "all-compared-to-{0}.pdf".format(comparison_name)))