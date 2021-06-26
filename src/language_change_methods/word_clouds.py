import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
from wordcloud import WordCloud
import math
import os
from matplotlib_venn import venn2, venn3
from matplotlib_venn_wordcloud import venn2_wordcloud, venn3_wordcloud


g_colours = ['#1b9e77', '#d95f02', '#7570b3']


def make_wordcloud(keywords):
    cloud = WordCloud(width = 800, height = 800,
                      background_color ='white',
                      min_font_size = 10)

    cloud.fit_words(keywords)
    return cloud

def draw_wordcloud(keywords, out_fp=None, size=(50, 50)):
    cloud = make_wordcloud(keywords)

    # plot the WordCloud image
    fig, ax = plt.subplots(figsize=size, facecolor = None)

    ax.imshow(cloud)
    ax.axis("off")
    plt.tight_layout(pad = 0)

    if out_fp is not None:
        fig.savefig(out_fp)
    return fig

def create_multi_word_cloud(tables, names, out_dir):
    clouds = {n: make_wordcloud(t) for n, t in zip(names, tables)}

    # plot the WordCloud image
    fig = plt.figure(figsize = (100, 100), facecolor = None)

    rows=math.ceil(len(names)/2)
    columns=math.ceil(len(names)/2)

    if rows == 1 and columns == 1:
        columns = 2

    for i in range(1, len(tables) + 1):
        ax = fig.add_subplot(rows, columns, i)
        ax.set_title(names[i-1], size=200)
        ax.imshow(clouds[names[i-1]])

    plt.tight_layout(pad = 10)

    if out_dir is not None:
        out_fp = os.path.join(out_dir, "multi-wordcloud.pdf")
        fig.savefig(out_fp)
    return fig


def get_overlap(g1, g2):
    g1_exclusive = g1[~g1.index.isin(g2.index)]

    g2_exclusive = g2[~g2.index.isin(g1.index)]

    overlap = g1[g1.index.intersection(g2.index)]

    return g1_exclusive, overlap, g2_exclusive

def get_average_lr(word, s1, s2):
    return (s1[word] + s2[word]) / 2


def draw_2way_venn_wordcloud(kws, names, colours=g_colours, out_dir=None):
    g1 = kws[0]
    g2 = kws[1]

    exg1, overlap, exg2 = get_overlap(g1, g2)

    avgd_overlap = pd.Series(overlap.index.to_series().apply(lambda x: get_average_lr(x, g1, g2)).values, index=overlap.index)

    word_count_dict = pd.concat([exg1.apply(abs), exg2.apply(abs), avgd_overlap.apply(abs)]).to_dict()

    fig, ax = plt.subplots(figsize=(75, 60))

    venn = venn2_wordcloud([set(g1.index), set(g2.index)],
                    word_to_frequency=word_count_dict, ax=ax, set_labels=names, set_colors=colours[:2])

    for text in venn.set_labels:
        text.set_fontsize(80)

    ax.set_title("2-way Venn Wordcloud for Groups:\n- {0}\n- {1}".format(names[0], names[1]), fontsize=100)


    if out_dir is not None:
        fig.savefig(os.path.join(out_dir, "{0}-{1}-venn.pdf".format(names[0], names[1])))

    return fig


def get_three_way_overlap(s1, s2, s3):
    out = dict()
    all_series = [s1, s2, s3]
    for s in all_series:
        for word in s.index:
            if word in out:
                continue

            matches = []
            for s2 in all_series:
                if word in s2:
                    matches.append(s2[word])
            val = sum(matches) / len(matches)

            out[word] = val

    return out


def draw_3way_venn_wordcloud(kws, names, colours=g_colours, out_dir=None):
    g1 = kws[0]
    g2 = kws[1]
    g3 = kws[2]

    threeway_word_count_dict = get_three_way_overlap(g1, g2, g3)

    fig, ax = plt.subplots(figsize=(75, 60))

    venn = venn3_wordcloud([set(g1.index), set(g2.index), set(g3.index)], set_labels=names,
                    word_to_frequency=threeway_word_count_dict, ax=ax, set_colors=g_colours)

    for text in venn.set_labels:
        text.set_fontsize(80)

    ax.set_title("3-way Venn Wordcloud for Groups:\n- {0}\n- {1}\n- {2}".format(names[0], names[1], names[2]), fontsize=100)

    if out_dir is not None:
        fig.savefig(os.path.join(out_dir, "{0}-{1}-{2}-venn.pdf".format(names[0], names[1], names[2])))

    return fig
    