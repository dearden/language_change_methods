import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import math
import os
import sys
from scipy.cluster.hierarchy import fcluster, dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from itertools import combinations
from colour import Color
from utility_functions import get_log_ratio


# This method calculates cosine distance between two vectors.
from scipy.spatial.distance import cosine as cosine_dist
# This method simply inverts it to get similarity.
cosine_sim = lambda x,y: 1 - cosine_dist(x,y)


colour_list = ["#d73027", "#4575b4", "#fdae61", "#abd9e9", "#fee090", "#f46d43", "#74add1", "#e0f3f8"]


class VNC():
    # Lambda that finds the midpoint of the dates in the clusters' index.
    find_midpoint = lambda x: pd.to_datetime(x.index.to_series().astype(np.int64).mean())

    def __init__(self, matrix, sim_method, label="", print_output=False):
        super().__init__()
        self.sim_method = sim_method
        self.label = label
        self.matrix = matrix
        self.d_list = self.vnc(matrix.copy(), print_output)


    # Formats the keyword matrix so there is one DataFrame per row. These will be the clusters.
    def make_initial_clusters(self, w_matrix):
        # Creates a list of initial clusters.
        elements = []
        for idx, row in w_matrix.iterrows():
            # Add a dictionary to the dataframe.
            # It has to be wrapped in a list so pandas doesn't mistake it as an iterable.
            elements.append([pd.Series({idx: row})])
        # Creates a new matrix of clusters.
        clusters = pd.DataFrame(elements, columns=["Elements"], index=w_matrix.index)
        clusters["Index_Val"] = np.arange(0, len(clusters))
        #     clusters["Elements"] = clusters["Elements"].apply(lambda x: [x])
        return clusters


    def calculate_inter_cluster_similarity(self, c1, c2):
        sims = []
        # Loop through the first cluster.
        for element1 in c1:
            # Loop through the second cluster.
            for element2 in c2:
                # Calculate the similarity.
                sim = self.sim_method(element1.values, element2.values)
                sims.append(sim)
        # Return the average.
        return sum(sims)/len(sims)
        

    def get_similarities(self, clusters):
        sims = dict()
        # Loop through each cluster.
        for i in range(0, len(clusters)-1):
            # Calculate the similarity to the next cluster.
            sim = self.calculate_inter_cluster_similarity(clusters.iloc[i]["Elements"], clusters.iloc[i+1]["Elements"])
            sims[i] = sim

        sims = pd.Series(sims)
        sims.index = clusters.index[:-1]
        return sims

    
    def vnc(self, w_matrix, print_output=False):
        # Create the initial cluster matrix (one cluster per row)
        clusters = self.make_initial_clusters(w_matrix)
        i = 0
        # The list that contains the info to make the dendrogram.
        d_list = list()
        while clusters.shape[0] > 1:
            sims = self.get_similarities(clusters)
            most_similar = sims.idxmax()

            # Get the dates that are most similar
            date1 = most_similar
            # Get the row with the index one up from the most similar (i.e. the one it was compared to)
            date2 = clusters.index[clusters.index.get_loc(most_similar) + 1]

            # Get the most similar row
            cluster1 = clusters.loc[date1]['Elements']
            cluster2 = clusters.loc[date2]['Elements']

            # Merge the rows by creating a new row with the clusters merged.
            merged_cluster = pd.concat([cluster1, cluster2])

            # Find the midpoint of all the dates in the new cluster.
            mid_point = VNC.find_midpoint(merged_cluster)

            # Calculate the new Index (the length of the initital array plus the current iteration)
            new_index = len(w_matrix) + i

            # Print the items being merged if "print_output" is set to True.
            if print_output:
                print("Merging {0} and {1} - Similarity of {2}".format(date1, date2, sims[date1]))

            # Append a row to the list for our dendrogram output
            d_list.append([clusters.loc[date1]["Index_Val"], 
                        clusters.loc[date2]["Index_Val"], 
                        1-sims[date1], 
                        len(merged_cluster)])

            # drop the two rows that were most similar
            clusters.drop([date1, date2], inplace=True)
            # add the new combined row
            clusters.loc[mid_point] = pd.Series({"Elements": merged_cluster, "Index_Val": new_index})

            # Sort the index again because it is 
            clusters.sort_index(ascending=True, inplace=True)

            # increase the iteration number
            i += 1
        
        return np.array(d_list)


    def draw_dendrogram(self, cutoff=None, label="", ax=None, colour="k", orientation="top"):
        given_ax=ax
        # Set up the axes
        if ax is None:
            fig, ax = plt.subplots(figsize=(50, 50))
        ax.set_title(label, fontsize=40)
        ax.tick_params(labelsize=30)

        # Plot the dendrogram.
        dendrogram(
            self.d_list,
            leaf_rotation=90.,  # rotates the x axis labels
            leaf_font_size=20.,  # font size for the x axis labels
            labels = self.matrix.index.to_series().apply(lambda x: x.strftime("%Y-%m")), 
            ax=ax,
            color_threshold=0,
            above_threshold_color=colour,
            orientation=orientation
        )
        if cutoff is not None:
            ax.axhline(y=cutoff, c='k')

        if given_ax is None:
            return fig, ax
        else:
            return

    def get_clusters(self, cutoff):
        clusters = fcluster(self.d_list, cutoff, criterion='distance')
        return clusters


class SimpleVNC(VNC):
    def get_similarities(self, kw_matrix):
        all_windows = list(kw_matrix.index)

        sims = dict()
        for i in range(0, len(all_windows)-1):
            diff = self.sim_method(kw_matrix.loc[all_windows[i]], kw_matrix.loc[all_windows[i+1]])
            sims[all_windows[i]] = diff

        sims = pd.Series(sims)
        return sims
    

    def vnc(self, w_matrix, print_output=False):
        index_vals = pd.Series(np.arange(0, len(w_matrix)), index=w_matrix.index)
        cluster_sizes = pd.Series(np.ones(len(w_matrix)), index=w_matrix.index)
        i = 0
        d_list = []
        num_elements = len(w_matrix)
        
        while w_matrix.shape[0] > 1:
            sims = self.get_similarities(w_matrix)
            most_similar = sims.idxmax()

            # Get the dates that are most similar
            date1 = most_similar
            # Get the row with the index one up from the most similar (i.e. the one it was compared to)
            date2 = w_matrix.index[w_matrix.index.get_loc(most_similar) + 1]

            # Get the most similar row
            row1 = w_matrix.loc[date1]
            row2 = w_matrix.loc[date2]

            # Logical AND the two rows to "merge" them.
            merged = np.mean([row1.values, row2.values], axis=0)
            # Find the midpoint of the two dates.
            mid_point = date1 + (date2 - date1)/2
            
            index_vals[mid_point] = num_elements + i
            cluster_sizes[mid_point] = cluster_sizes[date1] + cluster_sizes[date2]

            # Print the items being merged if "print_output" is set to True.
            if print_output:
                print("Merging {0} and {1} - Similarity of {2}".format(date1, date2, sims[date1]))
            
            # Append a row to the list for our dendrogram output
            d_list.append([index_vals[date1], 
                        index_vals[date2], 
                        1-sims[date1], 
                        cluster_sizes[mid_point]])
            
            index_vals.drop([date1, date2], inplace=True)
            cluster_sizes.drop([date1, date2], inplace=True)

            # drop the two rows that were most similar
            w_matrix.drop([date1, date2], inplace=True)
            # add the new combined row
            w_matrix.loc[mid_point] = merged

            # Sort the index again because it is 
            w_matrix.sort_index(ascending=True, inplace=True)
            index_vals.sort_index(ascending=True, inplace=True)
            cluster_sizes.sort_index(ascending=True, inplace=True)
            
            i += 1
        return np.array(d_list)


def load_data(data_dir, group_names):
    group_w_freqs = dict()
    for group in group_names:
        # Read data
        matrix = pd.read_csv(os.path.join(data_dir, "{0}-word-freq.csv".format(group)),
                                        index_col=0)
        # Convert the index to datetime.
        convert_to_date = lambda x: datetime.strptime(x, "%Y/%m/%d")
        matrix.set_index(matrix.reset_index()['index'].apply(convert_to_date), inplace=True)

        normalised = pd.DataFrame(matrix.drop("TOTAL", axis=1).values / matrix["TOTAL"].values[:,None], index=matrix.index, columns=matrix.drop("TOTAL", axis=1).columns)
        group_w_freqs[group] = normalised

    return group_w_freqs


def plot_vnc(f_matrix, out_file, colour, sim_method=cosine_sim, date_colour_map=None):
    vnc = VNC(f_matrix, sim_method)
    c, coph_dists = cophenet(vnc.d_list, pdist(f_matrix))
    print("Cophenetic Correlation Coefficient: {}".format(c))

    fig, ax = vnc.draw_dendrogram(colour=colour)

    for ytick in ax.yaxis.get_ticklabels():
            ytick.set_size(45)

    for xtick in ax.xaxis.get_ticklabels():
        if date_colour_map is not None:
            curr_text = xtick.get_text()
            xtick.set_color(date_colour_map[curr_text])
        xtick.set_size(45)

    fig.savefig(out_file)


def plot_vnc_per_group(group_w_freqs, out_file, colours=colour_list):
    # # VCA Algorithm
    #
    # Now we will loop through the keyword matrix and perform the clustering.
    #
    # The algorithm is roughly as follows:
    #
    # 1. Go through each row of keyword matrix.
    # 2. For each timepoint, calculate the similarity to the next timepoint.
    #     _(We will use Cosine because that is common for NLP vector comparisons)._
    # 3. Merge the two most similar months.  _(the lowest valued element from 2. with the following element)_.
    #     _For merging, we will have two methods: a - averaging the two elements, and b - keeping all elements in the cluster and compare using the average of all elements in the cluster._ features of the group.
    # 4. Repeat until one element remains.


    # Create colour gradient mapping.
    red = Color("red")
    dates = list(group_w_freqs.values())[0].index
    grad_colours = list(red.range_to(Color("green"),len(dates)))
    grad_colours = [c.hex_l for c in grad_colours]
    date_colour_map = {i.strftime("%Y-%m"): c for i, c in zip(dates, grad_colours)}

    for group, colour in zip(group_w_freqs.keys(), colours):
        print("GROUP: {}".format(group))
        f_name = os.path.join(out_file, "{}-vnc-dendrogram.pdf".format(group))
        f_matrix = group_w_freqs[group]
        plot_vnc(f_matrix, f_name, colour, date_colour_map=date_colour_map)


def plot_side_by_side_per_group(group_w_freqs, out_dir, name_suffix=None):
    interest_groups = list(group_w_freqs.keys())
    for combo in list(combinations(interest_groups, 2)):
        m1 = group_w_freqs[combo[0]]
        m2 = group_w_freqs[combo[1]]

        name = "-".join(combo if name_suffix is None else list(combo) + [name_suffix])
        plot_side_by_side_vnc(m1, m2, name, out_dir, gnames=combo)


def plot_horizontal_vnc(curr_vnc, ax, orientation, colour, title="", date_colour_map=None):
    with plt.rc_context({'lines.linewidth': 7}): # Was 7
        curr_vnc.draw_dendrogram(ax=ax, colour=colour, orientation=orientation)

    # plt.xticks(ax.xaxis.get_ticklocs(), matrix.index)

    for ytick in ax.yaxis.get_ticklabels():
        if date_colour_map is not None:
            curr_text = ytick.get_text()
            ytick.set_color(date_colour_map[curr_text])
        ytick.set_size(80)  # Was 45
        ytick.set_rotation(0)
        
    for xtick in ax.xaxis.get_ticklabels():
        curr_text = xtick.get_text()
        xtick.set_size(80)

    ax.set_title(title, fontsize=100)
    ax.grid(axis="y", alpha=0.5, color="gray")

def plot_side_by_side_vnc(m1, m2, name, out_dir, gnames=["", ""], fig_colours=["#377eb8", "#e41a1c"]):
    # Create colour gradient mapping.
    red = Color("red")
    dates = list(m1.index)
    colours = list(red.range_to(Color("green"),len(dates)))
    colours = [c.hex_l for c in colours]
    date_colour_map = {i.strftime("%Y-%m"): c for i, c in zip(dates, colours)}

    # This method calculates cosine distance between two vectors.
    from scipy.spatial.distance import cosine as cosine_dist
    # This method simply inverts it to get similarity.
    cosine_sim = lambda x,y: 1 - cosine_dist(x,y)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(100, 40), sharey=True)

    # Plot first VNC
    vnc1 = VNC(m1, cosine_sim)
    c, coph_dists = cophenet(vnc1.d_list, pdist(m1))
    print("Cophenetic Correlation Coefficient: {}".format(c))
    plot_horizontal_vnc(vnc1, ax1, "left", colour=fig_colours[0], title=gnames[0], date_colour_map=date_colour_map)

    # Plot second VNC.
    vnc2 = VNC(m2, cosine_sim)
    c, coph_dists = cophenet(vnc2.d_list, pdist(m2))
    print("Cophenetic Correlation Coefficient: {}".format(c))
    plot_horizontal_vnc(vnc2, ax2, "right", colour=fig_colours[1], title=gnames[1], date_colour_map=date_colour_map)

    for ytick in ax2.yaxis.get_ticklabels():
        ytick.set_rotation(0)

    plt.subplots_adjust(wspace=0.14, hspace=0)
    #plt.show()

    fig.savefig(os.path.join(out_dir, "{0}-comp-vnc.pdf".format(name)))



if __name__ == "__main__":
    # ## Load in the Data
    #
    # Read in the matrix for each of the groups we are interested in.
    if len(sys.argv) > 1:
        groups_fp = sys.argv[1]
    else:
        groups_fp = "/home/ed/Documents/hansard-stuff/Analysis/lab_con"

    interest_groups = []
    for fname in os.listdir(groups_fp):
        m = re.match(r"(\S+)-word-freq.csv", fname)
        if m:
            interest_groups.append(m.group(1))

    print("Groups we're looking at:\n{0}".format(interest_groups))
    group_w_freqs = load_data(groups_fp, interest_groups)

    plot_vnc_per_group(group_w_freqs, groups_fp)

    plot_vnc_per_group(group_w_freqs, groups_fp)