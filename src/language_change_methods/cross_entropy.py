import json
import pandas as pd
from datetime import datetime

from language_change_methods.utility_functions import get_time_windows, get_data_windows
from language_change_methods.cross_entropy_sampling import multi_member_splits_with_limit, multi_member_splits, get_end_of_windows


def get_CE_comparisons(fp):
    with open(fp) as results_file:
        results = json.load(results_file)

    comparisons = [{gsnap: {gtest: {datetime.strptime(w, "%Y-%m-%d"): pd.Series(run[gsnap][gtest][w]) for w in run[gsnap][gtest]} for gtest in run[gsnap]} for gsnap in run} for run in results["comparisons"]]
    return comparisons
    

def get_ends_of_windows(fp):
    with open(fp) as results_file:
        results = json.load(results_file)
            
    return [datetime.strptime(w, "%Y-%m-%d") for w in results["end_of_windows"]] 


def get_groups_toks_and_contribs(queries, gnames, all_contributions, all_toks, token_limit=60, param_list=None):
    """
    Gets the provided groups and tokens given pandas queries.
    """
    out_contribs = dict()
    out_toks = dict()

    if param_list is None:
        param_list = [[] for g in gnames]

    for query, gname, params in zip(queries, gnames, param_list):
        # Get contributions for each group.
        curr = all_contributions.query(query)

        # Get tokens for each group.
        curr_toks = all_toks[all_toks.index.isin(curr.index)]

        # Only keep tokens for contributions with >60 posts.
        curr_toks = curr_toks[curr_toks.apply(len) >= token_limit].apply(lambda x: x[:token_limit])

        # Get rid of contributions with <= 60 posts.
        curr = curr[curr.index.isin(curr_toks.index)]

        # Set output
        out_contribs[gname] = curr
        out_toks[gname] = curr_toks

    # Create combined list of contributions
    combined = pd.concat(list(out_contribs.values()), axis=0)

    return out_contribs, out_toks, combined


def single_CE_run(gnames, curr_contribs, curr_toks, curr_ref, curr_ref_toks,
                    win_size, win_step, n_runs, balanced_groups, w_limit,
                    token_limit, n_contribs_per_mp, queries, out_fp, member_field="member"):

    if w_limit:
        # For doing with a limit per MP
        comparisons, meta = multi_member_splits_with_limit(gnames,
                                                        list(curr_contribs.values()),
                                                        list(curr_toks.values()),
                                                        curr_ref, curr_ref_toks,
                                                        window_func=get_data_windows,
                                                        window_size=win_size, window_step=win_step,
                                                        n_runs=n_runs, balanced_groups=balanced_groups,
                                                        comp_method="CE", n_words_per_contribution=token_limit,
                                                        n_contribs_per_mp=n_contribs_per_mp, 
                                                        member_field=member_field)
    else:
        comparisons, meta = multi_member_splits(gnames,
                                            list(curr_contribs.values()),
                                            list(curr_toks.values()),
                                            curr_ref, curr_ref_toks,
                                            window_func=get_data_windows,
                                            window_size=win_size, window_step=win_step,
                                            n_runs=n_runs, balanced_groups=balanced_groups,
                                            comp_method="CE", n_words_per_contribution=token_limit, 
                                            member_field=member_field)

    end_of_windows = get_end_of_windows(pd.concat(list(curr_contribs.values()) + [curr_ref], axis=0),
                                                    get_data_windows, win_size, win_step)
    end_of_windows = [datetime.strftime(d, "%Y-%m-%d") for d in end_of_windows]

    comparisons_dict = [{gsnap: {gtest: {datetime.strftime(w, "%Y-%m-%d"): run[gsnap][gtest][w].to_dict() for w in run[gsnap][gtest]} for gtest in run[gsnap]} for gsnap in run} for run in comparisons]

    meta_dict = [{metaVal: {gname: {datetime.strftime(w, "%Y-%m-%d"): run[metaVal][gname][w] for w in run[metaVal][gname]} for gname in run[metaVal]} for metaVal in run} for run in meta]

    param_combo = {"win_type": "contributions", "win_size": win_size, "win_step": win_step,
                    "n_runs": n_runs, "balanced": balanced_groups, "comp_method": "CE",
                    "contrib_limit": w_limit, "token_limit": token_limit, "queries": queries, "gnames": gnames}

    out_dict = {"params": param_combo, "comparisons": comparisons_dict, "meta": meta_dict, "end_of_windows": end_of_windows}

    with open(out_fp, "w") as out_file:
        json.dump(out_dict, out_file)

    print("Written file: ", out_fp)