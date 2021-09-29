import pandas as pd
import numpy as np

import utils as utl
def decision_gini(df: pd.DataFrame, target: str, child: str = None) -> float:
    target_count = df.groupby(target)[target].count()
    pi = target_count / len(df)

    if child != None:
        table = df.groupby([target, child])[target].count().unstack()
        gini_knowing = 0
        for cat in table:
            p = table[cat].sum() / len(df)
            pi_child = table[cat] / table[cat].sum()
            gini = 1 - (pi_child ** 2).sum()
            gini_knowing = gini_knowing + gini * p
        return gini_knowing

    gini = 1 - (pi**2).sum()
    return gini

def decision_entropy(df: pd.DataFrame, target: str, child: str = None):
    target_count = df.groupby(target)[target].count()
    pi = target_count / len(df)

    if child != None:
        table = df.groupby([target, child])[target].count().unstack()
        entropy_knowing = 0
        for cat in table:
            p = table[cat].sum() / len(df)
            pi_child = table[cat] / table[cat].sum()
            entropy = - (pi_child * np.log2(pi_child)).sum()
            entropy_knowing = entropy_knowing + entropy * p
        return entropy_knowing

    entropy = - (pi * np.log2(pi)).sum()
    return entropy

def decision_info_gain(df: pd.DataFrame, target: str, child: str, score_type: str = 'gini'):
    if score_type == 'entropy':
        entropy = decision_entropy(df, target)
        entropy_knowing = decision_entropy(df, target, child)
        gain = entropy - entropy_knowing
    if score_type == 'gini':
        gain = decision_gini(df, target)
    return gain

def test_decision_overfit(df: pd.DataFrame, target: str, n_depths = None):
    score_list = []
    for i in range(1, n_depths):
        train_score, test_score = utl.decision_tree(df, target, encoding = True, max_depth=i, score = "return")
        score_list.append([train_score, test_score])
    return score_list