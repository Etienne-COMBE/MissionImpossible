import pandas as pd
import numpy as np

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

def test_gini(df: pd.DataFrame):
    sample = pd.DataFrame({"target": [1,1,1,1,0,0,0,0]})
    gini = decision_gini(sample, "target")

    assert gini == 0.5
    
def test_entropy(df: pd.DataFrame):
    sample = pd.DataFrame({"target": [1,1,1,1,0,0,0,0]})
    entropy = decision_entropy(sample, "target")

    assert entropy == 1