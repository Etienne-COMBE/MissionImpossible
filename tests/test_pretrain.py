import pandas as pd
import pytest

import src.pretrain as pt

# Gini index testing
def test_gini_min():
    sample = pd.DataFrame({"target": [1,1,1,1,1,1,1,1]})
    assert pt.decision_gini(sample, "target") == 0

def test_gini_max():
    sample = pd.DataFrame({"target": [1,1,1,1,0,0,0,0]})
    assert pt.decision_gini(sample, "target") == 0.5

def test_gini_example():
    sample = pd.DataFrame({"target": [0,1,1,2,2,2,2]})
    assert pt.decision_gini(sample, "target") == 4/7

# Entropy index testing
def test_entropy_min():
    sample = pd.DataFrame({"target": [1,1,1,1,1,1,1,1]})
    assert pt.decision_entropy(sample, "target") == 0

def test_entropy_max():
    sample = pd.DataFrame({"target": [1,1,1,1,0,0,0,0]})
    assert pt.decision_entropy(sample, "target") == 1

def test_entropy_example():
    sample = pd.DataFrame({"target": [0,1,2,2,3,3,3,3]})
    assert pt.decision_entropy(sample, "target") == 1.75

# Information gain testing (only with the entropy parameter)
def test_info_gain_max_entropy():
    sample = pd.DataFrame({"target": [1,1,1,1,0,0,0,0], "feature": [0,0,0,0,1,1,1,1]})
    assert pt.decision_info_gain(sample, "target", "feature", score_type = "entropy") == 1

def test_info_gain_min_entropy():
    sample = pd.DataFrame({"target": [1,1,1,1,0,0,0,0], "feature": [0,1,0,1,0,1,0,1]})
    assert pt.decision_info_gain(sample, "target", "feature", score_type = "entropy") == 0

def test_info_gain_exemple_entropy():
    sample = pd.DataFrame({"target": [0,1,2,2,3,3,3,3], "feature": [0,0,0,0,1,1,1,1]})
    assert pt.decision_info_gain(sample, "target", "feature", score_type = "entropy") == 1

# Information gain testing (only with the gini parameter)
def test_info_gain_max_gini():
    sample = pd.DataFrame({"target": [1,1,1,1,0,0,0,0], "feature": [0,0,0,0,1,1,1,1]})
    assert pt.decision_info_gain(sample, "target", "feature", score_type = "gini") == 0.5

def test_info_gain_min_gini():
    sample = pd.DataFrame({"target": [1,1,1,1,0,0,0,0], "feature": [0,1,0,1,0,1,0,1]})
    assert pt.decision_info_gain(sample, "target", "feature", score_type = "gini") == 0
"""
def test_info_gain_exemple_gini():
    sample = pd.DataFrame({"target": [0,1,2,2,3,3,3,3], "feature": [0,0,0,0,1,1,1,1]})
    assert pt.decision_info_gain(sample, "target", "feature", score_type = "gini") == 1
"""