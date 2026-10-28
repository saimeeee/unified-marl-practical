"""Explicit scripted label generation, retaining the delivered strict comparison."""
import torch

def labels_from_team_returns(reward0,reward1):
    if reward0.ndim!=3 or reward1.ndim!=3 or reward0.shape[-1]!=1 or reward1.shape[-1]!=1:
        raise ValueError("Provide [pairs,time,1] team rewards; individual rewards are never summed to manufacture labels")
    if reward0.shape[0]!=reward1.shape[0]: raise ValueError("Pair counts differ")
    # Source make_pref_datasets_104.py compares complete supplied segment sums.
    # Ties select trajectory0 (label 0), rather than silently creating 0.5.
    return (reward0.sum(1)<reward1.sum(1)).float()
