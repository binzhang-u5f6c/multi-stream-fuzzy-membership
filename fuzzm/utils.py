# -*- coding: utf-8 -*-
"""Utilities.

Utilities.
"""
from sklearn import metrics


def mmd_rbf(x, y, gamma=0.1):
    """MMD using rbf kernel."""
    xx = metrics.pairwise.rbf_kernel(x, x, gamma)
    yy = metrics.pairwise.rbf_kernel(y, y, gamma)
    xy = metrics.pairwise.rbf_kernel(x, y, gamma)
    return xx.mean() + yy.mean() - 2 * xy.mean()
