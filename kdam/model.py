# -*- coding: utf-8 -*-
"""Stream handlers.

Stream handlers.
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from skmultiflow.trees import HoeffdingAdaptiveTreeRegressor
from skmultiflow.trees import HoeffdingTreeRegressor
from skmultiflow.trees import iSOUPTreeRegressor


class StreamHandler:
    """Handle a stream in dynamic environment."""

    def __init__(self, base_learner, warning_fa=2, drift_fa=3,
                 random_state=None):
        """__init__ for StreamHandler."""
        if base_learner == 'linear':
            self.learner = LinearRegression()
        elif base_learner == 'tree':
            self.learner = DecisionTreeRegressor(random_state=random_state,
                                                 ccp_alpha=1)
        elif base_learner == 'ist':
            self.learner = iSOUPTreeRegressor(random_state=random_state)
        elif base_learner == 'ht':
            self.learner = HoeffdingTreeRegressor(
                    random_state=random_state)
        elif base_learner == 'hat':
            self.learner = HoeffdingAdaptiveTreeRegressor(
                    random_state=random_state)
        else:
            raise ValueError('Base learner type not found!')
        self.warning_fa = warning_fa
        self.drift_fa = drift_fa
        self.min_mean = 0x7FFFFFFF
        self.min_std = 0x7FFFFFFF
        self.drift_lvl = 0

    def reset(self):
        """Reset method."""
        self.min_mean = 0x7FFFFFFF
        self.min_std = 0x7FFFFFFF
        self.drift_lvl = 0

    def fit(self, x, y, sample_weight=None):
        """Fit method."""
        self.learner.fit(x, y, sample_weight=sample_weight)

    def score(self, x, y, score_only=False):
        """Score method.

        x: ndarray of size (n, d)
        y: ndarray of size (n,)
        """
        yhat = self.learner.predict(x)
        scores = (yhat - y) ** 2
        if not score_only:
            current_mean = scores.mean()
            current_std = scores.std()
            if current_mean + current_std < self.min_mean + self.min_std:
                self.min_mean = current_mean
                self.min_std = current_std
            if current_mean + current_std > \
                    self.min_mean + self.drift_fa * self.min_std:
                self.drift_lvl = 2
            elif current_mean + current_std > \
                    self.min_mean + self.warning_fa * self.min_std:
                self.drift_lvl = 1
            else:
                self.drift_lvl = 0
        else:
            scores -= self.min_mean
            if self.min_std != 0:
                scores /= self.min_std
            scores = 1 / (1 + np.exp(scores))
        return scores
