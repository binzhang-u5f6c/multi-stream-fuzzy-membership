# -*- coding: utf-8 -*-
"""Stream handlers.

Stream handlers.
"""
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans


class KMeansChi2TestDetector:
    """Test whether the distribution of two data batch is different."""

    def __init__(self, k, x, y):
        """__init__ for KMeansChi2TestDetector."""
        self.cluster = KMeans(k)
        xy = np.hstack((x, y.reshape(-1, 1)))
        self.cluster.fit(xy)
        self.labels, self.counts = \
            np.unique(self.cluster.labels_, return_counts=True)

    def detect(self, x, y):
        """Detect method."""
        xy = np.hstack((x, y.reshape(-1, 1)))
        labels2 = self.cluster.predict(xy)
        labels2 = np.hstack((labels2, self.labels))
        _, counts2 = np.unique(labels2, return_counts=True)
        counts2 -= 1
        obs = np.vstack((self.counts, counts2))
        _, p, _, _ = chi2_contingency(obs)
        return p

    def refit(self, x, y):
        """Refit method."""
        xy = np.hstack((x, y.reshape(-1, 1)))
        self.cluster.fit(xy)
        self.labels, self.counts = \
            np.unique(self.cluster.labels_, return_counts=True)


class SlidingWindow:
    """A sliding window to store data."""

    def __init__(self, window_size):
        """__init__ for SlidingWindow."""
        self.count = 0
        self.window_size = window_size
        self.window = np.ones(window_size)

    def append(self, val):
        """Append a value."""
        i = self.count % self.window_size
        self.window[i] = val
        self.count += 1

    def full(self):
        """Detect whether the window is full."""
        return self.count >= self.window_size

    def reset(self):
        """Reset the window."""
        self.count = 0


class StreamHandler:
    """Handle a stream in dynamic environment."""

    def __init__(self, base_learner, window_size, warning_fa=2, drift_fa=3):
        """__init__ for StreamHandler."""
        if base_learner == 'linear':
            self.learner = LinearRegression()
        elif base_learner == 'tree':
            self.learner = DecisionTreeRegressor(ccp_alpha=1)
        else:
            raise ValueError('Base learner type not found!')
        self.score_window = SlidingWindow(window_size)
        self.warning_fa = warning_fa
        self.drift_fa = drift_fa
        self.min_mean = 0x7FFFFFFF
        self.min_std = 0x7FFFFFFF
        self.drift_lvl = 0
        self.tca = None

    def reset(self):
        """Reset method."""
        self.score_window.reset()
        self.min_mean = 0x7FFFFFFF
        self.min_std = 0x7FFFFFFF
        self.drift_lvl = 0

    def fit(self, x, y):
        """Fit method."""
        if self.tca is not None:
            x = self.tca.trasfer(x)
        self.learner.fit(x, y)

    def score(self, x, y):
        """Score method."""
        yhat = self.learner.predict(x)
        score_ = (yhat - y) ** 2
        self.score_window.append(score_)
        if self.score_window.full():
            current_mean = self.score_window.window.mean()
            current_std = self.score_window.window.std()
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
        return score_

    def verified(self, x, y):
        """Verify method."""
        yhat = self.learner.predict(x)
        score = (yhat - y) ** 2
        return score.mean() + score.std() <= \
            self.min_mean + self.drift_fa * self.min_std
