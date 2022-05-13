# -*- coding: utf-8 -*-
"""Visualization on synthetic data sets.

Visualization on synthetic data sets.
"""
import numpy as np
from sklearn import linear_model

from fuzzm.model import StreamHandler

rng = np.random.default_rng(0)
x = rng.normal(0, 1, (5, 500, 10))
y = np.zeros((5, 500, 1))
w = rng.normal(0, 3, (10, 1))
b = np.ones((500, 1)) * rng.normal(0, 3, 1)
y[0] = np.dot(x[0], w) + b + rng.normal(0, 0.1, (500, 1))
y[1] = np.dot(x[1], w) + b + rng.normal(0, 0.1, (500, 1))
y[2] = np.dot(x[2], w) + b + rng.normal(0, 0.1, (500, 1))
w += rng.normal(1, 1, (10, 1))
b += rng.normal(1, 1, 1)
y[3] = np.dot(x[3], w) + b + rng.normal(0, 0.1, (500, 1))
y[4] = np.dot(x[4], w) + b + rng.normal(0, 0.1, (500, 1))

result = np.zeros((5, 5))
for i in range(5):
    for j in range(5):
        hdlr = StreamHandler(linear_model.Ridge(alpha=1), random_state=rng)
        hdlr.fit(x[i], y[i], x[j], y[j])
        loss, _ = hdlr.score(x[j], y[j])
        result[i, j] = hdlr.mf.membership(loss).detach().cpu().numpy().mean()
with open('result5.npy', 'wb') as f:
    np.save(f, result)
