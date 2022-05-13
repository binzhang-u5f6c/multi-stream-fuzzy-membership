# -*- coding: utf-8 -*-
"""Visualization on real-world data sets.

Visualization on real-world data sets.
"""
import numpy as np
from sklearn.tree import DecisionTreeRegressor

from fuzzm.loader import TrainStreams
from fuzzm.loader import WeatherStreams
from fuzzm.model import StreamHandler

ss = TrainStreams()
n, m, d = ss.x.shape
t1 = 2000
t2 = t1 + 1000
x, y = ss[t1:t2]
result = np.ones((m, m))
for i in range(m):
    jj = (i + 1) % m
    hdlr = StreamHandler(DecisionTreeRegressor())
    hdlr.fit(x[:, i, :], y[:, i], x[:, jj, :], y[:, jj])
    for j in range(m):
        loss, _ = hdlr.score(x[:, j, :], y[:, j])
        result[i, j] = hdlr.mf.membership(loss).detach().cpu().numpy().mean()
with open('result6.npy', 'wb') as f:
    np.save(f, result)

ss = WeatherStreams()
n, m, d = ss.x.shape
t1 = 3000
t2 = t1 + 1000
x, y = ss[t1:t2]
result = np.ones((m, m))
for i in range(m):
    jj = (i + 1) % m
    hdlr = StreamHandler(DecisionTreeRegressor())
    hdlr.fit(x[:, i, :], y[:, i], x[:, jj, :], y[:, jj])
    for j in range(m):
        loss, _ = hdlr.score(x[:, j, :], y[:, j])
        result[i, j] = hdlr.mf.membership(loss).detach().cpu().numpy().mean()
with open('result7.npy', 'wb') as f:
    np.save(f, result)
