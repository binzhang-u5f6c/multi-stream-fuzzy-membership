# -*- coding: utf-8 -*-
"""Evaluation of FuzzMDA on real-world data set Weather.

Evaluation of FuzzMDA on real-world data set Weather.
"""
import sys

import numpy as np
from sklearn import linear_model
from skmultiflow.trees import HoeffdingTreeRegressor
from skmultiflow.trees import HoeffdingAdaptiveTreeRegressor
from skmultiflow.meta import AdaptiveRandomForestRegressor

from fuzzm.loader import WeatherStreams
from fuzzm.model import StreamHandler

# load data
ss = WeatherStreams()
n, m, d = ss.x.shape
train_size = 500
batch_size = int(sys.argv[1])
x_train, y_train = ss[:train_size]
result = np.zeros((8, m, n//10))

# models
baseline1 = [StreamHandler(linear_model.Ridge(alpha=1)) for _ in range(m)]
for j, hdlr in enumerate(baseline1):
    jj = (j + 1) % m
    hdlr.fit(x_train[:, j, :], y_train[:, j],
             x_train[:, jj, :], y_train[:, jj])
baseline2 = [StreamHandler(linear_model.Ridge(alpha=1)) for _ in range(m)]
for j, hdlr in enumerate(baseline2):
    jj = (j + 1) % m
    hdlr.fit(x_train[:, j, :], y_train[:, j],
             x_train[:, jj, :], y_train[:, jj])
ht1 = [StreamHandler(HoeffdingTreeRegressor()) for _ in range(m)]
for j, hdlr in enumerate(ht1):
    jj = (j + 1) % m
    hdlr.fit(x_train[:, j, :], y_train[:, j],
             x_train[:, jj, :], y_train[:, jj])
ht2 = [StreamHandler(HoeffdingTreeRegressor()) for _ in range(m)]
for j, hdlr in enumerate(ht2):
    jj = (j + 1) % m
    hdlr.fit(x_train[:, j, :], y_train[:, j],
             x_train[:, jj, :], y_train[:, jj])
hat1 = [StreamHandler(HoeffdingAdaptiveTreeRegressor()) for _ in range(m)]
for j, hdlr in enumerate(hat1):
    jj = (j + 1) % m
    hdlr.fit(x_train[:, j, :], y_train[:, j],
             x_train[:, jj, :], y_train[:, jj])
hat2 = [StreamHandler(HoeffdingAdaptiveTreeRegressor()) for _ in range(m)]
for j, hdlr in enumerate(hat2):
    jj = (j + 1) % m
    hdlr.fit(x_train[:, j, :], y_train[:, j],
             x_train[:, jj, :], y_train[:, jj])
arf1 = [StreamHandler(AdaptiveRandomForestRegressor()) for _ in range(m)]
for j, hdlr in enumerate(arf1):
    jj = (j + 1) % m
    hdlr.fit(x_train[:, j, :], y_train[:, j],
             x_train[:, jj, :], y_train[:, jj])
arf2 = [StreamHandler(AdaptiveRandomForestRegressor()) for _ in range(m)]
for j, hdlr in enumerate(arf2):
    jj = (j + 1) % m
    hdlr.fit(x_train[:, j, :], y_train[:, j],
             x_train[:, jj, :], y_train[:, jj])

N = (n - train_size) // batch_size
for i in range(N//10):
    if i % 10 == 0:
        print('Processing {}/{} ...'.format(i, N))
    i1 = train_size + i * batch_size
    i2 = i1 + batch_size
    x, y = ss[i1:i2]
    # baseline1
    dlist = []
    nlist = []
    for j in range(m):
        result[0, j, i], drift = baseline1[j].score(x[:, j, :], y[:, j])
        if drift:
            dlist.append(j)
        else:
            nlist.append(j)
    for j in dlist:
        tlist = [ii for ii in nlist]
        tlist.append(j)
        jj = (j + 1) % m
        xx = x[:, tlist, :].reshape(-1, d)
        yy = y[:, tlist].reshape(-1)
        wght = np.ones(len(tlist)*batch_size)
        for ki, k in enumerate(tlist):
            if k == j:
                break
            k1 = ki * batch_size
            k2 = k1 + batch_size
            wght[k1:k2] *= baseline1[k].score(x[:, j, :], y[:, j], True).mean()
        baseline1[j].fit(xx, yy, x[:, jj, :], y[:, jj], sample_weight=wght)
    # baseline2
    dlist = []
    for j in range(m):
        result[1, j, i], drift = baseline2[j].score(x[:, j, :], y[:, j])
        if drift:
            dlist.append(j)
    for j in dlist:
        jj = (j + 1) % m
        xx = x[:, j, :].reshape(-1, d)
        yy = y[:, j].reshape(-1)
        baseline2[j].fit(xx, yy, x[:, jj, :], y[:, jj])

    # ht1
    dlist = []
    nlist = []
    for j in range(m):
        result[2, j, i], drift = ht1[j].score(x[:, j, :], y[:, j])
        if drift:
            dlist.append(j)
        else:
            nlist.append(j)
    for j in dlist:
        tlist = [ii for ii in nlist]
        tlist.append(j)
        jj = (j + 1) % m
        xx = x[:, tlist, :].reshape(-1, d)
        yy = y[:, tlist].reshape(-1)
        wght = np.ones(len(tlist)*batch_size)
        for ki, k in enumerate(tlist):
            if k == j:
                break
            k1 = ki * batch_size
            k2 = k1 + batch_size
            wght[k1:k2] *= ht1[k].score(x[:, j, :], y[:, j], True).mean()
        ht1[j].partial_fit(xx, yy, x[:, jj, :], y[:, jj], sample_weight=wght)
    # ht2
    dlist = []
    for j in range(m):
        result[3, j, i], drift = ht2[j].score(x[:, j, :], y[:, j])
        if drift:
            dlist.append(j)
    for j in dlist:
        jj = (j + 1) % m
        xx = x[:, j, :].reshape(-1, d)
        yy = y[:, j].reshape(-1)
        ht2[j].partial_fit(xx, yy, x[:, jj, :], y[:, jj])

    # hat1
    dlist = []
    nlist = []
    for j in range(m):
        result[4, j, i], drift = hat1[j].score(x[:, j, :], y[:, j])
        if drift:
            dlist.append(j)
        else:
            nlist.append(j)
    for j in dlist:
        tlist = [ii for ii in nlist]
        tlist.append(j)
        jj = (j + 1) % m
        xx = x[:, tlist, :].reshape(-1, d)
        yy = y[:, tlist].reshape(-1)
        wght = np.ones(len(tlist)*batch_size)
        for ki, k in enumerate(tlist):
            if k == j:
                break
            k1 = ki * batch_size
            k2 = k1 + batch_size
            wght[k1:k2] *= hat1[k].score(x[:, j, :], y[:, j], True).mean()
        hat1[j].partial_fit(xx, yy, x[:, jj, :], y[:, jj], sample_weight=wght)
    # hat2
    dlist = []
    for j in range(m):
        result[5, j, i], drift = hat2[j].score(x[:, j, :], y[:, j])
        if drift:
            dlist.append(j)
    for j in dlist:
        jj = (j + 1) % m
        xx = x[:, j, :].reshape(-1, d)
        yy = y[:, j].reshape(-1)
        hat2[j].partial_fit(xx, yy, x[:, jj, :], y[:, jj])

    # arf1
    dlist = []
    nlist = []
    for j in range(m):
        result[6, j, i], drift = arf1[j].score(x[:, j, :], y[:, j])
        if drift:
            dlist.append(j)
        else:
            nlist.append(j)
    for j in dlist:
        tlist = [ii for ii in nlist]
        tlist.append(j)
        jj = (j + 1) % m
        xx = x[:, tlist, :].reshape(-1, d)
        yy = y[:, tlist].reshape(-1)
        wght = np.ones(len(tlist)*batch_size)
        for ki, k in enumerate(tlist):
            if k == j:
                break
            k1 = ki * batch_size
            k2 = k1 + batch_size
            wght[k1:k2] *= arf1[k].score(x[:, j, :], y[:, j], True).mean()
        arf1[j].partial_fit(xx, yy, x[:, jj, :], y[:, jj], sample_weight=wght)
    # arf2
    dlist = []
    for j in range(m):
        result[7, j, i], drift = arf2[j].score(x[:, j, :], y[:, j])
        if drift:
            dlist.append(j)
    for j in dlist:
        jj = (j + 1) % m
        xx = x[:, j, :].reshape(-1, d)
        yy = y[:, j].reshape(-1)
        arf2[j].partial_fit(xx, yy, x[:, jj, :], y[:, jj])

with open('result3.weather.{:03d}.npy'.format(batch_size), 'wb') as f:
    np.save(f, result)
