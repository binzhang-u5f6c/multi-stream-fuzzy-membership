# -*- coding: utf-8 -*-
"""Test #1.

Equivalence of FuzzMDD with MMD.
"""
import time
import numpy as np
from sklearn import linear_model

from fuzzm.utils import mmd_rbf
from fuzzm.model import StreamHandler

result = np.zeros((2, 100))
t1, t2 = 0, 0
for i in range(100):
    rng = np.random.default_rng(i)
    x = rng.normal(0, 1, (1500, 10))
    y = np.zeros((1500, 1))
    w = rng.normal(0, 3, (10, 1))
    b = np.ones((500, 1)) * rng.normal(0, 3, 1)
    y[:500] = np.dot(x[:500], w) + b + rng.normal(0, 0.1, (500, 1))
    w += rng.normal(0, 0.1+0.01*i, (10, 1))
    b += rng.normal(0, 0.1+0.01*i, 1)
    y[500:1000] = np.dot(x[500:1000], w) + b + rng.normal(0, 0.1, (500, 1))
    y[1000:] = np.dot(x[1000:], w) + b + rng.normal(0, 0.1, (500, 1))

    hdlr = StreamHandler(linear_model.Ridge(alpha=1), random_state=rng)
    hdlr.fit(x[:500], y[:500], x[1000:], y[1000:])
    loss, _ = hdlr.score(x[500:1000], y[500:1000])
    tau1 = time.time()
    result[0, i] = hdlr.mf.membership(loss).detach().cpu().numpy().mean()
    tau2 = time.time()
    t1 += tau2 - tau1
    xx = np.ones((500, 11))
    xx[:, :10] = x[:500, :]
    xx[:, 10] = y[:500].reshape(-1)
    yy = np.ones((500, 11))
    yy[:, :10] = x[500:1000, :]
    yy[:, 10] = y[500:1000].reshape(-1)
    tau1 = time.time()
    result[1, i] = mmd_rbf(xx, yy)
    tau2 = time.time()
    t2 += tau2 - tau1
print(t1/100)
print(t2/100)
with open('result1.npy', 'wb') as f:
    np.save(f, result)
