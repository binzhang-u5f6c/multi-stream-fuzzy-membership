# -*- coding: utf-8 -*-
"""Test #2.

Validation of FuzzMDD
"""
import numpy as np
from sklearn import linear_model
from skmultiflow.drift_detection.ddm import DDM
from skmultiflow.drift_detection.hddm_a import HDDM_A
from skmultiflow.drift_detection.hddm_w import HDDM_W

from fuzzm.model import StreamHandler

count = np.zeros((10, 4, 2))
for i in range(10):
    rng = np.random.default_rng(i)
    x = rng.normal(0, 1, (1000, 10))
    y = np.zeros((1000, 1))
    w = rng.normal(0, 3, (10, 1))
    b = np.ones((500, 1)) * rng.normal(0, 3, 1)
    y[:500] = np.dot(x[:500], w) + b + rng.normal(0, 0.1, (500, 1))
    w += rng.normal(0, 0.1+0.1*i, (10, 1))
    b += rng.normal(0, 0.1+0.1*i, 1)
    y[500:] = np.dot(x[500:], w) + b + rng.normal(0, 0.1, (500, 1))

    for j in range(2000):
        choice1 = rng.choice(500, 50, replace=False)
        choice2 = rng.choice(500, 50, replace=False)
        choice2 += 500
        x1, x2 = x[choice1], x[choice2]
        y1, y2 = y[choice1], y[choice2]
        hdlr = StreamHandler(linear_model.Ridge(alpha=1), random_state=rng)
        hdlr.fit(x1, y1, x2, y2)
        choice3 = rng.choice(500, 50, replace=False)
        flag = 0
        if j % 2 == 0:
            choice3 += 500
            flag = 1
        x3, y3 = x[choice3], y[choice3]
        _, dr = hdlr.score(x3, y3)
        if dr is True and flag == 1:
            count[i, 0, 1] += 1
        if dr is False and flag == 0:
            count[i, 0, 0] += 1

        rgr = linear_model.Ridge(alpha=1)
        rgr.fit(x1, y1)
        ybar1 = rgr.predict(x1)
        mse = (ybar1 - y1) ** 2
        mse.sort()
        ybar2 = rgr.predict(x3)
        ddm = DDM(50)
        hddma = HDDM_A(warning_confidence=0.1)
        hddmw = HDDM_W(warning_confidence=0.1)
        for k in range(50):
            if abs(ybar1[k] - y1[k])**2 > mse[-2]:
                ddm.add_element(1)
                hddma.add_element(1)
                hddmw.add_element(1)
            else:
                ddm.add_element(0)
                hddma.add_element(0)
                hddmw.add_element(0)
        for k in range(50):
            if (ybar2[k] - y3[k])**2 > mse[-2]:
                ddm.add_element(1)
                hddma.add_element(1)
                hddmw.add_element(1)
            else:
                ddm.add_element(0)
                hddma.add_element(0)
                hddmw.add_element(0)
        if ddm.detected_warning_zone() and flag == 1:
            count[i, 1, 1] += 1
        if hddma.detected_warning_zone() and flag == 1:
            count[i, 2, 1] += 1
        if hddmw.detected_warning_zone() and flag == 1:
            count[i, 3, 1] += 1
        if not ddm.detected_warning_zone() and flag == 0:
            count[i, 1, 0] += 1
        if not hddma.detected_warning_zone() and flag == 0:
            count[i, 2, 0] += 1
        if not hddmw.detected_warning_zone() and flag == 0:
            count[i, 3, 0] += 1
print("Completed!")
with open('result2.npy', 'wb') as f:
    np.save(f, count)
