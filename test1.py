# -*- coding: utf-8 -*-
"""Parameter study of k.

Evaluate our method with different k.
k range from 5 to 14.
"""
import sys
from json import load
from copy import deepcopy as dcp

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from skmultiflow.trees import HoeffdingAdaptiveTreeRegressor
from skmultiflow.meta import AdaptiveRandomForestRegressor
from skmultiflow.meta import RegressorChain
from skmultiflow.trees import iSOUPTreeRegressor
from skmultiflow.trees import StackedSingleTargetHoeffdingTreeRegressor
from sklearn.metrics import mean_squared_error

from src.loader import TrainStreams
from src.loader import WeatherStreams
from src.loader import SensorStreams
from src.detector import KMeansChi2TestDetector

sys.setrecursionlimit(10000000)

# read configuration
with open('config.json', 'r') as f:
    config = load(f)
alpha = config["threshold"]


def main(streams_name, k, batch_size):
    """Wrap all methods to be evaluated."""
    # loading data
    if streams_name == 'train':
        streams = TrainStreams()
        m = 8
    elif streams_name == 'weather':
        streams = WeatherStreams()
        m = 10
    elif streams_name == 'sensor':
        streams = SensorStreams()
        m = 6
    x_train, y_train = streams[:batch_size]

    # initialize models
    baseline = [dcp(DecisionTreeRegressor(ccp_alpha=1)) for i in range(m)]
    for i, learner in enumerate(baseline):
        learner.fit(x_train[:, i, :], y_train[:, i])
    hat = [dcp(HoeffdingAdaptiveTreeRegressor()) for i in range(m)]
    for i, learner in enumerate(hat):
        learner.partial_fit(x_train[:, i, :], y_train[:, i])
    arf = [dcp(AdaptiveRandomForestRegressor(3)) for i in range(m)]
    for i, learner in enumerate(arf):
        learner.partial_fit(x_train[:, i, :], y_train[:, i])
    rc = RegressorChain(base_estimator=DecisionTreeRegressor())
    rc.fit(x_train.reshape(batch_size, -1), y_train)
    ist = iSOUPTreeRegressor()
    ist.fit(x_train.reshape(batch_size, -1), y_train)
    sstht = StackedSingleTargetHoeffdingTreeRegressor()
    sstht.fit(x_train.reshape(batch_size, -1), y_train)
    dams = [dcp(DecisionTreeRegressor(ccp_alpha=1)) for i in range(m)]
    for i, learner in enumerate(dams):
        learner.fit(x_train[:, i, :], y_train[:, i])
    km_chi2t_dtrs = [KMeansChi2TestDetector(k,
                                            x_train[-batch_size:, i, :],
                                            y_train[-batch_size:, i])
                     for i in range(m)]

    # start
    head_index = np.zeros(m, dtype='int32')
    n = (len(streams) - batch_size) // batch_size
    scores = np.zeros((7, n, m))
    for t in range(n):
        if t % 10 == 0:
            print('Processing chunk #{}/{}...'.format(t, n))
        # get data
        t1 = batch_size + t * batch_size
        t2 = t1 + batch_size
        x, y = streams[t1:t2]
        # make prediction and evaluate
        y_hat1 = np.zeros((batch_size, m))
        y_hat2 = np.zeros((batch_size, m))
        y_hat3 = np.zeros((batch_size, m))
        y_hat4 = rc.predict(x.reshape(batch_size, -1))
        y_hat5 = ist.predict(x.reshape(batch_size, -1))
        y_hat6 = sstht.predict(x.reshape(batch_size, -1))
        y_hat7 = np.zeros((batch_size, m))
        for i in range(m):
            y_hat1[:, i] = baseline[i].predict(x[:, i, :])
            y_hat2[:, i] = hat[i].predict(x[:, i, :])
            y_hat3[:, i] = arf[i].predict(x[:, i, :])
            y_hat7[:, i] = dams[i].predict(x[:, i, :])
        scores[0, t, :] = \
            mean_squared_error(y_hat1, y, multioutput='raw_values')
        scores[1, t, :] = \
            mean_squared_error(y_hat2, y, multioutput='raw_values')
        scores[2, t, :] = \
            mean_squared_error(y_hat3, y, multioutput='raw_values')
        scores[3, t, :] = \
            mean_squared_error(y_hat4, y, multioutput='raw_values')
        scores[4, t, :] = \
            mean_squared_error(y_hat5, y, multioutput='raw_values')
        scores[5, t, :] = \
            mean_squared_error(y_hat6, y, multioutput='raw_values')
        scores[6, t, :] = \
            mean_squared_error(y_hat7, y, multioutput='raw_values')
        # drift detection
        drift_list = []
        for i in range(m):
            p = km_chi2t_dtrs[i].detect(x[:, i, :], y[:, i])
            if p < alpha:
                drift_list.append(i)
        # drift adaptation
        if len(drift_list) > 0:
            rc.fit(x.reshape(batch_size, -1), y)
            ist.fit(x.reshape(batch_size, -1), y)
            sstht.fit(x.reshape(batch_size, -1), y)
        for i in drift_list:
            head_index[i] = t1
            km_chi2t_dtrs[i].refit(x[:, i, :], y[:, i])
            # incremental training using just xt, yt
            baseline[i].fit(x[:, i, :], y[:, i])
            hat[i].partial_fit(x[:, i, :], y[:, i])
            arf[i].partial_fit(x[:, i, :], y[:, i])
            # incremental training using data from other data streams
            xi, yi = [x[:, i, :]], [y[:, i]]
            for j in range(m):
                if j not in drift_list:
                    p = km_chi2t_dtrs[i].detect(x[:, j, :], y[:, j])
                    if p >= alpha:
                        xx, yy = streams[head_index[j]:t2]
                        xi.append(xx[:, j, :])
                        yi.append(yy[:, j])
            xi = np.vstack(xi)
            yi = np.hstack(yi)
            dams[i].fit(xi, yi)
    with open('results/{}.{}.{}.npy'.format(streams_name, k, batch_size),
              'wb') as f:
        np.save(f, scores)


if __name__ == "__main__":
    for streams_name in ['train', 'weather', 'sensor']:
        for k in range(5, 15):
            for batch_size in [50, 100, 200, 500]:
                print(streams_name, k, batch_size)
                main(streams_name, k, batch_size)
