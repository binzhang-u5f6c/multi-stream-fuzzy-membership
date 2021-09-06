# -*- coding: utf-8 -*-
"""Evaluating methods on a three real-world data sets.

A method to evaluate multiple methods on a specific dataset.
"""
from copy import deepcopy as dcp

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from skmultiflow.meta import RegressorChain
from skmultiflow.trees import iSOUPTreeRegressor
from skmultiflow.trees import StackedSingleTargetHoeffdingTreeRegressor

from kdam.loader import TrainStreams
from kdam.loader import WeatherStreams
from kdam.loader import SensorStreams
from kdam.model import StreamHandler


def get_streams(stream_name):
    """Get a stream."""
    if stream_name == 'train':
        streams = TrainStreams()
        m = 8
        d = 11
    elif stream_name == 'weather':
        streams = WeatherStreams()
        m = 10
        d = 8
    elif stream_name == 'sensor':
        streams = SensorStreams()
        m = 6
        d = 3
    return streams, m, d


def evaluate(dataset, training_size, w_size):
    """Evaluate a model on a dataset."""
    # loading data
    streams, m, d = get_streams(dataset)
    x_train, y_train = streams[:training_size]

    # initialize models
    baseline = [dcp(StreamHandler('tree')) for _ in range(m)]
    dams = [dcp(StreamHandler('tree')) for _ in range(m)]
    rc = RegressorChain(DecisionTreeRegressor(ccp_alpha=1))
    ist = iSOUPTreeRegressor()
    sstht = StackedSingleTargetHoeffdingTreeRegressor()
    for i in range(m):
        baseline[i].fit(x_train[:, i, :], y_train[:, i])
        dams[i].fit(x_train[:, i, :], y_train[:, i])
    rc.fit(x_train.reshape(-1, m*d), y_train)
    ist.fit(x_train.reshape(-1, m*d), y_train)
    sstht.fit(x_train.reshape(-1, m*d), y_train)

    # start
    n = len(streams)
    batch_num = (n - training_size) // w_size
    scores = np.zeros((5, batch_num, m))
    for t in range(batch_num):
        if t % 100 == 0:
            print('>>> processing batch #{} of {}'.format(t, batch_num))
        # get data
        t1 = training_size + t * w_size
        t2 = t1 + w_size
        # x: ndarray of size(n, m, d), y: ndarray of size (n, m)
        x, y = streams[t1:t2]

        # baseline
        drift_list = []
        for i in range(m):
            scores[0, t, i] = baseline[i].score(x[:, i, :], y[:, i]).mean()
            if baseline[i].drift_lvl == 2:
                drift_list.append(i)
        for i in drift_list:
            baseline[i].fit(x[:, i, :], y[:, i])
            baseline[i].reset()

        # dams
        drift_list = []
        not_drift_list = []
        for i in range(m):
            scores[1, t, i] = dams[i].score(x[:, i, :], y[:, i]).mean()
            if dams[i].drift_lvl == 2:
                drift_list.append(i)
            else:
                not_drift_list.append(i)
        for i in drift_list:
            training_set = [j for j in not_drift_list]
            training_set.append(i)
            sample_weight = []
            for j in not_drift_list:
                sample_weight.append(dams[j].score(x[:, i, :], y[:, i],
                                                   score_only=True))
            sample_weight.append(np.ones(w_size))
            sample_weight = np.hstack(sample_weight)
            dams[i].fit(x[:, training_set, :].reshape(-1, d),
                        y[:, training_set].reshape(-1),
                        sample_weight=sample_weight)
            dams[i].reset()

        # multi-output methods
        temp = (rc.predict(x.reshape(-1, m*d)) - y) ** 2
        scores[2, t, :] = temp.mean(axis=0)
        rc.fit(x.reshape(-1, m*d), y)
        temp = (ist.predict(x.reshape(-1, m*d)) - y) ** 2
        scores[3, t, :] = temp.mean(axis=0)
        ist.fit(x.reshape(-1, m*d), y)
        temp = (sstht.predict(x.reshape(-1, m*d)) - y) ** 2
        scores[4, t, :] = temp.mean(axis=0)
        sstht.fit(x.reshape(-1, m*d), y)
    return scores


if __name__ == "__main__":
    for streams in ['train', 'weather', 'sensor']:
        for w_size in [10, 20, 50, 100]:
            print('Evaluate on {} with batch size {}...'.format(streams,
                                                                w_size))
            scores = evaluate(streams, 500, w_size)
            filename = 'results/{}.{:03d}.npy'.format(streams, w_size)
            with open(filename, 'wb') as f:
                np.save(f, scores)
