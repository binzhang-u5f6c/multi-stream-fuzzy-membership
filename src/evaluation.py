# -*- coding: utf-8 -*-
"""Evaluating methods on a specific dataset.

A method to evaluate multiple methods on a specific dataset.
"""
from copy import deepcopy as dcp

import numpy as np

from .loader import TrainStreams
from .loader import WeatherStreams
from .loader import SensorStreams
from .model import StreamHandler


def get_streams(stream_name):
    """Get a stream."""
    if stream_name == 'train':
        streams = TrainStreams()
        m = 8
    elif stream_name == 'weather':
        streams = WeatherStreams()
        m = 10
    elif stream_name == 'sensor':
        streams = SensorStreams()
        m = 6
    return streams, m


def evaluate(dataset, training_size, w_size):
    """Evaluate a model on a dataset."""
    # loading data
    streams, m = get_streams(dataset)
    x_train, y_train = streams[:training_size]

    # initialize models
    baseline = [dcp(StreamHandler('tree', w_size)) for _ in range(m)]
    dams = [dcp(StreamHandler('tree', w_size)) for _ in range(m)]
    for i in range(m):
        baseline[i].fit(x_train[:, i, :], y_train[:, i])
        dams[i].fit(x_train[:, i, :], y_train[:, i])

    # start
    n = len(streams)
    scores = np.zeros((2, n, m))
    for t in range(training_size, n):
        # get data
        x, y = streams[t]  # x: ndarray of size(m, d), y: ndarray of size (m)
        x_train, y_train = streams[t-w_size:t]
        # baseline
        drift_list = []
        for i in range(m):
            scores[0, t, i] = baseline[i].score(x[i, :].reshape(1, -1), y[i])
            if t % w_size == 0 and baseline[i].drift_lvl == 2:
                drift_list.append(i)
        for i in drift_list:
            baseline[i].fit(x_train[:, i, :], y_train[:, i])
            baseline[i].reset()
        # dams
        drift_list = []
        not_drift_list = []
        for i in range(m):
            scores[1, t, i] = dams[i].score(x[i, :].reshape(1, -1), y[i])
            if t % w_size == 0 and dams[i].drift_lvl == 2:
                drift_list.append(i)
            else:
                not_drift_list.append(i)
        for i in drift_list:
            training_set = [j for j in not_drift_list]
            training_set.append(i)
            sample_weight = []
            for j in not_drift_list:
                sample_weight.append(dams[j].score(x_train[:, i, :],
                                                   y_train[:, i],
                                                   score_only=True))
            training_n = len(training_set) * w_size
            sample_weight.append(np.ones(w_size))
            sample_weight = np.hstack(sample_weight)
            dams[i].fit(x_train[:, training_set, :].reshape(training_n, -1),
                        y_train[:, training_set].reshape(training_n),
                        sample_weight=sample_weight)
            dams[i].reset()
    return scores
