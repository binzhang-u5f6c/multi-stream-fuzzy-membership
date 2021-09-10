# -*- coding: utf-8 -*-
"""Test #2 & #3.

Evaluate multiple methods on two real-world data sets.
"""
from copy import deepcopy as dcp

import numpy as np
import matplotlib.pyplot as plt

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


def evaluate(dataset, training_size, w_size, random_state=None):
    """Evaluate a model on a dataset."""
    # loading data
    streams, m, d = get_streams(dataset)
    x_train, y_train = streams[:training_size]

    # initialize models
    baseline = [dcp(StreamHandler('tree', random_state=random_state))
                for _ in range(m)]
    dams = [dcp(StreamHandler('tree', random_state=random_state))
            for _ in range(m)]
    ht = [dcp(StreamHandler('ht', random_state=random_state))
          for _ in range(m)]
    hat = [dcp(StreamHandler('hat', random_state=random_state))
           for _ in range(m)]
    ist = StreamHandler('ist', random_state=random_state)
    for i in range(m):
        baseline[i].fit(x_train[:, i, :], y_train[:, i])
        dams[i].fit(x_train[:, i, :], y_train[:, i])
    ist.fit(x_train.reshape(-1, m*d), y_train)

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
                sample_weight_j = \
                        dams[j].score(x[:, i, :], y[:, i], score_only=True)
                sample_weight.append(np.ones(w_size)*sample_weight_j.mean())
            sample_weight.append(np.ones(w_size))
            sample_weight = np.hstack(sample_weight)
            dams[i].fit(x[:, training_set, :].reshape(-1, d),
                        y[:, training_set].reshape(-1),
                        sample_weight=sample_weight)
            dams[i].reset()

        # ht
        drift_list = []
        for i in range(m):
            scores[2, t, i] = ht[i].score(x[:, i, :], y[:, i]).mean()
            if ht[i].drift_lvl == 2:
                drift_list.append(i)
        for i in drift_list:
            ht[i].fit(x[:, i, :], y[:, i])
            ht[i].reset()

        # hat
        drift_list = []
        for i in range(m):
            scores[3, t, i] = hat[i].score(x[:, i, :], y[:, i]).mean()
            if hat[i].drift_lvl == 2:
                drift_list.append(i)
        for i in drift_list:
            hat[i].fit(x[:, i, :], y[:, i])
            hat[i].reset()

        # iSOUPTree
        scores[4, t, :] = ist.score(x.reshape(-1, m*d), y).mean(axis=0)
        if ist.drift_lvl == 2:
            ist.fit(x.reshape(-1, m*d), y)
    return scores.mean(axis=1)


if __name__ == "__main__":
    result_train = np.zeros((6, 5, 8, 5))
    result_sensor = np.zeros((6, 5, 6, 5))
    for i in range(5):
        rng = np.random.RandomState(i+1)
        for j, w_size in enumerate([10, 20, 50, 100, 200, 500]):
            print('Evaluate on train with batch size {}...'.format(w_size))
            result_train[j, :, :, i] = \
                evaluate('train', 500, w_size, random_state=rng)
            print('Evaluate on sensor with batch size {}...'.format(w_size))
            result_sensor[j, :, :, i] = \
                evaluate('sensor', 500, w_size, random_state=rng)

    # exp 2
    for i in range(8):
        plt.subplot(2, 4, i+1)
        plt.title('Stream{}'.format(i+1))
        plt.xlabel('Mini-batch size')
        plt.ylabel('mean squared error')
        plt.xticks(list(range(6)), [10, 20, 50, 100, 200, 500])
        plt.plot(result_train[:, 0, i, :].mean(axis=3),
                 '-*r', label='Baseline')
        plt.plot(result_train[:, 1, i, :].mean(axis=3), '-*b', label='KDAM')
        plt.legend()
    plt.subplots_adjust(top=2, bottom=0, left=0, right=4)
    plt.savefig('exp2train.png')
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.title('Stream{}'.format(i+1))
        plt.xlabel('Mini-batch size')
        plt.ylabel('mean squared error')
        plt.xticks(list(range(6)), [10, 20, 50, 100, 200, 500])
        plt.plot(result_sensor[:, 0, i, :].mean(axis=3),
                 '-*r', label='Baseline')
        plt.plot(result_sensor[:, 1, i, :].mean(axis=3), '-*b', label='KDAM')
        plt.legend()
    plt.subplots_adjust(top=2, bottom=0, left=0, right=3)
    plt.savefig('exp2sensor.png')

    # exp 3
    for i in range(8):
        print('{}'.format(i+1), end='&')
        print('{:.1f} $\\pm$ {:.1f}'.format(result_train[3, 2, i, :].mean(),
                                            result_train[3, 2, i, :].std()),
              end='&')
        print('{:.1f} $\\pm$ {:.1f}'.format(result_train[3, 3, i, :].mean(),
                                            result_train[3, 3, i, :].std()),
              end='&')
        print('{:.1f} $\\pm$ {:.1f}'.format(result_train[3, 4, i, :].mean(),
                                            result_train[3, 4, i, :].std()),
              end='&')
        print('{:.1f} $\\pm$ {:.1f}'.format(result_train[3, 0, i, :].mean(),
                                            result_train[3, 0, i, :].std()),
              end='&')
    for i in range(6):
        print('{}'.format(i+1), end='&')
        print('{:.1f} $\\pm$ {:.1f}'.format(result_sensor[3, 2, i, :].mean(),
                                            result_sensor[3, 2, i, :].std()),
              end='&')
        print('{:.1f} $\\pm$ {:.1f}'.format(result_sensor[3, 3, i, :].mean(),
                                            result_sensor[3, 3, i, :].std()),
              end='&')
        print('{:.1f} $\\pm$ {:.1f}'.format(result_sensor[3, 4, i, :].mean(),
                                            result_sensor[3, 4, i, :].std()),
              end='&')
        print('{:.1f} $\\pm$ {:.1f}'.format(result_sensor[3, 0, i, :].mean(),
                                            result_sensor[3, 0, i, :].std()),
              end='&')
