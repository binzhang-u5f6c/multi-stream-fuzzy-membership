# -*- coding: utf-8 -*-
"""Test on Aug 1.

Compare our method with baseline (vanila DDM).
* Drift detection via DDM.
* Add data from other data streams if the erorr is small.
* Only use the data in current time window (sliding window).
"""
import numpy as np

from src.evaluation import evaluate

for streams_name in ['train', 'weather', 'sensor']:
    print("Stream {}...".format(streams_name))
    scores = evaluate(streams_name, 500, 25)
    with open('results/{}.npy'.format(streams_name), 'wb') as f:
        np.save(f, scores)
