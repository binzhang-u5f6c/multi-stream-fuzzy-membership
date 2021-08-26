# -*- coding: utf-8 -*-
"""Test on Aug 1.

Compare our method with baseline (vanila DDM).
* Data arrive batch by batch
* Drift detection via DDM.
* Add data from other data streams with mse as weight.
* Only use the data in current time window (sliding window).
"""
import numpy as np

from src.evaluation import evaluate

for streams_name in ['train', 'weather', 'sensor']:
    print("Stream {}...".format(streams_name))
    print("  batch size of 10")
    scores = evaluate(streams_name, 500, 10)
    with open('results/{}010.npy'.format(streams_name), 'wb') as f:
        np.save(f, scores)
    print("  batch size of 20")
    scores = evaluate(streams_name, 500, 20)
    with open('results/{}020.npy'.format(streams_name), 'wb') as f:
        np.save(f, scores)
    print("  batch size of 30")
    scores = evaluate(streams_name, 500, 30)
    with open('results/{}030.npy'.format(streams_name), 'wb') as f:
        np.save(f, scores)
    print("  batch size of 50")
    scores = evaluate(streams_name, 500, 50)
    with open('results/{}050.npy'.format(streams_name), 'wb') as f:
        np.save(f, scores)
    print("  batch size of 100")
    scores = evaluate(streams_name, 500, 100)
    with open('results/{}100.npy'.format(streams_name), 'wb') as f:
        np.save(f, scores)
    print("  batch size of 200")
    scores = evaluate(streams_name, 500, 200)
    with open('results/{}200.npy'.format(streams_name), 'wb') as f:
        np.save(f, scores)
    print("  batch size of 500")
    scores = evaluate(streams_name, 500, 500)
    with open('results/{}500.npy'.format(streams_name), 'wb') as f:
        np.save(f, scores)
