# -*- coding: utf-8 -*-
"""Test #3.

Generate some synthetic data and varify the method.
"""
import numpy as np

from sklearn.linear_model import Ridge


def generate_data(n=1000, d=10, rng=None):
    """Generate synthetic data."""
    if rng is None:
        rng = np.random.RandomState()
    w1 = rng.uniform(-10, 10, d)
    b1 = rng.uniform(-10, 10)
    x1 = rng.normal(0, 1, (n, d))
    y1 = np.dot(x1, w1) + b1 + rng.normal(0, 0.1, n)

    w2 = w1 + rng.normal(0, 0.1, d)
    b2 = b1 + rng.normal(0, 0.1)
    x2 = rng.normal(0, 1, (n, d))
    y2 = np.dot(x2, w2) + b2 + rng.normal(0, 0.1, n)

    drift_feature = rng.randint(d)
    w3 = w1 + rng.normal(0, 0.1, d)
    w3[drift_feature] += rng.normal(0, 3)
    b3 = b1 + rng.normal(0, 0.1)
    x3 = rng.normal(0, 1, (n, d))
    y3 = np.dot(x3, w3) + b3 + rng.normal(0, 0.1, n)

    w4 = w1 + rng.normal(0, 3, d)
    b4 = b1 + rng.normal(0, 0.1)
    x4 = rng.normal(0, 1, (n, d))
    y4 = np.dot(x4, w4) + b4 + rng.normal(0, 0.1, n)

    return x1, y1, x2, y2, x3, y3, x4, y4


if __name__ == "__main__":
    scores = np.zeros((12, 6, 5, 5))
    for i in range(12):
        # the ith stream
        rng = np.random.RandomState(i+1)
        x1, y1, x2, y2, x3, y3, x4, y4 = generate_data(rng=rng)
        rgr2 = Ridge(random_state=rng)
        rgr2.fit(x2[:900], y2[:900])
        mse2 = (rgr2.predict(x2[900:]) - y2[900:]) ** 2
        mean2, std2 = mse2.mean(), mse2.std()
        rgr3 = Ridge(random_state=rng)
        rgr3.fit(x3[:900], y3[:900])
        mse3 = (rgr3.predict(x3[900:]) - y3[900:]) ** 2
        mean3, std3 = mse3.mean(), mse3.std()
        rgr4 = Ridge(random_state=rng)
        rgr4.fit(x4[:900], y4[:900])
        mse4 = (rgr4.predict(x4[900:]) - y4[900:]) ** 2
        mean4, std4 = mse2.mean(), mse2.std()
        for j, batch_size in enumerate([10, 20, 50, 100, 200, 500]):
            # different mini-batch size
            for k in range(5):
                # repeat 5 times
                train_index = rng.randint(0, 900, batch_size)
                x_train, y_train = x1[train_index], y1[train_index]
                x_test, y_test = x1[900:], y1[900:]
                rgr1 = Ridge(random_state=rng)
                rgr1.fit(x_train, y_train)
                yhat = rgr1.predict(x_test)
                mse = np.abs((yhat - y_test) / y_test)
                scores[i, j, 0, k] = mse.mean()

                yhat2 = rgr2.predict(x_test)
                weight2 = (yhat2 - y_test) ** 2
                weight2 -= mean2
                if std2 != 0:
                    weight2 /= std2
                weight2 = 1 / (1 + np.exp(weight2))
                weight2 = np.ones(batch_size) * weight2.mean()
                yhat3 = rgr3.predict(x_test)
                weight3 = (yhat3 - y_test) ** 2
                weight3 -= mean3
                if std3 != 0:
                    weight3 /= std3
                weight3 = 1 / (1 + np.exp(weight3))
                weight3 = np.ones(batch_size) * weight3.mean()
                yhat4 = rgr4.predict(x_test)
                weight4 = (yhat4 - y_test) ** 2
                weight4 -= mean4
                if std4 != 0:
                    weight4 /= std4
                weight4 = 1 / (1 + np.exp(weight4))
                weight4 = np.ones(batch_size) * weight4.mean()

                weight1 = np.ones(batch_size)
                x_train2 = np.vstack((x_train, x2[train_index]))
                y_train2 = np.hstack((y_train, y2[train_index]))
                rgr1.fit(x_train2, y_train2,
                         sample_weight=np.hstack((weight1, weight2)))
                yhat = rgr1.predict(x_test)
                mse = np.abs((yhat - y_test) / y_test)
                scores[i, j, 1, k] = mse.mean()
                x_train3 = np.vstack((x_train, x3[train_index]))
                y_train3 = np.hstack((y_train, y3[train_index]))
                rgr1.fit(x_train3, y_train3,
                         sample_weight=np.hstack((weight1, weight3)))
                yhat = rgr1.predict(x_test)
                mse = np.abs((yhat - y_test) / y_test)
                scores[i, j, 2, k] = mse.mean()
                x_train4 = np.vstack((x_train, x4[train_index]))
                y_train4 = np.hstack((y_train, y4[train_index]))
                rgr1.fit(x_train4, y_train4,
                         sample_weight=np.hstack((weight1, weight4)))
                yhat = rgr1.predict(x_test)
                mse = np.abs((yhat - y_test) / y_test)
                scores[i, j, 3, k] = mse.mean()
                x_train5 = np.vstack((x_train, x2[train_index],
                                      x3[train_index], x4[train_index]))
                y_train5 = np.hstack((y_train, y2[train_index],
                                      y3[train_index], y4[train_index]))
                rgr1.fit(x_train5, y_train5,
                         sample_weight=np.hstack((weight1, weight2,
                                                  weight3, weight4)))
                yhat = rgr1.predict(x_test)
                mse = np.abs((yhat - y_test) / y_test)
                scores[i, j, 4, k] = mse.mean()

    # plot
    with open('result2.5.npy', 'wb') as f:
        np.save(f, scores)
