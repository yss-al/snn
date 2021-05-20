import numpy as np

from sdnn.layers import Linear
from sdnn.loss import MSELoss
from numpy.random import default_rng

rng = default_rng(0)
# print(rng.standard_normal(10))


def init_weight():
    # y0 dim: (1, 2)
    w1 = np.array([[0.2, 0.3],
                   [0.4, 0.2],
                   [0.3, 0.4]], dtype='f')
    # y1 dim: (1, 3)
    w2 = np.array([[0.2, 0.3, 0.4],
                   [0.4, 0.2, 0.3],
                   [0.3, 0.4, 0.2]], dtype='f')
    # y2 dim: (1, 3)
    w3 = np.array([[0.2, 0.3, 0.4],
                   [0.4, 0.2, 0.3]], dtype='f')
    # y3 dim: (1, 2)
    return w1, w2, w3


def get_input():
    return np.array([[1.0, 2.0]], dtype='f')


def get_label():
    return np.array([[0.4, 0.8]], dtype='f')


def get_model():

    w1, w2, w3 = init_weight()
    fc1 = Linear(2, 3)
    fc1.weights = w1
    fc2 = Linear(3, 3)
    fc2.weights = w2
    fc3 = Linear(3, 2)
    fc3.weights = w3

    return [fc1, fc2, fc3]


if __name__ == '__main__':
    x = get_input()
    y = get_label()
    print('input:')
    print(x)
    model = get_model()
    for m in model:
        y_hat = m.forward(x)
        x = y_hat
    print('pred:')
    print(y_hat)
    loss = MSELoss(y_hat, y)
    print('loss')
    print(loss.forward())

    print('grad: dL/dy') 
    dy_grad = loss.backward()
    print(dy_grad)
    for m in reversed(model):
        print(m._prev_act)
        print(m.weights)
        dy_grad, dW, _ = m.backward(dy_grad)
        print(dy_grad)

    for m in model:
        print(m.grad)
    
