import numpy as np


class Loss:
    def __init__(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass


class MSELoss(Loss):
    def __init__(self, pred, real):
        super(MSELoss, self).__init__()
        self.pred = pred
        self.real = real

    def forward(self):
        return np.power(self.pred - self.real, 2).mean()

    def backward(self):
        return 2 * (self.pred - self.real) / np.prod(self.pred.size)
