import torch
import torch.nn as nn
import numpy as np

def set_seed(seed=0):
    seed = 42069  # set a random seed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

class CMlp(nn.Module):

    def __init__(self, r=(1., 1., 0.)):
        super(CMlp, self).__init__()
        w1, w2, w3 = init_weight()
        self.noise = [r[0], r[1] / r[0], 1 / r[1]]

        self.fc1 = nn.Linear(2, 3, False)
        self.fc1.weight.data = torch.from_numpy(w1)
        self.fc1.weight.data.mul_(self.noise[0])

        self.fc2 = nn.Linear(3, 3, False)
        self.fc2.weight.data = torch.from_numpy(w2)
        self.fc2.weight.data.mul_(self.noise[1])

        self.fc3 = nn.Linear(3, 2, False)
        self.fc3.weight.data = torch.from_numpy(w3)
        self.fc3.weight.data.mul_(self.noise[2])
        self.fc3.weight.data.add_(r[2])
        self.y2 = None

    def forward(self, x):
        y1 = self.fc1(x)
        self.y2 = self.fc2(y1)
        return self.fc3(self.y2)


if __name__ == '__main__':
    # setup gpu or cpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    r = (0.2, 0.4, 0.8)
    x = torch.tensor([[0.2, 0.3]], device=device)
    y_hat = torch.tensor([[0.5, 0.5]], device=device)

    net = CMlp().to(device)
    print('----------- plaintext weight ---------------')
    for p in net.parameters():
        print(p.data)
    y = net(x)
    print('y: ', y)
    criterion = nn.MSELoss()
    loss = criterion(y, y_hat)
    loss.backward()
    print('----------- plaintext grad -----------------')
    index = 0
    for p in net.parameters():
        if index == 2:
            w3 = p.grad
        print(p.grad)
        index += 1
    print('----------- ciphertext weight ---------------')

    net_c = CMlp(r).to(device)
    for p in net_c.parameters():
        print(p.data)
    y_c = net_c(x)
    print('yc: ', y_c)
    criterion = nn.MSELoss()
    print(y + net_c.y2.sum() * 0.8)


