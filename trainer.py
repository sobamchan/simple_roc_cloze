from data import get_loader
from model import FTMLP_SUM

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


class Trainer:

    def __init__(self, data_dir, batch_size, ft_path, lr, use_cuda=True):
        train_loader, test_loader = get_loader(data_dir, batch_size)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.use_cuda = use_cuda

        model = FTMLP_SUM(ft_path, use_cuda=use_cuda)
        if use_cuda:
            model.cuda()
        optimizer = optim.SGD(model.parameters(), lr=lr)
        self.model = model
        self.optimizer = optimizer
        self.criterion = nn.MSELoss()

    def train_one_epoch(self, iepoch):
        losses = []
        for batch in self.train_loader:
            y = self.model(batch['stories'], batch['options'])
            target = Variable(batch['answers'].float())
            if self.use_cuda:
                target = target.cuda()

            loss = self.criterion(y, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.data.tolist()[0])
