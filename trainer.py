import numpy as np
from data import get_loader
from model import FTMLP_SUM

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


class Trainer:

    def __init__(self, data_dir, batch_size, ft_path,
                 nlayers, nemb, nhidden, lr, optim_type='sgd', use_cuda=True):
        train_loader, test_loader = get_loader(data_dir, batch_size)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.use_cuda = use_cuda

        model = FTMLP_SUM(ft_path, nlayers, nemb, nhidden, use_cuda=use_cuda)
        if use_cuda:
            model.cuda()

        if optim_type == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=lr)
        elif optim_type == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optim_type == 'adagrad':
            optimizer = optim.Adagrad(model.parameters(), lr=lr)

        self.model = model
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss()

    def train_one_epoch(self, iepoch):
        self.model.train()
        losses = []
        for batch in self.train_loader:
            stories = list(zip(*batch['stories']))
            options = list(zip(*batch['options']))
            y = self.model(stories, options)
            target = Variable(batch['answers'].long())

            if self.use_cuda:
                target = target.cuda()

            loss = self.criterion(y, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.data.tolist()[0])
        return np.mean(losses)

    def evaluate(self):
        self.model.eval()
        accs = []
        losses = []
        for batch in self.test_loader:
            stories = list(zip(*batch['stories']))
            options = list(zip(*batch['options']))
            y = self.model(stories, options)
            target = Variable(batch['answers'].long())

            if self.use_cuda:
                target = target.cuda()

            _, pred = torch.max(y, 1)
            loss = self.criterion(y, target)
            losses.append(loss.data.tolist()[0])
            accs.append(torch.mean((pred == target).float()).data.tolist()[0])

        return np.mean(accs), np.mean(losses)
