import numpy as np
import torch
import torch.nn as nn
from gensim.models import FastText


class FTMLP_SUM(nn.Module):

    def __init__(self, ft_path, nlayers=3,
                 nemb=300, nhidden=500, use_cuda=True):
        super(FTMLP_SUM, self).__init__()

        layers = []
        for i in range(nlayers):
            din = nemb * 2 if i == 0 else nhidden
            dout = 2 if i == (nlayers - 1) else nhidden
            layers.append(nn.Linear(din, dout))
            if i < (nlayers - 1):
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
        layers.append(nn.Softmax(1))

        self.ft = FastText.load_binary_data(ft_path)

        self.use_cuda = use_cuda
        self.nlayers = nlayers
        self.nemb = nemb
        self.nhidden = nhidden
        self.layers = nn.Sequential(*layers)

    def forward(self, stories, options):
        story_vec = np.zeros(self.nemb)
        option_vec = np.zeros(self.nemb)

        for story in stories:
            for w in story.split():
                try:
                    story_vec += self.ft.wv[w]
                except KeyError:
                    story_vec += self.ft.wv['graph-out-of-vocab']

        for option in options:
            for w in option.split():
                try:
                    option_vec += self.ft.wv[w]
                except KeyError:
                    option_vec += self.ft.wv['graph-out-of-vocab']

        x = np.concatenate((story_vec, option_vec))
        x = torch.from_numpy(x)
        if self.use_cuda:
            x = x.cuda()
        y = self.layers(x)
        return y
