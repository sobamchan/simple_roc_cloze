import numpy as np
import torch
import torch.nn as nn
from fastText import load_model
from torch.autograd import Variable


class FTMLP_SUM(nn.Module):

    def __init__(self, ft_path, nlayers=3,
                 nemb=300, nhidden=500, use_cuda=True):
        super().__init__()

        layers = []
        for i in range(nlayers):
            din = nemb * 2 if i == 0 else nhidden
            dout = 2 if i == (nlayers - 1) else nhidden
            layers.append(nn.Linear(din, dout))
            if i < (nlayers - 1):
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
        layers.append(nn.Softmax(1))

        print('Loading FastText binary...')
        self.ft = load_model(ft_path)

        self.use_cuda = use_cuda
        self.nlayers = nlayers
        self.nemb = nemb
        self.nhidden = nhidden
        self.layers = nn.Sequential(*layers)

    def forward(self, b_stories, b_options):
        bsz = len(b_stories)
        story_vec = np.zeros((bsz, self.nemb))
        option_vec = np.zeros((bsz, self.nemb))

        for i, stories in enumerate(b_stories):
            for story in stories:
                for w in story.split():
                    try:
                        story_vec[i] += self.ft.get_word_vector(w)
                    except KeyError:
                        story_vec[i] +=\
                            self.ft.get_word_vector('graph-out-of-vocab')
                story_vec[i] /= len(story.split())

        for i, options in enumerate(b_options):
            for option in options:
                for w in option.split():
                    try:
                        option_vec[i] += self.ft.get_word_vector(w)
                    except KeyError:
                        option_vec[i] +=\
                            self.ft.get_word_vector('graph-out-of-vocab')
                option_vec[i] /= len(option.split())

        x = np.concatenate((story_vec, option_vec), axis=1)
        x = torch.from_numpy(x).float()
        if self.use_cuda:
            x = x.cuda()
        x = Variable(x)
        y = self.layers(x)
        return y
