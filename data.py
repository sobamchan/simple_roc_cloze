from pathlib import Path
import pandas as pd
from nltk import word_tokenize

from torch.utils import data
from torch.utils.data.sampler import RandomSampler


def prepro(s):
    return ' '.join(word_tokenize(s)).lower()


class ValidationFormatData(data.Dataset):

    def __init__(self, dpath):
        df = pd.read_csv(dpath)
        s1s = [prepro(s)
               for s in df.InputSentence1.values.tolist()]
        s2s = [prepro(s)
               for s in df.InputSentence2.values.tolist()]
        s3s = [prepro(s)
               for s in df.InputSentence3.values.tolist()]
        s4s = [prepro(s)
               for s in df.InputSentence4.values.tolist()]
        self.stories = list(zip(s1s, s2s, s3s, s4s))

        opt1s = [prepro(s)
                 for s in df.RandomFifthSentenceQuiz1]
        opt2s = [prepro(s)
                 for s in df.RandomFifthSentenceQuiz2]
        self.opts = list(zip(opt1s, opt2s))

        self.answers = df.AnswerRightEnding.values.tolist()

    def __len__(self):
        return len(self.answers)

    def __getitem__(self, idx):
        return {
                'stories': self.stories[idx],
                'options': self.opts[idx],
                'answers': self.answers[idx],
                }


def get_loader(data_dir, batch_size):
    data_dir = Path(data_dir)
    train_ds = ValidationFormatData(data_dir / 'train.csv')
    test_ds = ValidationFormatData(data_dir / 'test.csv')

    train_loader = data.DataLoader(train_ds,
                                   batch_size=batch_size,
                                   sampler=RandomSampler(train_ds))
    test_loader = data.DataLoader(test_ds,
                                  batch_size=batch_size,
                                  sampler=RandomSampler(test_ds))

    return train_loader, test_loader
