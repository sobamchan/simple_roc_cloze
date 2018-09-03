import pandas as pd
from nltk import word_tokenize


def prepro(s):
    return ' '.join(word_tokenize(s))


class ValidationData:

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
