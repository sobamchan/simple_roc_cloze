import argparse

from data import get_loader
from logger import Logger


def getargs():
    p = argparse.ArgumentParser()
    p.add_argument('--odir', type=str)
    p.add_argument('--ddir', type=str, default='../DATA/ROC')
    p.add_argument('--ftpath', type=str, default='../DATA/wiki.en.bin')

    return p.parse_args()


def main(args, logger):
    _, test_loader = get_loader(args.ddir, int(1e+10))
    ds = test_loader.dataset
    stories = ds.stories
    opts = ds.stories
    answer = Variable(ds.answers.long())


if __name__ == '__main__':
    args = getargs()
    logger = Logger(args.odir)

    logger.log(str(args))

    main(args, logger)
