import os
import argparse

import torch

from trainer import Trainer
from logger import Logger


def getargs():
    p = argparse.ArgumentParser()
    p.add_argument('--odir', type=str)
    p.add_argument('--gpu-id', default=2)
    p.add_argument('--no-cuda', action='store_false')
    p.add_argument('--epochs', type=int, default=1000)

    p.add_argument('--ddir', type=str, default='../DATA/ROC')
    p.add_argument('--ftpath', type=str, default='../DATA/wiki.en.bin')

    p.add_argument('--bsz', type=int, default=32)
    p.add_argument('--lr', type=float, default=0.1)

    p.add_argument('--nlayers', type=int, default=3)
    p.add_argument('--nemb', type=int, default=300)
    p.add_argument('--nhidden', type=int, default=500)

    return p.parse_args()


def main(args, logger):
    t = Trainer(
            args.ddir,
            args.bsz,
            args.ftpath,
            args.nlayers,
            args.nemb,
            args.nhidden,
            args.lr,
            args.use_cuda
            )

    best_acc = -1
    for iepc in range(1, args.epochs + 1):
        logger.log('%dth epoch' % iepc)
        tr_loss = t.train_one_epoch(iepc)
        val_acc, val_loss = t.evaluate()

        if best_acc < val_acc:
            best_acc = val_acc
            logger.log('Best accuracy achived: %f!!!' % val_acc)
        # else:
        #     for pg in t.optimizer.param_groups:
        #         pg['lr'] *= 0.8
        #         lr = pg['lr']
        #     logger.log('Decrease lr to %f' % lr)

        logger.dump({
            'epoch': iepc,
            'tr_loss': tr_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            })


if __name__ == '__main__':
    args = getargs()
    logger = Logger(args.odir)

    # GPU usage
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    logger.log('using GPU id: %s' % os.environ['CUDA_VISIBLE_DEVICES'])
    if not args.no_cuda and torch.cuda.is_available():
        args.use_cuda = True
    else:
        args.use_cuda = False

    logger.log(str(args))

    main(args, logger)
