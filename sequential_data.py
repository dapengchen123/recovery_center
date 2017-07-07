# system tool
from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import sys


# computation tool
import torch
import numpy as np


# device tool
import torch.backends.cudnn as cudnn



# utilis
from utils.logging import Logger
from utils.serialization import load_checkpoint, save_cnn_checkpoint, save_rnn_checkpoint
# reid
from reid.loss.oim import OIMLoss
from reid.loss.triplet import TripletLoss
from reid.data import get_data
from reid.model import ResNet
from reid.model import LSTMCC
from reid.train import SeqTrainer
from reid.evaluation import Evaluator

# 1, data



def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True


    # log file
    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))


    # Create data loaders
    if args.loss == 'triplet':
        assert args.num_instances > 1, 'TripletLoss requires num_instances > 1'
        assert args.batch_size % args.num_instances == 0, \
            'num_instances should divide batch_size'

    dataset, num_classes, train_loader, val_loader, test_loader = \
        get_data(args.dataset, args.split, args.data_dir,
             args.batch_size, args.seq_len, args.seq_srd,
                 args.workers, args.num_instances)


    # create CNN model
    cnn_model = ResNet(args.depth,  num_features=args.features, pretrained=True, dropout=args.dropout)
    cnn_model = torch.nn.DataParallel(cnn_model).cuda()

    # create RNN model
    if args.train_mode == 'cnn_rnn':
        rnn_input = cnn_model.module.feat.in_features
    else:
        rnn_input = args.features



    rnn_model = LSTMCC(cnn_model.module.feat.in_features, args.hidden, args.lstm_layer, args.lstm_drop)
    rnn_model.cuda()

    # Criterion model
    if args.loss == 'xentropy':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.loss == 'oim':
        if args.train_mode == 'cnn':
            hidden = args.features
        else:
            hidden = args.hidden
        criterion = OIMLoss(hidden, num_classes,
                            scalar=args.oim_scalar, momentum=args.oim_momentum)
    elif args.loss == 'triplet':
        criterion = TripletLoss(margin=args.triplet_margin)
    else:
        raise ValueError("Cannot recognize loss type:", args.loss)
    criterion.cuda()


    # Optimizer
    cnn_param_groups = cnn_model.parameters()
    cnn_optimizer = torch.optim.SGD(cnn_param_groups, lr=args.cnnlr, momentum=args.momentum,
                                     weight_decay=args.weight_decay,
                                     nesterov=True)

    def adjust_cnnlr(epoch):
        lr = args.cnnlr * (0.1 ** (epoch // 40))
        for g in cnn_optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    rnn_param_groups = rnn_model.parameters()
    rnn_optimizer = torch.optim.Adam(rnn_param_groups, lr=args.rnnlr, weight_decay=args.weight_decay)

    def adjust_rnnlr(epoch):
        lr = args.rnnlr if epoch <= 100 else args.rnnlr * (0.001 ** (epoch - 100) / 50)
        for g in rnn_optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)



    # Trainer
    trainer = SeqTrainer(cnn_model, rnn_model, criterion)

    # Evaluator
    evaluator = Evaluator(cnn_model, rnn_model)

    # Loading previous training model
    # Load from CNN checkpoint
    if args.train_mode == 'fixcnn_rnn':
        checkpoint = load_checkpoint(osp.join('pretrain/cnnmodel_best.pth.tar'))
        cnn_model.load_state_dict(checkpoint['state_dict'])
        for param in cnn_model.parameters():
            param.requires_grad = False


    if args.train_mode == 'cnn_rnn':
        checkpoint = load_checkpoint(osp.join('pretrain/cnnmodel_best.pth.tar'))
        cnn_model.load_state_dict(checkpoint['state_dict'])

    best_top1 = 0

    # Start training

    for epoch in range(args.start_epoch, args.epochs):
        if args.train_mode == 'cnn' or args.train_mode == 'cnn_rnn':
            adjust_cnnlr(epoch)
        if args.train_mode == 'cnn_rnn' or args.train_mode == 'fixcnn_rnn':
            adjust_rnnlr(epoch)
        trainer.train(epoch, train_loader, cnn_optimizer, rnn_optimizer, args.train_mode)

        if epoch % 3 == 0:
            top1 = evaluator.evaluate(test_loader,  args.train_mode)
            is_best = top1 > best_top1

            if args.train_mode == 'cnn_rnn' or args.train_mode == 'cnn':
                save_cnn_checkpoint({
                    'state_dict': cnn_model.state_dict(),
                    'epoch': epoch + 1,
                    'best_top1': best_top1,
                }, is_best, fpath=osp.join(args.logs_dir, 'cnn_checkpoint.pth.tar'))

            if args.train_mode == 'cnn_rnn' or args.train_mode == 'fixcnn_rnn':
                save_rnn_checkpoint({
                    'state_dict': cnn_model.state_dict(),
                    'epoch': epoch + 1,
                    'best_top1': best_top1,
                }, is_best, fpath=osp.join(args.logs_dir, 'rnn_checkpoint.pth.tar'))



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ID Training ResNet Model")

    # DATA
    parser.add_argument('-d', '--dataset', type=str, default='ilidsvidsequence',
                        choices=['ilidsvidsequence'])
    parser.add_argument('-b', '--batch-size', type=int, default=32)

    parser.add_argument('-j', '--workers', type=int, default=4)

    parser.add_argument('--seq_len', type=int, default=2)

    parser.add_argument('--seq_srd', type=int, default=2)

    parser.add_argument('--split', type=int, default=0)

    parser.add_argument('--num-instances', type=int, default=0,
                        help="To be useful in verification loss")


    # MODEL
     # CNN model
    parser.add_argument('--depth', type=int, default=50,
                        choices=[18, 34, 50, 101, 152])
    parser.add_argument('--features', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
     # LSTM model
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--lstm_layer', type=int, default=1)
    parser.add_argument('--lstm_drop', type=float, default=0.2)

     # Criterion model
    parser.add_argument('--loss', type=str, default='oim',
                        choices=['xentropy', 'oim', 'triplet'])
    parser.add_argument('--oim-scalar', type=float, default=30)
    parser.add_argument('--oim-momentum', type=float, default=0.5)
    parser.add_argument('--triplet-margin', type=float, default=0.5)

    # OPTIMIZER
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--cnnlr', type=float, default=0.001)
    parser.add_argument('--rnnlr', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--cnn_resume', type=str, default='', metavar='PATH')


    # TRAINER
    parser.add_argument('--train_mode', type=str, default='cnn_rnn',
                        choices=['cnn_rnn', 'cnn', 'fixcnn_rnn'])
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)

    # misc
    working_dir = osp.dirname(osp.abspath(__file__))

    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))




    # main function
    main(parser.parse_args())