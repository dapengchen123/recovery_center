from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable
from reid.evaluation import accuracy
from reid.loss import OIMLoss, TripletLoss
from utils.meters import AverageMeter


# mode decide how to train the model

class BaseTrainer(object):
    def __init__(self, cnn_model, rnn_model, criterion):
        super(BaseTrainer, self).__init__()
        self.cnn_model = cnn_model
        self.rnn_model = rnn_model
        self.criterion = criterion

    def train(self, epoch, data_loader, cnn_opt, rnn_opt, mode, print_freq=10):
        self.cnn_model.train()
        self.rnn_model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()

        for i, inputs in enumerate(data_loader):
            if inputs[2].size()[0] < data_loader.batch_size/2:
                print("too less batch {}".format(inputs[3].size()[0]))
                continue

            data_time.update(time.time() - end)
            inputs, targets = self._parse_data(inputs)
            loss, prec1 = self._forward(inputs, targets, mode)
            losses.update(loss.data[0], targets.size(0))
            precisions.update(prec1, targets.size(0))
            if mode == 'cnn' or mode == 'cnn_rnn':
                cnn_opt.zero_grad()
            if mode == 'cnn_rnn' or mode == 'fixcnn_rnn':
                rnn_opt.zero_grad()

            loss.backward()

            if mode == 'cnn' or mode == 'cnn_rnn':
                cnn_opt.step()
            if mode == 'cnn_rnn' or mode == 'fixcnn_rnn':
                rnn_opt.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets, mode):
        raise NotImplementedError


class SeqTrainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, flows, pids, camid = inputs
        inputs = [Variable(imgs), Variable(flows)]
        targets = Variable(pids).cuda()
        return inputs, targets

    def _forward(self, inputs, targets, mode):

        out_feat = self.cnn_model(inputs[0], inputs[1], mode)
        if mode == 'cnn_rnn' or mode == 'fixcnn_rnn':
            out_feat = self.rnn_model(out_feat)

        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            loss = self.criterion(out_feat, targets)
            prec, = accuracy(out_feat.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, OIMLoss):
            loss, outputs = self.criterion(out_feat, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, TripletLoss):
            loss, prec = self.criterion(out_feat, targets)
        else:
            raise ValueError("Unsupported loss:", self.criterion)

        return loss, prec

