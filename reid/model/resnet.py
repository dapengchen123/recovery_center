from __future__ import absolute_import
import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, \
    resnet152


class ResNet(nn.Module):
    __factory = {
        18: resnet18,
        34: resnet34,
        50: resnet50,
        101: resnet101,
        152: resnet152,
    }

    def __init__(self, depth, num_features=128, pretrained=True, dropout=0.5):
        super(ResNet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained

        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)

        ### At the bottom of CNN network

        conv0 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        init.kaiming_normal(conv0.weight, mode='fan_out')

        self.conv0 = conv0
        self.base = ResNet.__factory[depth](pretrained=pretrained)
        self.num_features = num_features
        self.dropout = dropout


        out_planes = self.base.fc.in_features
        # Append new layers
        self.feat = nn.Linear(out_planes, self.num_features)
        self.feat_bn = nn.BatchNorm1d(self.num_features)
        init.kaiming_normal(self.feat.weight, mode='fan_out')
        init.constant(self.feat.bias, 0)
        init.constant(self.feat_bn.weight, 1)
        init.constant(self.feat_bn.bias, 0)

        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)

    def forward(self, imgs, motions, mode):

        img_size = imgs.size()
        motion_size = motions.size()
        batch_sz = img_size[0]
        seq_len = img_size[1]
        imgs = imgs.view(-1, img_size[2], img_size[3], img_size[4])
        motions = motions.view(-1, motion_size[2], motion_size[3], motion_size[4])
        motions = motions[:, 1:3]

        for name, module in self.base._modules.items():
            if name == 'conv1':
                x = module(imgs)+self.conv0(motions)
                continue

            if name == 'avgpool':
                break

            x = module(x)



        ### average pooling ###
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        if mode == 'cnn_rnn':
            return x.view(batch_sz, seq_len, -1)

        x = self.feat(x)
        x = self.feat_bn(x)
        x = x / x.norm(2, 1).expand_as(x)

        if mode == 'cnn':
            if self.dropout > 0:
                x = self.drop(x)

        x = x.view(batch_sz, seq_len, self.num_features)

        if mode == 'cnn':
            return torch.squeeze(torch.mean(x, 1))

        return x