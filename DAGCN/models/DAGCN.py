#!/usr/bin/python
# -*- coding:utf-8 -*-
from torch import nn
import warnings
import torch
from models.MRF_GCN import MRF_GCN
from models.CNN import CNN



class DAGCN_features(nn.Module):
    def __init__(self, pretrained=False):
        super(DAGCN_features, self).__init__()
        self.model_cnn = CNN(pretrained)
        self.model_GCN = MRF_GCN(pretrained)

        self.__in_features = 256*1

    def forward(self, x):
        x1 = self.model_cnn(x)
        x2 = self.model_GCN(x1)
        return x2

    def output_num(self):
        return self.__in_features