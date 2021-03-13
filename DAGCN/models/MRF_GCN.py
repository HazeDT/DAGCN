#!/usr/bin/python
# -*- coding:utf-8 -*-
from torch import nn
import warnings
import torch
from torch_geometric.nn import  ChebConv, BatchNorm
from torch_geometric.utils import dropout_adj


class GGL(torch.nn.Module):
    '''
    Grapg generation layer
    '''

    def __init__(self,):
        super(GGL, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(256,10),
            nn.Sigmoid())

    def forward(self, x):
        x = x.view(x.size(0), -1)
        atrr = self.layer(x)
        values, edge_index = Gen_edge(atrr)
        return values.view(-1), edge_index

def Gen_edge(atrr):
    atrr = atrr.cpu()
    A = torch.mm(atrr, atrr.T)
    maxval, maxind = A.max(axis=1)
    A_norm = A / maxval
    k = A.shape[0]
    values, indices = A_norm.topk(k, dim=1, largest=True, sorted=False)
    edge_index = torch.tensor([[],[]],dtype=torch.long)

    for i in range(indices.shape[0]):
        index_1 = torch.zeros(indices.shape[1],dtype=torch.long) + i
        index_2 = indices[i]
        sub_index = torch.stack([index_1,index_2])
        edge_index = torch.cat([edge_index,sub_index],axis=1)

    return values, edge_index

class MultiChev(torch.nn.Module):
    def __init__(self, in_channels,):
        super(MultiChev, self).__init__()
        self.scale_1 = ChebConv(in_channels,400,K=1)
        self.scale_2 = ChebConv(in_channels,400,K=2)
        self.scale_3 = ChebConv(in_channels,400,K=3)

    def forward(self, x, edge_index,edge_weight ):
        scale_1 = self.scale_1(x, edge_index,edge_weight )
        scale_2 = self.scale_2(x, edge_index,edge_weight )
        scale_3 = self.scale_3(x, edge_index,edge_weight )
        return torch.cat([scale_1,scale_2,scale_3],1)

class MultiChev_B(torch.nn.Module):
    def __init__(self, in_channels,):
        super(MultiChev_B, self).__init__()
        self.scale_1 = ChebConv(in_channels,100,K=1)
        self.scale_2 = ChebConv(in_channels,100,K=2)
        self.scale_3 = ChebConv(in_channels,100,K=3)
    def forward(self, x, edge_index,edge_weight ):
        scale_1 = self.scale_1(x, edge_index,edge_weight )
        scale_2 = self.scale_2(x, edge_index,edge_weight )
        scale_3 = self.scale_3(x, edge_index,edge_weight )
        return torch.cat([scale_1,scale_2,scale_3],1)



class MRF_GCN(nn.Module):
    '''
    This code is the implementation of MRF-GCN
    T. Li et al., "Multi-receptive Field Graph Convolutional Networks for Machine Fault Diagnosis"
    '''
    def __init__(self, pretrained=False, in_channel= 256, out_channel=10):
        super(MRF_GCN, self).__init__()
        if pretrained == True:
            warnings.warn("Pretrained model is not available")

        self.atrr = GGL()
        self.conv1 = MultiChev(in_channel)
        self.bn1 = BatchNorm(1200)
        self.conv2 = MultiChev_B(400 * 3)
        self.bn2 = BatchNorm(300)
        self.layer5 = nn.Sequential(
            nn.Linear(300, 256),
            nn.ReLU(inplace=True),
            nn.Dropout())


    def forward(self, x):

        edge_atrr, edge_index = self.atrr(x)
        edge_atrr = edge_atrr.cuda()
        edge_index = edge_index.cuda()
        edge_index, edge_atrr = dropout_adj(edge_index,edge_atrr)
        x = self.conv1(x, edge_index, edge_weight =  edge_atrr)
        x = self.bn1(x)
        x = self.conv2(x, edge_index, edge_weight =  edge_atrr)
        x = self.bn2(x)
        x = x.view(x.size(0), -1)
        x = self.layer5(x)
        return x