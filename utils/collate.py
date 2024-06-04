import torch
from math import floor, ceil
from torch.nn import ConstantPad2d


def collate_by_center_padding(maxrows):
    def collate(batch):
        tensors = [d[0] for d in batch]
        labels = [d[1] for d in batch]
        for i in range(len(tensors)):
            tensors[i] = ConstantPad2d(((0,0,floor((maxrows-tensors[i].shape[0])/2),ceil((maxrows-tensors[i].shape[0])/2))),0.0)(tensors[i])
        tensors = torch.stack(tensors,dim=0)
        labels = torch.Tensor(labels).to(dtype=torch.int64)
        return tensors, labels
    return collate


def collate_by_end_padding(maxrows):
    def collate(batch):
        tensors = [d[0] for d in batch]
        labels = [d[1] for d in batch]
        for i in range(len(tensors)):
            tensors[i] = ConstantPad2d((0,0,0,maxrows-tensors[i].shape[0]),0.0)(tensors[i])
        tensors = torch.stack(tensors,dim=0)
        labels = torch.Tensor(labels).to(dtype=torch.int64)
        return tensors, labels
    return collate


def collate_by_stripping_pen_tail(cols_i, cols_j):
    def collate(batch):
        tensors = [d[0] for d in batch]
        labels = [d[1] for d in batch]
        for i in range(len(tensors)):
            tensors[i] = tensors[i][:,cols_i:cols_j+1]
        tensors = torch.stack(tensors,dim=0)
        labels = torch.Tensor(labels).to(dtype=torch.int64)
        return tensors, labels
    return collate
