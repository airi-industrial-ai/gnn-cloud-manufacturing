import torch
import numpy as np
import torch.nn as nn
from glob import glob
import torch.nn.functional as F

from torch.utils.data import Dataset
from dgl.dataloading import GraphDataLoader
from sklearn.model_selection import train_test_split

from cloudmanufacturing.graph import dglgraph

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

def dglgraph_fixed(graph, oper_max=20):
        '''
        Create padding for features on graph edges
        it's necessary because problems have different dimentions
        '''
        ncolumns = graph.ndata['feat']['o'].shape[1]
        graph.ndata['feat'] = {'o': F.pad(graph.ndata['feat']['o'], [0, oper_max - ncolumns])}
        return graph

class GraphDataset(Dataset):
    def __init__(self, problems, gammas, deltas):
        self.problems = problems
        self.gammas = gammas
        self.deltas = deltas

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx):
        problem = self.problems[idx]
        gamma = self.gammas[idx]
        delta = self.deltas[idx]
        graph = dglgraph(problem, gamma, delta)
        graph = dglgraph_fixed(graph)
        return graph.to(device)

def create_info_file(path_to_solutions, problems):
    info = {}
    for idx in [int(i.split('_')[-2]) for i in glob(f'{path_to_solutions}/*') if 'op' in i]:
        info[idx] = {}
        info[idx]['gamma'] = np.load(f'{path_to_solutions}/gamma_{idx}_op.npy')
        info[idx]['delta'] = np.load(f'{path_to_solutions}/delta_{idx}_op.npy')
        info[idx]['problem'] = problems[idx]
    return info

def create_graph_dataset(info_file, train_size=0.8):
    train_idx, test_idx = train_test_split(
         list(info_file.keys()),
         random_state=42,train_size=train_size
    )
    print('train size: ', len(train_idx))
    print('test size: ', len(test_idx))

    train_dataset = GraphDataset([info_file[i]['problem'] for i in train_idx],
                             [info_file[i]['gamma'] for i in train_idx],
                             [info_file[i]['delta'] for i in train_idx])

    test_dataset = GraphDataset([info_file[i]['problem'] for i in test_idx],
                                [info_file[i]['gamma'] for i in test_idx],
                                [info_file[i]['delta'] for i in test_idx])
    return train_dataset, test_dataset

def create_dataloader(train_dataset, test_dataset,
                      train_batch=100, test_batch=20):
    train_loader = GraphDataLoader(
        train_dataset, batch_size=train_batch, shuffle=True
    )
    test_loader = GraphDataLoader(
        test_dataset, batch_size=test_batch, shuffle=True
    )
    return train_loader, test_loader

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss)

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss