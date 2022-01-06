import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from graph_col_ml_train import MinGraphColModel
from graph_col_ml_train import test

def graph_to_vec(g):
    vec = list()
    n = g.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            vec.append(g[i][j])
    return vec

def get_ml_model():
    model = MinGraphColModel()
    model.load_state_dict(torch.load('./ml_data/graph_col_model.pt'))
    return model

def predict_col_min(g, model):
  x = torch.from_numpy(np.array([graph_to_vec(g)], dtype='float')).float()
  dataset = TensorDataset(x, torch.from_numpy(np.zeros(len(x))).long())
  loader = DataLoader(dataset, batch_size=1)
  k_pred = test(loader, model)[0] + 1
  return k_pred