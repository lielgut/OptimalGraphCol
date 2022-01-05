import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

def graph_to_vec(g):
    vec = list()
    n = g.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            vec.append(g[i][j])
    return vec

class MinGraphColModel(nn.Module):
    def __init__(self):
        super(MinGraphColModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc0 = nn.Linear(45, 100)
        self.bnorm0 = nn.BatchNorm1d(num_features=100)
        self.fc1 = nn.Linear(100, 50)
        self.bnorm1 = nn.BatchNorm1d(num_features=50)
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.bnorm0(self.fc0(x)))
        x = F.relu(self.bnorm1(self.fc1(x)))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
def test(loader, model):
    all_pred = list()
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            output = model(x)
            pred = output.max(1, keepdim=True)[1]
            for y_hat in pred:
                all_pred.append(int(y_hat))
    return all_pred

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