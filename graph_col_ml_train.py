import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
device = "cuda" if torch.cuda.is_available() else "cpu"

class MinGraphColModel(nn.Module):
    def __init__(self):
        super(MinGraphColModel, self).__init__()
        self.fc0 = nn.Linear(45, 100)
        self.bnorm0 = nn.BatchNorm1d(num_features=100)
        self.fc1 = nn.Linear(100, 50)
        self.bnorm1 = nn.BatchNorm1d(num_features=50)
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        x = F.relu(self.bnorm0(self.fc0(x)))
        x = F.relu(self.bnorm1(self.fc1(x)))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(model, optimizer, train_loader):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, labels.long())
        loss.backward()
        optimizer.step()

def validate(loader, model):
    model.eval()
    loss = 0
    corr = 0
    with torch.no_grad():
        for x, y in loader:
            output = model(x)
            loss += F.nll_loss(output, y.long()).item()
            pred = output.max(1, keepdim=True)[1]
            corr += pred.eq(y.view_as(pred)).sum()

    loss /= len(loader.dataset)
    acc = 100. * (corr / len(loader.dataset))

    return float(loss), float(acc)

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

def dnn():
    batch_size = 64

    # load data
    graphs = np.loadtxt('ml_data\data\graphs.txt', dtype='float')
    labels = np.loadtxt('ml_data\data\labels.txt', dtype='int') - 1

    for i in range(10):
        rand = np.random.get_state()
        np.random.shuffle(graphs)
        np.random.set_state(rand)
        np.random.shuffle(labels)

    sep = len(graphs) - 5000
    data_x = graphs[:sep]
    data_y = labels[:sep]
    test_x = graphs[sep:]
    test_y = labels[sep:]

    # seperate to train and validation
    sep = int(0.8 * len(data_x))
    train_x = torch.from_numpy(data_x[:sep]).float().to(device)
    train_y = torch.from_numpy(data_y[:sep]).long().to(device)
    val_x = torch.from_numpy(data_x[sep:]).float().to(device)
    val_y = torch.from_numpy(data_y[sep:]).long().to(device)
    test_x = torch.from_numpy(test_x).float().to(device)
    test_y = torch.from_numpy(test_y).long().to(device)

    train_dataset = TensorDataset(train_x, train_y)
    val_dataset = TensorDataset(val_x, val_y)
    test_dataset = TensorDataset(test_x, test_y)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    new_model = MinGraphColModel().to(device)
    new_opt = optim.Adam(new_model.parameters(), lr=0.001)

    best_acc = 0
    torch.save(new_model.state_dict(), './best_model.pt')

    for epoch in range(1, 13 + 1):
        train(new_model, new_opt, train_loader)
        val_loss, val_acc = validate(val_loader, new_model)
        print(f'epoch {epoch}: accuracy = {int(val_acc)}%')
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(new_model.state_dict(), './best_model.pt')
    new_model.load_state_dict(torch.load('./best_model.pt'))

    pred = test(test_loader, new_model)
    corr = 0
    for i in range(len(test_x)):
        if test_y[i] == pred[i]:
            corr = corr + 1
    print('accuracy: {:.2f}%'.format(100.0 * (corr / len(test_x))))