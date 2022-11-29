from random import shuffle
from utils.read_data import read_federated_data
import h5py
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import os
import numpy as np

from flearn.nnactor import NNActor
from flearn.server import Server
from model.ABIDE.mlp import construct_model
import torch.optim as optim
from utils.model_utils import accuracy, adjusted_balanced_accuracy, fscore

def main():

    train_loaders, _, test_loaders = read_federated_data('abide')
    '''
    y_test = np.array([1, 1, 1, 1, 1, 2])
    #preds = np.array([2, 2, 2, 2, 2, 2])
    preds = np.array([1, 1, 1, 1, 1, 1])
    print(f1_score(y_test, preds, pos_label=1))
    print(balanced_accuracy_score(y_test, preds, adjusted=True))
    '''

    actor = NNActor(0, data_dict={'train': train_loaders[1], 'test': test_loaders[2]}, model=construct_model(), optimizer=optim.Adam, loss_fn=nn.CrossEntropyLoss, metric_fns=[accuracy, adjusted_balanced_accuracy])
    
    sever = Server(1)
    print(sever.name)
    num, scores, loss, _, _ = actor.solve_epochs(num_epochs=30)
    print(num, scores)
    num_samples, scores, loss = actor.test_locally()
    print(num_samples, scores, loss)

'''
class ABIDE1Model(nn.Module):
    def __init__(self):
        super(ABIDE1Model, self).__init__()
        self.clser = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1128, 32), 
            nn.BatchNorm1d(32),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(32, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(8, 2)
        )
    
    def forward(self, x):
        return self.clser(x)

    

def main():
    base_path = '/home/duan/workspace/FedGroup/data'
    trainset = ABIDE1Dataset(base_path, train=True, transform=transforms.ToTensor())
    testset = ABIDE1Dataset(base_path, train=False, transform=transforms.ToTensor())
    print(len(trainset), len(testset))

    abide_train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
    abide_test_loader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

    net = ABIDE1Model().cuda()

    import torch.optim as optim
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    """Train"""
    def train(data_loader, optimizer, loss_fn):
        net.train()
        loss_all = 0
        total = 0
        correct = 0
        for i, data in enumerate(data_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)

            loss_all += loss.item()
            total += labels.size(0)
            pred = outputs.data.max(1)[1]
            correct += pred.eq(labels.view(-1)).sum().item()

            loss.backward()
            optimizer.step()

        return loss_all / len(data_loader), correct/total

    def test(data_loader, loss_fun):
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in data_loader:
                output = net(data)
                test_loss += loss_fun(output, target).item()
                pred = output.data.max(1)[1]
                correct += pred.eq(target.view(-1)).sum().item()
                total += target.size(0)

        test_loss /= len(data_loader)
        correct /= total
        print(' Test loss: {:.4f} | Test acc: {:.4f}'.format(test_loss, correct))
        
    for epoch in range(10):
        print(f"Epoch: {epoch}" )
        loss,acc = train(abide_train_loader, optimizer, loss_fn)
        #loss,acc = train(abide_test_loader, optimizer, loss_fn)
        print('Train loss: {:.4f} | Train acc : {:.4f}'.format(loss,acc))

    test(abide_test_loader, loss_fn)
'''
main()