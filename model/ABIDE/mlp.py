import torch.nn as nn
import torch.optim as optim

'''
class ABIDE_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.clser = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

    def forward(self, x):
        return self.clser(x)
'''
class ABIDE_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.clser = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(32, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(8, 2)
        )
        # The default loss function
        self.loss_fn = nn.CrossEntropyLoss
        # The default actor type
        self.actor_type = 'NNActor'

    def forward(self, x):
        return self.clser(x)

def construct_model():
    return ABIDE_MLP()