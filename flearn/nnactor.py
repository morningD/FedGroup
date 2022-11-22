from flearn.actor import Actor
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from collections import OrderedDict
import numpy as np

'''
DNN Actor with torch train and test functions
'''
class NNActor(Actor):
    def __init__(self, id, actor_type:str='base_nn', data_dict:dict={}, 
                model:nn.Module=None, optimizer:optim=None, loss_fn=None, metric_fns=[]):
        super(NNActor, self).__init__(id, actor_type)
        self.data_dict = data_dict
        self.model = model
        self.optimizer = optimizer(self.model.parameters(), lr=0.0001, momentum=0.9)
        self.loss_fn = loss_fn()
        self.metric_fns = metric_fns
        self.state.update({'init_params': None, 'latest_params': None, 'latest_updates': None, 
                            'local_soln': None, 'local_gradient': None, 'optimizer': None, 'scores': []})

        # Because mantain the whole model for each client is expensive,
        # so we share the model object and just save the model state like dataloaders and state dict
        
        self.__preprocess()

    def __preprocess(self):
        if 'train' in self.data_dict:
            self.train_loader = self.data_dict['train']
            self.check_trainable()
        if 'test' in self.data_dict:
            self.test_loader = self.data_dict['test']
            self.check_testable()

        self.state['init_params'] = deepcopy(self.model.state_dict())
        self.state['latest_params'] = deepcopy(self.model.state_dict())
        self.state['optimizer'] = deepcopy(self.optimizer.state_dict())


    def train(self):
        self.check_trainable()
    
    def solve_epochs(self, num_epochs:int=1, pretrain:bool=False):
        # Set the train mode flag
        self.model.train()
        train_scores, train_loss = [], []
        num_samples = len(self.train_loader.dataset)
        
        # Load model state and optimizer state from this client
        self.__load_state()
        t0_model_state = self.state['latest_params']

        for epoch in range(num_epochs):
            loss_sum = 0
            preds_list, targets_list = [], []
            for i, data in enumerate(self.train_loader, 0):
                # Get the training data, the data is a list of [inputs, labels]
                inputs, labels = data
                # zero the paremeter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels.long())

                loss_sum += loss.item()
                pred = outputs.data.max(1)[1]
                preds_list.append(pred.numpy())
                targets_list.append(labels.numpy())

                loss.backward()
                self.optimizer.step()

            loss_mean = loss_sum / num_samples
            y_true, y_pred = np.hstack(targets_list), np.hstack(preds_list)
            scores = tuple([mfn(y_true, y_pred) for mfn in self.metric_fns])

            train_loss.append(loss_mean)
            train_scores.append(scores)

        t1_model_state = deepcopy(self.model.state_dict())
        gradient = OrderedDict({k: v1-v0 for k, v0, v1 in zip(t0_model_state.keys(), t0_model_state.values(), t1_model_state.values())})

        if pretrain == True:
            # We don't update local solution and gradient if in pretrain mode
            pass
        else:
            self.state['local_soln'] = t1_model_state
            self.state['local_gradient'] = gradient

        return num_samples, train_scores, train_loss, t1_model_state, gradient

    def solve_steps(self, num_steps:int=1, pretrain:bool=False):
        # Set the train mode flag
        self.model.train()
        train_scores, train_loss = [], []
        num_samples = len(self.train_loader.dataset)
        
        # Load model state and optimizer state from this client
        self.__load_state()
        t0_model_state = self.state['latest_params']
        step = 0
        
        while(step < num_steps):
            loss_sum = 0
            preds_list, targets_list = [], []
            for i, data in enumerate(self.train_loader, 0):
                # Get the training data, the data is a list of [inputs, labels]
                inputs, labels = data
                # zero the paremeter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels.long())

                loss_sum += loss.item()
                pred = outputs.data.max(1)[1]
                preds_list.append(pred.numpy())
                targets_list.append(labels.numpy())

                loss.backward()
                self.optimizer.step()
                step += 1
                if step == num_steps: 
                    break

        y_true, y_pred = np.hstack(targets_list), np.hstack(preds_list)
        loss_mean = loss_sum / y_true.size

        train_loss.append(loss_mean)
        scores = tuple([mfn(y_true, y_pred) for mfn in self.metric_fns])
        train_scores.append(scores)

        t1_model_state = deepcopy(self.model.state_dict())
        gradient = OrderedDict({k: v1-v0 for k, v0, v1 in zip(t0_model_state.keys(), t0_model_state.values(), t1_model_state.values())})

        if pretrain == True:
            # We don't update local solution and gradient if in pretrain mode
            pass
        else:
            self.state['local_soln'] = t1_model_state
            self.state['local_gradient'] = gradient

        return num_samples, train_scores, train_loss, t1_model_state, gradient

    def test(self):
        pass

    def test_locally(self):
        pass

    def check_trainable(self):
        pass

    def check_testable(self):
        pass

    def __load_state(self):
        self.model.load_state_dict(self.state['latest_params'])
        self.optimizer.load_state_dict(self.state['optimizer'])

    def __difference_state(): pass
