from flearn.actor import Actor
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from collections import OrderedDict
import numpy as np
from utils.model_utils import calculate_model_state_difference

'''
DNN Actor with torch train and test functions
Args:
    data_dict -> Configurations dict of nnactor: {'train': train data loader, 'test': test data loader, \
        'le': local train epoch count, 'lr': learnning rate, 'bs': train batch size} # TODO: Add more settings support
'''
class NNActor(Actor):
    def __init__(self, id, actor_type:str='base_nn', data_dict:dict={}, 
                model:nn.Module=None, optimizer:optim=optim.AdamW, loss_fn=None, metric_fns=[]):
        super().__init__(id, actor_type)
        self.data_dict = data_dict
        self.model = model
        if optimizer is not None:
            #self.optimizer = optimizer(self.model.parameters(), lr=0.0001, momentum=0.9)
            self.optimizer = optimizer(self.model.parameters(), lr=self.data_dict['lr'])

        if loss_fn is not None:
            self.loss_fn = loss_fn()
        else:
            # We use the defalut loss fun defined in module class
            self.loss_fn = self.model.loss_fn()
    
        self.metric_fns = metric_fns
        self.state.update({'init_params': None, 'latest_params': None, 'latest_updates': None, 
                            'local_soln': None, 'local_gradient': None, 'optimizer': None,
                            'train_scores_history': [], 'train_loss_history': [], 
                            'test_scores_history': [], 'test_loss_history': [],
                            'step_count': 0})

        # Note Because mantain the whole model for each client is expensive,
        # so we share the model object and just save the model state like dataloaders and state dict
        
        self.__preprocess()

    def __preprocess(self):
        self.train_loader, self.test_loader = None, None
        if 'train' in self.data_dict:
            self.train_loader = self.data_dict['train']

            # Specify the train batch size if need
            if 'bs' in self.data_dict:
                try:
                    new_bs = self.data_dict['bs']
                    self.train_loader = torch.utils.data.DataLoader(self.train_loader.dataset, batch_size = new_bs, shuffle=True)
                except:
                    print('Batch Size ERROR.')

            # Specify the local epoch
            if 'le' in self.data_dict:
                self.local_epochs = self.data_dict['le']
            else:
                self.local_epochs = 1
            self.check_trainable()

        if 'test' in self.data_dict:
            self.test_loader = self.data_dict['test']
            self.check_testable()

        # Copy the parameters of initialized model and optimizer as begin model state
        self.state['init_params'] = deepcopy(self.model.state_dict())
        self.state['latest_params'] = deepcopy(self.model.state_dict())
        self.state['optimizer'] = deepcopy(self.optimizer.state_dict())

    # Training with locally dataset
    # Return: number of training samples, list of training scores, list of training loss, new model parameters, model updates
    def train_locally(self):
        num_samples, train_scores, train_loss, t1_model_state, gradient = self.solve_epochs(self.local_epochs)
        # Append the training history
        self.state['train_scores_history'] += train_scores
        self.state['train_loss_history'] += train_loss
        return num_samples, train_scores, train_loss, t1_model_state, gradient

    # The train procedure of NNActor is train locally
    def train(self):  
        if self.check_trainable() == False:
            return
        return self.train_locally()
    
    def solve_epochs(self, num_epochs:int=1, pretrain:bool=False):
        # Set the train mode flag
        self.model.train()
        train_scores, train_loss = [], []
        num_samples = len(self.train_loader.dataset)
        
        # Load model state and optimizer state from this client
        self.__load_state(is_train=True)
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
                self.state['step_count'] += 1

            loss_mean = loss_sum / num_samples
            y_true, y_pred = np.hstack(targets_list), np.hstack(preds_list)
            # Calculate score every epoch
            scores = tuple([mfn(y_true, y_pred) for mfn in self.metric_fns])

            train_loss.append(loss_mean)
            train_scores.append(scores)

        t1_model_state = deepcopy(self.model.state_dict())
        #gradient = OrderedDict({k: v1-v0 for k, v0, v1 in zip(t0_model_state.keys(), t0_model_state.values(), t1_model_state.values())})
        gradient = calculate_model_state_difference(t0_model_state, t1_model_state)

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
        self.__load_state(is_train=True)
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

        self.state['step_count'] += num_steps
        y_true, y_pred = np.hstack(targets_list), np.hstack(preds_list)
        loss_mean = loss_sum / y_true.size

        train_loss.append(loss_mean)
        # Calculate scores only after training finish
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
        if self.check_testable() == False:
            return
        num_samples, test_scores, test_loss = self.test_locally()
        self.state['test_scores_history'] += test_scores
        self.state['test_loss_history'] += test_loss
        return num_samples, test_scores, test_loss
    
    ''' Test on the local test dataset'''
    def test_locally(self):
        # Set the test mode flag
        self.model.eval()
        test_scores, test_loss = [], []
        num_samples = len(self.test_loader.dataset)
        
        # Load model state and optimizer state from this client
        self.__load_state(is_train=False)

        with torch.no_grad():
            loss_sum = 0
            preds_list, targets_list = [], []
            for inputs, labels in self.test_loader:
                # forward
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels.long())

                loss_sum += loss.item()
                pred = outputs.data.max(1)[1]
                preds_list.append(pred.numpy())
                targets_list.append(labels.numpy())

            loss_mean = loss_sum / num_samples
            y_true, y_pred = np.hstack(targets_list), np.hstack(preds_list)
            scores = tuple([mfn(y_true, y_pred) for mfn in self.metric_fns])
            print(y_true)
            print(y_pred)

            test_loss.append(loss_mean)
            test_scores.append(scores)

        return num_samples, test_scores, test_loss

        
    # Check whether local trainable
    def check_trainable(self):
        self.state['trainable'] = False
        if self.train_loader:
            # This actor has data to train
            self.state['train_size'] = len(self.train_loader.dataset)
            if self.state['train_size'] > 0:
                self.state['trainable'] = True
        else:
            self.state['train_size'] = 0
        
        return self.state['trainable']

    # Check whether local testable
    def check_testable(self):
        self.state['testable'] = False
        if self.test_loader:
            # This actor has data to test
            self.state['test_size'] = len(self.test_loader.dataset)
            if self.state['test_size'] > 0 and len(self.metric_fns) > 0:
                self.state['testable'] = True
        else:
            self.state['test_size'] = 0
        
        return self.state['testable']
    
    # Manually apply the gradient to the model parameters and fresh latest_update
    # It is useful for apply aggregated gradients
    @torch.no_grad()
    def apply_gradient(self, gradients):
        t0_params = deepcopy(self.state['latest_params'])
        for name in gradients:
            torch.add(self.state['latest_params'][name], gradients[name])
        self.state['latest_updates'] = calculate_model_state_difference(t0_params, self.state['latest_params'])
        del t0_params
        return
    
    def __load_state(self, is_train):
        if self.model is not None:
            self.model.load_state_dict(self.state['latest_params'])

        if self.optimizer is not None and is_train == True:
            # Only load optimizer state when training
            self.optimizer.load_state_dict(self.state['optimizer'])



    def __difference_state(): pass
