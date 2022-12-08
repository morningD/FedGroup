from torch.utils.data import ConcatDataset, DataLoader
from utils.trainer_utils import read_json_config, fix_seed, federated_boardcast, federated_averaging_aggregate, summary_results
from utils.read_data import read_federated_data
from utils.model_utils import calculate_model_state_difference
from pathlib import Path
import importlib
from flearn.server import Server
import random
import time

_cfg_path = str(Path(__file__).parent.parent.parent.joinpath('utils').joinpath('config_template.json'))

class FedAvg(object):
    def __init__(self, dsname, mname, cfg_path, actor_type='default'):
        # Read config from json file
        trainer = self.__class__.__name__
        cfg = read_json_config(cfg_path, dsname, mname, trainer)
        self.trainer_cfg, self.client_cfg, self.preprocess_cfg = cfg['trainer'], cfg['client'], cfg['preprocess']
        
        # Set configs as attribute
        for key, val in self.trainer_cfg.items(): 
            setattr(self, key, val)

        # Fix seed
        if hasattr(self, 'seed') != True: self.seed = 0 
        fix_seed(self.seed)

        # Create actors
        self.construct_actors(dsname, mname, actor_type, self.trainer_cfg, self.client_cfg, self.preprocess_cfg)

    def construct_actors(self, dsname, mname, actor_type, trainer_cfg, client_cfg, preprocess_cfg):
        ''' 1, Read dataset and construct dataloaders for each client '''
        train_loaders, val_loaders, test_loaders = read_federated_data(dsname, preprocess_cfg)
        self.num_train_clients, self.num_test_clients = len(train_loaders), len(test_loaders)

        ''' 2, Get model loader according to dataset and model name and construct the model '''
        # Set the model loader according to the dataset and model name
        model_path = f"model.{dsname.split('_')[0]}.{mname}"
        self.model_loader = importlib.import_module(model_path).construct_model
        # Construct the model. Note: we just maintain one model
        self.model = self.model_loader()
        print(self.model) # DEBUG

        ''' 3, Confirm the actor type and create actors (servers and clients) '''
        if actor_type == 'default':
            self.actor_type = self.model.actor_type

        actor_path = f'flearn.{self.actor_type.lower()}'
        actor_loader = getattr(importlib.import_module(actor_path), self.actor_type)
        common_data_dict = {'le': client_cfg['local_epochs'], 'lr': client_cfg['learning_rate']}
        
        # Create training and test clients
        if self.train_test_separate == False:
            self.train_clients = [actor_loader(id=cid, model=self.model, data_dict={
                                **common_data_dict, 'train': train_loaders[cid], 'test': test_loaders[cid]})
                                for cid in range(self.num_train_clients)]
            self.test_clients = self.train_clients
        else:
            self.train_clients = [actor_loader(id=cid, model=self.model, data_dict={
                                **common_data_dict, 'train': train_loaders[cid]})
                                for cid in range(self.num_train_clients)]
            self.test_clients = [actor_loader(id=int(cid+self.num_train_clients), model=self.model, data_dict={
                                **common_data_dict, 'test': test_loaders[cid]})
                                for cid in range(self.num_test_clients)]
        
        # Create Server and the local actor for Server
        sid = -1
        server_data_dict = {}
        if self.train_locally == True:
            # Train model on server, training data is the collection of all federated training data
            bs = next(iter(train_loaders[0]))[0].shape[0]
            train_datasets = [loader.dataset for loader in train_loaders]
            server_data_dict['train'] = DataLoader(ConcatDataset(train_datasets), batch_size=bs, shuffle=True) 
        if self.eval_locally == True:
            # Test on server, testing data is the collection of all federated testing data
            test_datasets = [loader.dataset for loader in test_loaders]
            server_data_dict['test'] = DataLoader(ConcatDataset(test_datasets), batch_size=64, shuffle=False)
        self.server = Server(id=sid, local_actor=actor_loader(sid, model=self.model, data_dict={**server_data_dict, **common_data_dict}))

        ''' 4, Set the uplink for clients and downlink for server '''
        self.server.add_downlink(self.train_clients+self.test_clients)
        self.server.refresh()
        for c in self.train_clients + self.test_clients:
            c.add_uplink(self.server)

    def train(self):
        for comm_round in range(self.num_rounds):
            ''' 0, Init time record '''
            train_time, test_time, agg_time = 0, 0, 0
            
            ''' 1, Random select train clients '''
            selected_clients = self.select_train_clients(comm_round, self.clients_per_round)

            ''' 2, Server broadcasts global model to selected clients '''
            federated_boardcast(self.server, selected_clients, actor_type=self.actor_type)

            ''' 3, Train selected clients or train server's local actor '''
            start_time = time.time()
            if self.train_locally == True:
                train_results = self.server.train_locally()
            else:
                train_results = self.server.train(selected_clients)
            train_time = round(time.time() - start_time, 3)

            if train_results == None: continue # Skip to next round
            
            ''' 4, Federated aggregate gradients '''
            nks, gradients = [rst[0] for rst in train_results], [rst[4] for rst in train_results]
            federated_averaging_aggregate(self.server, gradients, nks)

            ''' 5, Run testing '''
            test_results = None
            if comm_round % self.eval_every == 0 or comm_round == self.num_rounds-1:
                start_time = time.time()
                if self.eval_locally == True:
                    test_results = self.server.test_locally()
                else:
                    # Run test on all test clients
                    selected_test_clients = self.test_clients
                    test_results = self.server.test(selected_test_clients)
                test_time = round(time.time() - start_time, 3)

            ''' 6, Summary train and test results'''
            #print('score:', train_results[0][1])
            #print(f'comm round:{comm_round}, test results: {test_results}')
            summary_results(comm_round, train_results, test_results, self.actor_type)
            


    def select_train_clients(self, comm_round, num_clients=20):
        ''' selects num_clients clients weighted by number of samples from possible_clients
            This function is using random selection strategy, u can replace it with other strategies like active learnning
        
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(num_trainable_clients))
        
        Return:
            list of selected clients objects
        '''
        # Get all trainable nodes from server's downlink
        _, trainable_nodes = self.server.check_selected_trainable(self.server.downlink)
        num_clients = min(num_clients, len(trainable_nodes))
        if num_clients > 0:
            random.seed(comm_round+self.seed)  # make sure for each comparison, we are selecting the same clients each round
            selected_clients = random.sample(trainable_nodes, num_clients)
            random.seed(self.seed) # Restore the seed
            return selected_clients
        else:
            return []

trainer = FedAvg('ABIDE', 'mlp', _cfg_path)
trainer.train()
