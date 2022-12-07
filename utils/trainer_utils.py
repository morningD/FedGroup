import json
import torch
import numpy as np
import random
from utils.model_utils import calculate_model_state_difference
from termcolor import colored

'''
    Read the config of trainer,
    The type of trainer contain: fedavg, fedgroup, ...
'''

def read_json_config(path, dsname, model, trainer):
    with open(path, 'r') as fh:
        cfg = json.load(fh)

    config = {}

    dsname, model, trainer = dsname.lower(), model.lower(), trainer.lower()
    config['trainer'] = __refine_config(cfg['trainer_config'], dsname, model, trainer)
    config['client'] = __refine_config(cfg['client_config'], dsname, model, trainer)
    if trainer in ['ifca', 'fesem', 'fedgroup']:
        config['group'] = __refine_config(cfg['group_config'], dsname, model, trainer)
    return config

# This function is boring, don't read it.
def __refine_config(config, dsname, model, trainer):
    # For example, synthetic(0,0), synthetic(1,1), ... -> synthetic
    def __find_match_config(full_name, keys):
        best_match_len = -1
        best_match = None
        for key in keys:
            if key in full_name:
                if len(key) > best_match_len:
                    best_match = key
        return best_match

    # common settings
    if 'common' in config:
        # Find the match dataset name and trainer name
        match_dsname = __find_match_config(dsname, config.keys())
        match_trainer = __find_match_config(trainer, config.keys())
        # Add common configuration first
        refined_config = config['common']

        if match_trainer:
            # Secondlly, add trainer configuration (if exist)
            trainer_config = config[match_trainer]
            refined_config.update(trainer_config)

        if match_dsname:
            # Try to find the model config in dataset config
            match_model = __find_match_config(model, config[match_dsname].keys())
            model_config = None
            if match_model:
                model_config = config[match_dsname].pop(match_model)
            dataset_config = config[match_dsname]
            # Thirdly, add dataset configuration (if exist)
            refined_config.update(dataset_config)
            # Finally, add model configuration (if exist)
            if model_config:
                refined_config.update(model_config)
        
        return refined_config
    else:
        return None

def fix_seed(seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

# Defind the model boardcast procedure of federated traning
def federated_boardcast(server, clients, actor_type='NNActor'):
    if actor_type == 'NNActor':
        for c in clients:
            t0_global_model = server.local_actor.state['latest_params']
            # The selected client calculate the latest_update (This may be many rounds apart)
            c.state['latest_updates'] = calculate_model_state_difference(c.state['latest_params'], t0_global_model)
            # Set model parameters of clients to global model parameters
            c.state['latest_params'] = t0_global_model

# This function disable gradient calculation
@torch.no_grad()
def federated_averaging_aggregate(server, gradients, nks):
    actor_type = server.local_actor.model.actor_type
    
    # We can define difference aggregation strategy for difference actor type
    if actor_type == 'NNActor':
        agg_gradient_dict = __weighted_aggregate_model_dict(gradients, nks)
        server.local_actor.apply_gradient(agg_gradient_dict)
        return 

def __weighted_aggregate_model_dict(gradients, weights):
    # Aggregate the updates according their weights
    normalws = np.array(weights, dtype=float) / np.sum(weights, dtype=np.float)
    agg_gradient_dict = {}
    
    ''' We dont aggregate the BN layer, Ref: https://arxiv.org/abs/2102.07623 '''
    param_names = __layer_names_filter(list(gradients[0].keys()), filter_type='BN')

    for name in param_names:
        param_values = [g[name].detach().cpu().numpy() for g in gradients]
        weighted_param_values = np.sum([pv * weight for pv, weight in zip(param_values, normalws)], axis=0)
        agg_gradient_dict[name] = torch.from_numpy(weighted_param_values)
    return agg_gradient_dict

# Filter some unwant aggregate layer names
def __layer_names_filter(layer_names, filter_type):
    names_to_filter = []
    if filter_type == 'BN':
        for name in layer_names:
            if 'running_mean' in name:
                names_to_filter += [
                    name,
                    name.replace('running_mean', 'num_batches_tracked'),
                    name.replace('running_mean', 'running_var'),
                    name.replace('running_mean', 'weight'),
                    name.replace('running_mean', 'bias'),
                ]

    retained_names = [name for name in layer_names if name not in names_to_filter]
    return retained_names

def summary_results(comm_round, train_results=None, test_results=None, actor_type='NNActor'):
    '''
    Inputs:
        The <train_results> are <test_results> list of client's training and testing statistical data
        See the <train>/<test> function of train/test actor for more informations
        For example: <train_results> of NNActor: 
            num_samples, train_scores, train_loss, t1_model_state, gradient, metric_names
    '''
    if actor_type == 'NNActor':
        
        if train_results:
            nks = [rst[0] for rst in train_results]
            # The -1 means we only consider the train scores of lastest epoch or step
            train_scores = [rst[1][-1] for rst in train_results]
            train_losses = [rst[2][-1] for rst in train_results]
            
            weighted_train_scores = np.average(train_scores, weights=nks, axis=0)
            weighted_train_loss = np.average(train_losses, weights=nks)
            
            print(colored(f'Round {comm_round}, Train Score: {weighted_train_scores},\
                    Train Loss: {round(weighted_train_loss, 4)}', 'blue', attrs=['reverse']))
        if test_results:
            nks = [rst[0] for rst in test_results]
            test_scores = [rst[1][-1] for rst in test_results]
            test_losses = [rst[2][-1] for rst in test_results]
            weighted_test_scores = np.average(test_scores, weights=nks, axis=0)
            weighted_test_loss = np.average(test_losses, weights=nks)
            print(colored(f'Round {comm_round}, Test ACC: {weighted_test_scores},\
                    Test Loss: {round(weighted_test_loss, 4)}', 'red', attrs=['reverse']))

