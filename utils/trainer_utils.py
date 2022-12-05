import json
import torch
import numpy as np
import random

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