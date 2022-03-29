import json
import random
import torch
import numpy as np
import os
import wandb

def read_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
    return data

def get_arguments(args):
    model_args = AttrDict(args.ModelArguments)
    training_args = AttrDict(args.TrainingArguments)
    data_args = AttrDict(args.DataArguments)
    logging_args = AttrDict(args.LoggingArguments)
    return model_args, training_args, data_args, logging_args

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        
def wandb_login(project_name):
    os.environ["WANDB_PROJECT"] = project_name
    wandb.login()

def set_seeds(seed=42):
    """ A function that fixes a random seed for reproducibility """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # for faster training, but not deterministic