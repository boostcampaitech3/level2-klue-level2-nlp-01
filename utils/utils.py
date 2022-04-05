import json
import random
import torch
import numpy as np
import os

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
        
def wandb_init(logging_args):
    os.environ['WANDB_API_KEY'] = logging_args.WANDB_API_KEY
    os.environ['WANDB_ENTITY'] = logging_args.WANDB_ENTITY
    os.environ['WANDB_PROJECT'] = logging_args.WANDB_PROJECT
    os.environ['WANDB_NAME'] = logging_args.WANDB_NAME
    os.environ['WANDB_LOG_MODEL'] = logging_args.WANDB_LOG_MODEL

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