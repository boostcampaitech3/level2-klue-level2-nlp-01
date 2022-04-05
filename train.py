import pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from utils import *
from ensemble import *
from ray import tune
from models import *
import wandb
import copy
import random

def train():
  set_seed(42)
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  tokenizer, model = TokenizerAndModelForKlueReTask()
  tokenizer2, model2 = TokenizerAndModelForKlueReTask()
  with open('./train_dataframe.pkl', 'rb') as f:
    dataset = pickle.load(f)
  
  print(device)

  kfold_train_dataset, kfold_train_label, kfold_valid_dataset, kfold_valid_label, test_dataset, test_label = KFold_split_dataset(dataset, num_folds=5, split_ratio=0.2)
  ensemble_first_step = StackingEnsembleFirstStep(kfold_train_dataset, kfold_train_label, kfold_valid_dataset, kfold_valid_label)
  
  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(
    output_dir='./results',          # output directory
    save_total_limit=5,
    num_train_epochs=1,              # total number of training epochs
    learning_rate=3e-05,               # learning_rate
    gradient_accumulation_steps=1,
    save_strategy='steps',
    per_device_train_batch_size=32,  # batch size per device during training
    per_device_eval_batch_size=32,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=100,              # log saving step.
    evaluation_strategy='steps', # evaluation strategy to adopt during training
    eval_steps = 500,            # evaluation step.
    load_best_model_at_end = True,
    report_to=['wandb'],
    metric_for_best_model='micro f1 score',
    greater_is_better=True
  )

  training_args2 = TrainingArguments(
    output_dir='./results',          # output directory
    save_total_limit=5,
    num_train_epochs=1,              # total number of training epochs
    learning_rate=2e-05,               # learning_rate
    gradient_accumulation_steps=2,
    save_strategy='steps',
    per_device_train_batch_size=32,  # batch size per device during training
    per_device_eval_batch_size=32,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=100,              # log saving step.
    evaluation_strategy='steps', # evaluation strategy to adopt during training
    eval_steps = 500,            # evaluation step.
    load_best_model_at_end = True,
    report_to=['wandb'],
    metric_for_best_model='micro f1 score',
    greater_is_better=True
  )
  ensemble_first_step.add_arguments(tokenizer, model, training_args, ImbalancedSamplerTrainer,
                                    tokenized_dataset, RE_Dataset, compute_metrics)
  ensemble_first_step.add_arguments(tokenizer2, model2, training_args2, ImbalancedSamplerTrainer,
                                    tokenized_dataset, RE_Dataset, compute_metrics)
  ensemble_first_step.train(seed_value=42)
  #model.save_pretrained('./best_model')
  #tokenizer.save_pretrained('./best_model')
  print('Finished!')

def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

'''def model_init():
    model_config =  AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 30

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
    # KBS) add special tokens
    model.resize_token_embeddings(tokenizer.vocab_size + added_token_num)
    print(model.parameters)
    model.to(device)

    return model

def hp_space_ray(trial):
    config = {
        "learning_rate" : tune.loguniform(1e-5, 5e-5),
        "num_train_epochs" : tune.choice(range(3, 15)),
        "per_device_train_batch_size" : tune.choice([16, 32]),
        "gradient_accumulation_steps" : tune.choice(range(1, 3))
    }
    return config

def compute_objective(metrics):
    f1 = metrics["eval_micro f1 score"]
    return f1'''

if __name__ == '__main__':
  os.environ['WANDB_API_KEY'] = 'f5b1f2d16ad90a4bfefca9e344309d152509ac3b'
  os.environ['WANDB_ENTITY'] = 'plzanswer'
  os.environ['WANDB_PROJECT'] = 'test-project'
  os.environ['WANDB_NAME'] ="KBS-Test-Run"
  os.environ['WANDB_LOG_MODEL'] = 'True'
  train()