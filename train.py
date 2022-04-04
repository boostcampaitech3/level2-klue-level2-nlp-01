import pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from utils import *
from ray import tune
from models import *
import wandb
import copy
import random

def train():
  # load model and tokenizer
  # load dataset
  # train_dataset = load_data("./dataset/train/train.csv")
  # KBS) dataset 변경
  set_seed(42)
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  tokenizer, model = TokenizerAndModelForKlueReTask()
  train_dataset = load_data("./dataset/train/alternate_train copy.csv")
  train_dataset, dev_dataset = split_train_valid_stratified(train_dataset, split_ratio=0.2)

  train_label = label_to_num(train_dataset['label'].values)
  dev_label = label_to_num(dev_dataset['label'].values)

  # tokenizing dataset
  tokenized_train = tokenized_dataset(train_dataset, tokenizer)
  tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)
  RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

  print(device)
  
  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(
    output_dir='/opt/ml/code/level2-klue-level2-nlp-01/results',          # output directory
    save_total_limit=5,
    num_train_epochs=3,              # total number of training epochs
    learning_rate=3e-05,               # learning_rate
    gradient_accumulation_steps=2,
    save_strategy='steps',
    per_device_train_batch_size=32,  # batch size per device during training
    per_device_eval_batch_size=32,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='/opt/ml/code/level2-klue-level2-nlp-01/logs',            # directory for storing logs
    logging_steps=100,              # log saving step.
    evaluation_strategy='steps', # evaluation strategy to adopt during training
    eval_steps = 500,            # evaluation step.
    load_best_model_at_end = True,
    report_to=['wandb'],
    metric_for_best_model='micro f1 score',
    greater_is_better=True
  )

  trainer = ImbalancedSamplerTrainer(
    model=model,                         
    args=training_args,                 
    train_dataset=RE_train_dataset,         
    eval_dataset=RE_dev_dataset,            
    compute_metrics=compute_metrics      
  )

  # trainer = Trainer(
  #    model=model,                       
  #    args=training_args,                
  #    train_dataset=RE_train_dataset,       
  #    eval_dataset=RE_dev_dataset,          
  #    compute_metrics=compute_metrics      
  #  )

  # train model
  # best_hyperparameter = trainer.hyperparameter_search(
  #   direction="maximize", 
  #   backend="ray", 
  #   hp_space=hp_space_ray,
  #   compute_objective=compute_objective
  #   )
  trainer.train()
  # with open('/opt/ml/code/level2-klue-level2-nlp-01/best_model/best_hyperparameter.pkl', 'wb') as f:
  #     pickle.dump(best_parameter, f)
  model.save_pretrained('./best_model')
  tokenizer.save_pretrained('./best_model')

def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

# def model_init():
#     model_config =  AutoConfig.from_pretrained(MODEL_NAME)
#     model_config.num_labels = 30

#     model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
#     # KBS) add special tokens
#     model.resize_token_embeddings(tokenizer.vocab_size + added_token_num)
#     print(model.parameters)
#     model.to(device)

#     return model

# def hp_space_ray(trial):
#     config = {
#         "learning_rate" : tune.loguniform(1e-5, 5e-5),
#         "num_train_epochs" : tune.choice(range(3, 15)),
#         "per_device_train_batch_size" : tune.choice([16, 32]),
#         "gradient_accumulation_steps" : tune.choice(range(1, 3))
#     }
#     return config

# def compute_objective(metrics):
#     f1 = metrics["eval_micro f1 score"]
#     return f1

if __name__ == '__main__':
  os.environ['WANDB_API_KEY'] = 'f5b1f2d16ad90a4bfefca9e344309d152509ac3b'
  os.environ['WANDB_ENTITY'] = 'plzanswer'
  os.environ['WANDB_PROJECT'] = 'test-project'
  os.environ['WANDB_NAME'] ="KBS-Test-Run"
  os.environ['WANDB_LOG_MODEL'] = 'True'
  train()