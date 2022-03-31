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
import wandb
import copy

MODEL_NAME = "klue/roberta-large"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
added_token_num = tokenizer.add_special_tokens({"additional_special_tokens":["[SUBJ]", "[/SUBJ]", "[OBJ]", "[/OBJ]"]})
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train():
  # load model and tokenizer
  # load dataset
  # train_dataset = load_data("./dataset/train/train.csv")
  # KBS) dataset 변경
  train_dataset = load_data("./dataset/train/alternate_train.csv")
  train_dataset, dev_dataset = split_train_valid_stratified(train_dataset, split_ratio=0.2)
  # dev_dataset = load_data("../dataset/train/dev.csv") # validation용 데이터는 따로 만드셔야 합니다.

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
    save_total_limit=5,              # number of total save model.
    save_steps=500,                 # model saving step.
    num_train_epochs=15,              # total number of training epochs
    learning_rate=1e-5,               # learning_rate
    gradient_accumulation_steps=2,
    per_device_train_batch_size=32,  # batch size per device during training
    per_device_eval_batch_size=32,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='/opt/ml/code/level2-klue-level2-nlp-01/logs',            # directory for storing logs
    logging_steps=100,              # log saving step.
    evaluation_strategy='steps', # evaluation strategy to adopt during training
                                # `no`: No evaluation during training.
                                # `steps`: Evaluate every `eval_steps`.
                                # `epoch`: Evaluate every end of epoch.
    eval_steps = 500,            # evaluation step.
    load_best_model_at_end = True,
    report_to=['wandb'],
    metric_for_best_model='micro f1 score',
    greater_is_better=True
  )

  trainer = ImbalancedSamplerTrainer(
    model_init=model_init,                         
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
  best_hyperparameter = trainer.hyperparameter_search(
    direction="maximize", 
    backend="ray", 
    hp_space=hp_space_ray,
    compute_objective=compute_objective
)
  # model.save_pretrained('./best_model')
  tokenizer.save_pretrained('/opt/ml/code/level2-klue-level2-nlp-01/best_model')
  with open('/opt/ml/code/level2-klue-level2-nlp-01/best_model/best_hyperparameter.pickle', 'wb') as f:
      pickle.dump(best_hyperparameter, f)

def model_init():
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
        "learning_rate" : tune.loguniform(1e-4, 5e-4),
        "num_train_epochs" : tune.choice(range(1, 15)),
        "seed" : tune.choice(range(1, 41)),
        "per_device_train_batch_size" : tune.choice([4, 8, 16, 32]),
        "gradient_accumulation_steps" : tune.choice(range(1, 2))
    }
    return config

def compute_objective(metrics):
    metrics = copy.deepcopy(metrics)
    f1 = metrics.pop("eval_micro f1 score")
    return f1

if __name__ == '__main__':
  os.environ['WANDB_API_KEY'] = 'f5b1f2d16ad90a4bfefca9e344309d152509ac3b'
  os.environ['WANDB_ENTITY'] = 'plzanswer'
  os.environ['WANDB_PROJECT'] = 'test-project'
  os.environ['WANDB_NAME'] ="KBS-Test-Run"
  os.environ['WANDB_LOG_MODEL'] = 'True'
  train()