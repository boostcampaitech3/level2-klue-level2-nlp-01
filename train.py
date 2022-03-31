import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from utils import *
import wandb

def train():
  # load model and tokenizer
  # MODEL_NAME = "bert-base-uncased"
  MODEL_NAME = "klue/roberta-large"
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  # KBS) add special tokens
  added_token_num = tokenizer.add_special_tokens({"additional_special_tokens":["[SUBJ]", "[/SUBJ]", "[OBJ]", "[/OBJ]"]})

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

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  print(device)
  # setting model hyperparameter
  model_config =  AutoConfig.from_pretrained(MODEL_NAME)
  model_config.num_labels = 30

  model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
  # KBS) add special tokens
  model.resize_token_embeddings(tokenizer.vocab_size + added_token_num)
  print(model.parameters)
  model.to(device)
  
  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(
    output_dir='./results',          # output directory
    save_total_limit=5,              # number of total save model.
    save_steps=500,                 # model saving step.
    num_train_epochs=10,              # total number of training epochs
    learning_rate=3e-5,               # learning_rate
    gradient_accumulation_steps=2,
    per_device_train_batch_size=32,  # batch size per device during training
    per_device_eval_batch_size=32,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=100,              # log saving step.
    evaluation_strategy='steps', # evaluation strategy to adopt during training
                                # `no`: No evaluation during training.
                                # `steps`: Evaluate every `eval_steps`.
                                # `epoch`: Evaluate every end of epoch.
    eval_steps = 500,            # evaluation step.
    load_best_model_at_end = True,
    report_to=['wandb'],
    metric_for_best_model='micro f1 score'
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
  trainer.train()
  model.save_pretrained('./best_model')
  tokenizer.save_pretrained('./best_model')

if __name__ == '__main__':
  os.environ['WANDB_API_KEY'] = 'f5b1f2d16ad90a4bfefca9e344309d152509ac3b'
  os.environ['WANDB_ENTITY'] = 'plzanswer'
  os.environ['WANDB_PROJECT'] = 'test-project'
  os.environ['WANDB_NAME'] ="KBS-Test-Run"
  os.environ['WANDB_LOG_MODEL'] = 'True'
  train()