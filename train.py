import argparse
import utils
import wandb
import sys
from models import MT5SequenceClassification

import sklearn
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import MT5ForConditionalGeneration
from load_data import *


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
    save_total_limit=1,
    num_train_epochs=13,              # total number of training epochs
    learning_rate=1.827e-05,               # learning_rate
    gradient_accumulation_steps=1,
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

  # trainer = ImbalancedSamplerTrainer(
  #   model=model,                         
  #   args=training_args,                 
  #   train_dataset=RE_train_dataset,         
  #   eval_dataset=RE_dev_dataset,            
  #   compute_metrics=compute_metrics      
  # )

  trainer = Trainer(
     model=model,                       
     args=training_args,                
     train_dataset=RE_train_dataset,       
     eval_dataset=RE_dev_dataset,          
     compute_metrics=compute_metrics      
   )

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


#
#
# PIPELINE_MAP = {
#     "google/mt5-large" : 'MT5SequenceClassification',
# }
#
# def klue_re_micro_f1(preds, labels):
#     """KLUE-RE micro f1 (except no_relation)"""
#     label_list = ['no_relation', 'org:top_members/employees', 'org:members',
#         'org:product', 'per:title', 'org:alternate_names',
#         'per:employee_of', 'org:place_of_headquarters', 'per:product',
#         'org:number_of_employees/members', 'per:children',
#         'per:place_of_residence', 'per:alternate_names',
#         'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
#         'per:spouse', 'org:founded', 'org:political/religious_affiliation',
#         'org:member_of', 'per:parents', 'org:dissolved',
#         'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
#         'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
#         'per:religion']
#     no_relation_label_idx = label_list.index("no_relation")
#     label_indices = list(range(len(label_list)))
#     label_indices.remove(no_relation_label_idx)
#     return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0
#
# def klue_re_auprc(probs, labels):
#     """KLUE-RE AUPRC (with no_relation)"""
#     labels = np.eye(30)[labels]
#
#     score = np.zeros((30,))
#     for c in range(30):
#         targets_c = labels.take([c], axis=1).ravel()
#         preds_c = probs.take([c], axis=1).ravel()
#         precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
#         score[c] = sklearn.metrics.auc(recall, precision)
#     return np.average(score) * 100.0
#
# def compute_metrics(pred):
#     """ validation을 위한 metrics function """
#     labels = pred.label_ids
#     preds = pred.predictions.argmax(-1)
#     probs = pred.predictions
#
#     # calculate accuracy using sklearn's function
#     f1 = klue_re_micro_f1(preds, labels)
#     auprc = klue_re_auprc(probs, labels)
#     acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.
#
#     wandb.log({"micro f1": f1, "auprc": auprc, "acc": acc})
#
#     return {
#         'micro f1 score': f1,
#         'auprc' : auprc,
#         'accuracy': acc,
#     }
#
# def label_to_num(label):
#     num_label = []
#     with open('dict_label_to_num.pkl', 'rb') as f:
#       dict_label_to_num = pickle.load(f)
#     for v in label:
#       num_label.append(dict_label_to_num[v])
#
#     return num_label
#
def train(model_args, train_args, data_args):
    utils.set_seeds(data_args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load model and tokenizer
    # MODEL_NAME = "bert-base-uncased"
#
#     MODEL_NAME = model_args.MODEL_NAME
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#
#     # load dataset
#
#     train_dataset = load_data(data_args.data)
#     train_dataset, dev_dataset = split_train_valid_stratified(train_dataset, split_ratio=0.2)
#
#     train_label = label_to_num(train_dataset['label'].values)
#     dev_label = label_to_num(dev_dataset['label'].values)
#
#     # tokenizing dataset
#     tokenized_train = tokenized_dataset(train_dataset, tokenizer)
#     tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)
#
#     # make dataset for pytorch.
#     RE_train_dataset = RE_Dataset(tokenized_train, train_label)
#     RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)
#
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#
#     print(device)
#     # setting model hyperparameter
#     model_config = AutoConfig.from_pretrained(MODEL_NAME)
#     model_config.num_labels = 30
#
#     model_architecture = getattr(sys.modules[__name__], "MT5SequenceClassification")
#     model =  model_architecture.from_pretrained(MODEL_NAME, config=model_config)
#     print(model.config)
#     model.parameters
#     model.to(device)
#
#     # train_args.output_dir = os.path.join(train_args.output_dir, train_args.run_name)
#     print(train_args.output_dir)
#     # 사용한 option 외에도 다양한 option들이 있습니다.
#     # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
#     training_args = TrainingArguments(
#         **train_args
#     )
#     trainer = Trainer(
#         model=model,                                     # the instantiated 🤗 Transformers model to be trained
#         args=training_args,                           # training arguments, defined above
#         train_dataset=RE_train_dataset,             # training dataset
#         eval_dataset=RE_dev_dataset,                   # evaluation dataset
#         compute_metrics=compute_metrics             # define metrics function
#     )
#
#     # train model
#     trainer.train()
#     model.save_pretrained('./best_model')

def main(args):
   model_args, train_args, data_args, logging_args = utils.get_arguments(args)
   utils.wandb_init(logging_args)
   train(model_args, train_args, data_args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/train_config.json', help='config.json file')

    args = parser.parse_args()
    config = utils.read_json(args.config)

    parser.set_defaults(**config)
    args = parser.parse_args()

    main(args)
