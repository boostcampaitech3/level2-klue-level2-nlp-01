import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from torch.optim import AdamW
from utils import *


def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = ['no_relation', 'org:top_members/employees', 'org:members',
       'org:product', 'per:title', 'org:alternate_names',
       'per:employee_of', 'org:place_of_headquarters', 'per:product',
       'org:number_of_employees/members', 'per:children',
       'per:place_of_residence', 'per:alternate_names',
       'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
       'per:spouse', 'org:founded', 'org:political/religious_affiliation',
       'org:member_of', 'per:parents', 'org:dissolved',
       'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
       'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
       'per:religion']
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0

def compute_metrics(pred):
  """ validation을 위한 metrics function """
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  probs = pred.predictions

  # calculate accuracy using sklearn's function
  f1 = klue_re_micro_f1(preds, labels)
  auprc = klue_re_auprc(probs, labels)
  acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.

  return {
      'micro f1 score': f1,
      'auprc' : auprc,
      'accuracy': acc,
  }

def label_to_num(label):
  num_label = []
  with open('dict_label_to_num.pkl', 'rb') as f:
    dict_label_to_num = pickle.load(f)
  for v in label:
    num_label.append(dict_label_to_num[v])
  
  return num_label

def train():
  # load model and tokenizer
  # MODEL_NAME = "bert-base-uncased"
  MODEL_NAME = "klue/roberta-large"
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

  # load dataset
  train_dataset = load_data("./dataset/train/train.csv")
  train_dataset, dev_dataset = split_train_valid_stratified(train_dataset, split_ratio=0.2)

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
  print(model.config)
  print(model.parameters)
  model.to(device)

  # Argument로 넣어줘야할 부분임
  batch_size = 32
  lr = 5e-5
  num_train_epochs = 3
  
  train_loader = DataLoader(dataset=RE_train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=4)
  dev_loader = DataLoader(dataset=RE_dev_dataset,
                          batch_size=batch_size,
                          num_workers=4)
  
  optimizer = AdamW(model.parameters(),
                  lr=lr)
  total_steps = (int(len(train_dataset) / batch_size) + 1) * num_train_epochs
  scheduler = Warmup_Decay_Scheduler(optimizer, 500, total_steps)
  
  step = 0
  best_f1 = 0.0
  model.train()
  model.to(device)
  for epoch in range(num_train_epochs):
      print(f'----------Epoch {epoch} start----------')

      for idx, batch in enumerate(tqdm(train_loader)):
          optimizer.zero_grad()
          for key, val in batch.items():
              batch[key] = val.to(device)
          loss = focal_loss(model, batch)
          loss.backward()
          optimizer.step()

          if step % 100 == 0:
              print(f'Epoch : {epoch} | Step : {step}')
              print(f'loss : {loss}, learning_rate : {scheduler.get_last_lr()[0]}')
          
          if step % 500 == 0:
              probs = []
              labels = []
              print('----------Validation----------')
              model.eval()
              for idx_, batch_ in enumerate(tqdm(dev_loader)):
                  for key, val in batch_.items():
                      batch_[key] = val.to(device)
                  loss = focal_loss(model, batch_)
                  outputs = model(**batch_)
                  prob = F.softmax(outputs.get('logits'), dim=1)
                  label = batch_.get('labels')
                  probs.append(prob)
                  labels.append(label)
              probs = torch.cat(probs, dim=0)
              labels = torch.cat(labels, dim=0)
              pred = {'predictions' : probs, 'label_ids' : labels}
              metrics = compute_metrics(pred)
              print(f'micro f1 : {metrics["micro f1 score"]} | auprc : {metrics["auprc"]} | accuracy : {metrics["accuracy"]}')
              model.save_pretrained(f'./results/checkpoint-{step}')
              if metrics['micro f1 score'] > best_f1:
                  model.save_pretrained('./best_model/pytorch_model.bin')
                  print('New Best Model Saved!')
              
              model.train()
              
          step += 1
          scheduler.step()
    
  print('Train Finished!')

def main():
  train()

if __name__ == '__main__':
  main()
