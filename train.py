import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer, AutoModelForMaskedLM
import torch
from load_data_copy import *
from torchsampler import ImbalancedDatasetSampler
from torch.utils.data import DataLoader
from transformers import EarlyStoppingCallback
import wandb

class ImbalancedSamplerTrainer(Trainer):
    def get_train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset

        def get_label(dataset):
            return dataset["labels"]

        train_sampler = ImbalancedDatasetSampler(
          train_dataset
        )

        return DataLoader(
          train_dataset,
          batch_size=self.args.train_batch_size,
          sampler=train_sampler,
          collate_fn=self.data_collator,
          drop_last=self.args.dataloader_drop_last,
          num_workers=self.args.dataloader_num_workers,
          pin_memory=self.args.dataloader_pin_memory,
          )

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
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def klue_re_micro_f1_2(preds, labels):
  return sklearn.metrics.f1_score(labels, preds, average="micro")

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
  wandb.log({"micro f1": f1, "auprc": auprc, "acc": acc})

  return {
      'micro f1 score': f1,
      'auprc' : auprc,
      'accuracy': acc,
  }

def compute_metrics2(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  probs = pred.predictions

  f1 = klue_re_micro_f1_2(preds, labels)
  acc = accuracy_score(labels, preds)
  wandb.log({"micro f1": f1, "acc": acc})

  return {
    'micro f1 score': f1,
    'accuracy' : acc
  }

def label_to_num(label):
  num_label = []
  with open('dict_label_to_num.pkl', 'rb') as f:
    dict_label_to_num = pickle.load(f)
  for v in label:
    num_label.append(dict_label_to_num[v])
  
  return num_label

def label_to_num_only_relation(label):
  num_label = []
  for v in label:
    if v == 'no_relation':
      num_label.append(0)
    else:
      num_label.append(1)
  
  return num_label


def train():
  # load model and tokenizer
  # MODEL_NAME = "bert-base-uncased"
  # MODEL_NAME = 'roberta-large'
  # MODEL_NAME = 'roberta-base'
  MODEL_NAME = "klue/roberta-large"
  # MODEL_NAME = 'xlm-roberta-xlarge'
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  # tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

  # load dataset
  # train_dataset, dev_dataset = load_data_and_split("../dataset/train/train.csv", val_ratio=0.1)
  # train_dataset = load_data("../dataset/train/train2.csv")
  train_dataset, dev_dataset = load_data_and_split("../dataset/train/final_train_data.pkl", val_ratio=0.1, file_type='pkl')
  # train_dataset = load_data("../dataset/train/alternate_train.csv")
  # dev_dataset = load_data("../dataset/train/dev.csv") # validation용 데이터는 따로 만드셔야 합니다.

  # train_label = label_to_num_only_relation(train_dataset['label'].values)
  # dev_label = label_to_num_only_relation(dev_dataset['label'].values)
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
  model_config = AutoConfig.from_pretrained(MODEL_NAME)
  # model_config = RobertaConfig.from_pretrained(MODEL_NAME)
  model_config.num_labels = 30
  # model_config.num_labels = 2

  # model =  XLMRobertaXLForSequenceClassification.from_pretrained("xlm-roberta-xlarge", config=model_config)
  model =  AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
  # model =  RobertaForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
  print(model.config)
  model.parameters
  model.to(device)

  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(
    output_dir='./results',          # output directory
    save_total_limit=5,              # number of total save model.
    save_steps=500,                 # model saving step.
    num_train_epochs=15,              # total number of training epochs
    learning_rate=1e-5,               # learning_rate
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    gradient_accumulation_steps=2,    # gradient accumulation
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
    report_to='wandb',
    run_name='hgb_klue-roberta-large_bs-32_split_on-aug',
    # wandb_project='test-project',
    # wandb_entity='plzanswer'
    metric_for_best_model='micro f1 score',
    fp16=True,
    fp16_opt_level='01'
  )
  trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,         # training dataset
    eval_dataset=RE_dev_dataset,             # evaluation dataset
    compute_metrics=compute_metrics,         # define metrics function
    # callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
  )

  # train model
  trainer.train()
  model.save_pretrained('./best_model/klue-roberta-large_bs-32_split_on-aug')
def main():
  train()

if __name__ == '__main__':
  os.environ['WANDB_API_KEY'] = '7d7f3632e5a46c6fbb87481879d0b57c6757481b'
  os.environ['WANDB_ENTITY'] = 'plzanswer'
  os.environ['WANDB_PROJECT'] = 'test-project'
  os.environ['WANDB_LOG_MODEL'] = 'True'
  main()
