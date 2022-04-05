import argparse
import utils
import sys
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import MT5ForConditionalGeneration
from load_data import *
import load_data
from models import *
import models
import pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from utils import *
import wandb

def hp_space_ray(trial):
    config = {k : eval(v) for k, v in trial.items()}
    return config

def compute_objective(metrics):
    f1 = metrics["eval_micro f1 score"]
    return f1

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

def label_to_num(label):
    num_label = []
    with open('dict_label_to_num.pkl', 'rb') as f:
      dict_label_to_num = pickle.load(f)
    for v in label:
      num_label.append(dict_label_to_num[v])

    return num_label

def train(model_args, train_args, data_args, hpyer_args):
    utils.set_seeds(data_args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # load model and tokenizer
    MODEL_NAME = model_args.MODEL_NAME
    ARCHITECTURE = model_args.architecture

    if hasattr(sys.modules[__name__], ARCHITECTURE):
        if ARCHITECTURE == "TokenizerAndModelForKlueReTask":
            tokenizer, model = getattr(sys.modules[__name__], ARCHITECTURE)(MODEL_NAME)
        else:
            model = getattr(sys.modules[__name__], ARCHITECTURE)(MODEL_NAME)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    else:
        model = AutoModelForSequenceClassification(MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # train_dataset = load_data("./dataset/train/alternate_train copy.csv")
    train_dataset = load_data(data_args.data)
    train_dataset, dev_dataset = split_train_valid_stratified(train_dataset, split_ratio=0.2)

    train_label = label_to_num(train_dataset['label'].values)
    dev_label = label_to_num(dev_dataset['label'].values)

    # tokenizing dataset
    tokenized_train = getattr(load_data, data_args.tokenized_dataset)(train_dataset, tokenizer)
    tokenized_dev = getattr(load_data, data_args.tokenized_dataset)(dev_dataset, tokenizer)

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

    print(device)
    print(model.config)
    model.to(device)

    train_args.output_dir = os.path.join(train_args.output_dir, train_args.run_name)
    os.makedirs(train_args.output_dir, exist_ok=True)
    print(f'model will be save at : {train_args.output_dir}')

    training_args = TrainingArguments(
        **train_args
    )

    trainer = Trainer(
        model=model,                                     # the instantiated 🤗 Transformers model to be trained
        args=training_args,                           # training arguments, defined above
        train_dataset=RE_train_dataset,             # training dataset
        eval_dataset=RE_dev_dataset,                   # evaluation dataset
        compute_metrics=compute_metrics             # define metrics function
    )

    print('Checking Hyper Parameter Search...')
    if not hpyer_args:
        print('Hyper Parameter Search Exists...')
        hyper_config = hp_space_ray(hpyer_args)
        best_hyperparameter = trainer.hyperparameter_search(
          direction="maximize",
          backend="ray",
          hp_space=lambda _: hyper_config,
          compute_objective=compute_objective
        )
    else:
        print('No Hyper Parameter Search...')
    # best_hyperparameter = trainer.hyperparameter_search(
    #   direction="maximize",
    #   backend="ray",
    #   hp_space=hp_space_ray,
    #   compute_objective=compute_objective
    #   )

    # print('Checking K-fold Cross Validation...')

    # train model
    trainer.train()
    best_path = os.path.join(model_args.best_model_dir, train_args.run_name)
    os.makedirs(best_path, exist_ok=True)
    print(f'best model will be save at : {best_path}')
    model.save_pretrained(best_path)

    if not hpyer_args:
        best_hyperparameter_path = os.path.join(best_path, 'best_hyperparameter')
        os.makedirs(best_path, exist_ok=True)
        with open(best_hyperparameter_path, 'wb') as f:
            pickle.dump(best_hyperparameter, f)

def main(args):
   model_args, train_args, data_args, logging_args, hpyer_args = utils.get_arguments(args)

   # wandb setting
   utils.wandb_init(logging_args)
   train(model_args, train_args, data_args, hpyer_args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/train_config.json', help='config.json file')

    args = parser.parse_args()
    config = utils.read_json(args.config)

    parser.set_defaults(**config)
    args = parser.parse_args()

    main(args)
