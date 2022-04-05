#!/usr/bin/env python
# coding: utf-8

from transformers import MT5ForConditionalGeneration, T5Tokenizer
from transformers import MT5ForConditionalGeneration
import utils
import torch
import pickle
import argparse
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score
import os
import utils

def klue_re_micro_f1_mt5(preds, labels):
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
    label_list.remove("no_relation")
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_list) * 100.0

def evaluate(args):
    # {'MODEL_NAME': 'google/mt5-large',
    #  'model_dir': './models/mt5-large-sequence-generation-batch-32-Adafactor/checkpoint-1600',
    #  'eval_path': './models/mt5-large-sequence-generation-batch-32-Adafactor/RE_dev_dataset.pickle'}

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = MT5ForConditionalGeneration.from_pretrained(args.model_dir)
    model.to(device)

    tokenizer = T5Tokenizer.from_pretrained(args.MODEL_NAME)

    with open(args.eval_path, mode='rb') as f:
        RE_dev_dataset = pickle.load(f)

    dev_dataloader = DataLoader(RE_dev_dataset, batch_size=32, shuffle=False, )
    dev_dataloader2 = DataLoader(RE_dev_dataset, batch_size=32, shuffle=False, )

    preds_lst = []
    with torch.no_grad():
        model.eval()
        for data in tqdm(dev_dataloader2):
            input_ids_ = data['input_ids']
            attention_mask_ = data['attention_mask']

            outputs = model.generate(input_ids = input_ids_.to(device), attention_mask = attention_mask_.to(device))

            outputs_cpu = outputs.detach().cpu()
            decoded_prediction = tokenizer.batch_decode(outputs_cpu, skip_special_tokens=True)
            preds_lst.extend(decoded_prediction)

    label_lst = []
    sentence_lst = []

    for data in tqdm(dev_dataloader):
        input_ids = data['input_ids']
        sentence = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        sentence_lst.extend(sentence)
        labels = data['labels']
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        label_lst.extend(decoded_labels)

    f1_micro = klue_re_micro_f1_mt5(label_lst, preds_lst)
    f1_micro_path = os.path.join(args.model_dir, 'f1_micro_score.txt')
    print(f'saving f1_mirco : {f1_micro}')
    with open(f1_micro_path, "w") as file:
        file.write(f'{f1_micro}')

    print('saving inference result')
    inference_result = os.path.join(args.model_dir, 'f1_micro_result.csv')
    df = pd.DataFrame(list(zip(label_lst, preds_lst, sentence_lst)), columns = ['label','preds','sentence'])
    df.to_csv(inference_result, sep='\t')

def main(args):
   model_args = utils.AttrDict(args.ModelArguments)
   evaluate(model_args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/m5t_evaluate_config.json', help='config.json file')

    args = parser.parse_args()
    config = utils.read_json(args.config)

    parser.set_defaults(**config)
    args = parser.parse_args()
    main(args)



