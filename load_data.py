import pickle as pickle
import os
import pandas as pd
import torch
from collections import defaultdict
import random

class RE_Dataset(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

def preprocessing_dataset(dataset):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  """ 현재 구현되어 있는 부분은 개체에 대한 단어만 데이터셋에 들어가게됨"""
  subject_entity = []
  object_entity = []
  subject_span = []
  object_span = []
  for i,j in zip(dataset['subject_entity'], dataset['object_entity']): # word, start_idx, end_idx, type
    sub_data = i[1:-1]
    obj_data = j[1:-1]

    sub_data_parsed = re.findall(r"'[^\']+'", sub_data)
    obj_data_parsed = re.findall(r"'[^\']+'", obj_data)

    sub_word = sub_data_parsed[1][1:-1]
    obj_word = obj_data_parsed[1][1:-1]

    sub_data = i[1:-1].split(', ')
    obj_data = j[1:-1].split(', ')
    for d in sub_data:
        if d.startswith("'start_idx'"):
            sub_start_idx = d.split(': ')[1]
        if d.startswith("'end_idx'"):
            sub_end_idx = d.split(': ')[1]
    sub_idx = (int(sub_start_idx), int(sub_end_idx))

    for d in obj_data:
        if d.startswith("'start_idx'"):
            obj_start_idx = d.split(': ')[1]
        if d.startswith("'end_idx'"):
            obj_end_idx = d.split(': ')[1]
    obj_idx = (int(obj_start_idx), int(obj_end_idx))

    subject_entity.append(sub_word)
    object_entity.append(obj_word)
    subject_span.append(sub_idx)
    object_span.append(obj_idx)
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,
                              'subject_span':subject_span, 'object_span':object_span, 'label':dataset['label']})
  return out_dataset

def load_data(dataset_dir):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir, sep='\t')
  dataset = preprocessing_dataset(pd_dataset)
  
  return dataset

# KBS) split함수
def split_train_valid_stratified(dataset, split_ratio=0.2):
    train_idx_list = [idx for idx in range(len(dataset['label']))]
    valid_idx_list = []
    indices_dict = defaultdict(list)
    for idx, label in enumerate(dataset['label']):
        indices_dict[label].append(idx)

    for key, idx_list in indices_dict.items():
        valid_idx_list.extend(idx_list[:int(len(idx_list) * split_ratio)])

    train_idx_list = list(set(train_idx_list) - set(valid_idx_list))
    train_dataset = dataset.iloc[train_idx_list]
    valid_dataset = dataset.iloc[valid_idx_list]

    return train_dataset, valid_dataset

def tokenized_dataset(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
    temp = ''
    temp = e01 + '[SEP]' + e02
    concat_entity.append(temp)
  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      )
  return tokenized_sentences


class RE_Dataset_test_mt5(torch.utils.data.Dataset):
    """MT5ForConditionalGeneration 위한 class """

    def __init__(self, dataset, tokenizer, max_len=384):
        self.data = dataset
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        #         target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"input_ids": source_ids,
                "attention_mask": src_mask}

    def _build(self):
        for index, row in self.data.iterrows():
            subject_entity = row['subject_entity'].strip()
            object_entity = row['object_entity'].strip()
            sentence = row['sentence'].strip()

            # option1
            # input_ = ''
            # input_ = input_ + 'What is the relationship between ' + subject_entity + ' and ' + \
            #          object_entity + ' in the following sentence? ' + sentence

            # option2
            input_ = '[Relation Extraction] '
            input_ = input_ + 'subject: ' + subject_entity + ' object: ' + \
                     object_entity + ' sentence: ' + sentence

            # tokenize inputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input_],
                # padding="longest",
                max_length=self.max_len,
                pad_to_max_length=True,
                return_tensors="pt"
            )

            self.inputs.append(tokenized_inputs)

class RE_Dataset_mt5(torch.utils.data.Dataset):
    """MT5ForConditionalGeneration 위한 class """

    def __init__(self, dataset, tokenizer, max_len=384):
        self.data = dataset
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []
        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()
        target_ids[target_ids == self.tokenizer.pad_token_id] = -100

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        #         target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"input_ids": source_ids,
                "attention_mask": src_mask,
                "labels": target_ids}

    def _build(self):
        for index, row in self.data.iterrows():
            subject_entity = row['subject_entity'].strip()
            object_entity = row['object_entity'].strip()
            sentence = row['sentence'].strip()

            # option1
            # input_ = ''
            # input_ = input_ + 'What is the relationship between ' + subject_entity + ' and ' + \
            #          object_entity + ' in the following sentence? ' + sentence

            # option2
            input_ = '[Relation Extraction] '
            input_ = input_ + 'subject: ' + subject_entity + ' object: ' + \
                     object_entity + ' sentence: ' + sentence

            target_ = ''
            label_ = row['label'].strip()
            target_ = target_ + label_

            # tokenize inputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input_],
                # padding="longest",
                max_length=self.max_len,
                pad_to_max_length=True,
                return_tensors="pt"
            )
            # tokenize targets
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [target_],
                # padding="longest",
                max_length=12,
                pad_to_max_length=True,
                return_tensors="pt"
            )

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)

        shuffle_lst = list(zip(self.inputs, self.targets))
        random.shuffle(shuffle_lst)
        self.inputs, self.targets = zip(*shuffle_lst)



