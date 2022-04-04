from collections import defaultdict
from logging.config import valid_ident
import pickle as pickle
import os
import pandas as pd
import torch
import random
import re


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

def preprocessing_dataset(dataset, state):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  subject_entity = []
  object_entity = []
  for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
    i = i[1:-1].split(',')[0].split(':')[1]
    j = j[1:-1].split(',')[0].split(':')[1]

    subject_entity.append(i)
    object_entity.append(j)
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})

  if state == 'train':
    print("==== start augmentation ====")
    augmented = make_augmented(out_dataset, 'swap', len(out_dataset))
    print("===== augmented sentence length(swap) : ", len(augmented), " =====")
    augmented_data = pd.concat([out_dataset, augmented])

    augmented2 = make_augmented(out_dataset, 'delete', len(augmented_data))
    print("===== augmented sentence length(delete) : ", len(augmented2), " =====")
    augmented_data = pd.concat([augmented_data, augmented2])  
    print("==== end augmentation ====")
  else:
    augmented_data = out_dataset.copy()
  
  return augmented_data
  
  
  # return out_dataset

def make_augmented(dataset, eda, num):
  augmented = pd.DataFrame(columns=['id', 'sentence', 'subject_entity', 'object_entity', 'label'])
  id = num
  for i in range(10000):
    print(i,"/", 10000)
    if dataset['label'][i] != 'no_relation':
      augmented_sentences = ''
      # print("====", dataset['sentence'][i])
      augmented_sentences = EDA(sentence=dataset['sentence'][i], subject=dataset['subject_entity'][i], object=dataset['object_entity'][i], what_eda=eda)
      # print("setence : ", augmented_sentences)
      augmented.loc[i] = [id, augmented_sentences, dataset['subject_entity'][i], dataset['object_entity'][i], dataset['label'][i]]
      id += 1
    
  return augmented

def load_data(dataset_dir, state):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir, sep='\t')
  dataset = preprocessing_dataset(pd_dataset, state)
  
  return dataset

def split_train_valid_stratified(dataset, split_ratio = 0.2):
  train_idx_list = [idx for idx in range(len(dataset['label']))]
  vaild_idx_list = []
  indices_dict = defaultdict(list)
  for idx, label in enumerate(dataset['label']):
    indices_dict[label].append(idx)
  
  for key, idx_list in indices_dict.items():
    vaild_idx_list.extend(idx_list[:int(len(idx_list) * split_ratio)])
    
  train_idx_list = list(set(train_idx_list) - set(vaild_idx_list))
  train_dataset = dataset.iloc[train_idx_list]
  valid_dataset = dataset.iloc[vaild_idx_list]
  
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

def random_deletion(words, sub, obj, prob):
  if len(words) == 1:
    return words
  
  new_words = []
  for word in words:
    r = random.uniform(0, 1)
    if r > prob or (word in sub) or (sub in word) or (word in obj) or (obj in word):
      new_words.append(word)
  
  if len(new_words) == 0:
    rand_int = random.randint(0, len(word)-1)
    return [words[rand_int]]
  
  return new_words

def random_swap(words, n):
  new_words = words.copy()
  for _ in range(n):
    new_words = swap_word(new_words)
    
    return new_words

def swap_word(new_words):
  random_idx_1 = random.randint(0, len(new_words)-1)
  random_idx_2 = random_idx_1
  counter = 0
  
  while random_idx_2 == random_idx_1:
    random_idx_2 = random.randint(0, len(new_words)-1)
    counter += 1
    if counter > 3:
      return new_words
    
  new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
  return new_words

def EDA(sentence, subject, object, what_eda, alpha_rs = 0.1, p_rd = 0.15):
  # sentence = get_only_hangul(sentence)
  words = sentence.split(' ')
  # print("words : ", words)
  num_words = len(words)
  
  n_rs = max(1, int(alpha_rs * num_words))
  
  #random swap
  if what_eda == 'swap':
    augmented_sentences = random_swap(words, n_rs)
    
  #random delete, subject와 object에 있는 것은 삭제하면 안됨!
  if what_eda == 'delete':
    augmented_sentences = random_deletion(words, subject, object, p_rd)
    
  for i in range(len(augmented_sentences)):
    augmented_sentences[i] = str(augmented_sentences[i])
  
  augmented_str = " ".join(augmented_sentences)
  # print("make sentence : ", augmented_str)

  return augmented_str
  
  
  