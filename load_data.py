import pickle as pickle
import os
import pandas as pd
import torch
from collections import defaultdict
from transformers import Trainer
from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler
import torch.nn.functional as F

class RE_Dataset(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset # {'input_ids': ~, 'token_type_ids': ~, 'attention_mask': ~}
    self.labels = labels # [0, 1, 1, 0, 0, ...]

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
  for i,j in zip(dataset['subject_entity'], dataset['object_entity']): # word, start_idx, end_idx, type
    i = i[1:-1].split(',')[0].split(':')[1] # 'word': '비틀즈', ..., 'type': 'ORG' => '비틀즈'
    j = j[1:-1].split(',')[0].split(':')[1] # 'word': '조지 해리슨', ..., 'type': 'PER' => '조지 해리슨'

    subject_entity.append(i)
    object_entity.append(j)
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
  return out_dataset

def load_data(dataset_dir):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
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
      ) # [CLS]subject_entity[SEP]object_entity[SEP]sentence.....[PAD][PAD][PAD][SEP]
        # => input_ids, token_type_ids, attention_mask
  return tokenized_sentences

# KBS) ImbalancedSamplerTrainer
# https://velog.io/@ks2515/transformers-custom-trainer-%EB%A7%8C%EB%93%A4%EA%B8%B0 참조
# https://www.dlology.com/blog/multi-class-classification-with-focal-loss-for-imbalanced-datasets/ 참조
class ImbalancedSamplerTrainer(Trainer):
    def get_train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset

        def get_label(dataset):
            return dataset.labels

        train_sampler = ImbalancedDatasetSampler(train_dataset, callback_get_label=get_label)

        return DataLoader(
            train_dataset,
            shuffle=False,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        labels = F.one_hot(labels, num_classes=30)
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        alpha = 4.0
        gamma = 2.0
        epsilon=1.e-9
        prob = torch.add(F.softmax(logits, dim=-1), epsilon)
        cross_entropy = torch.multiply(labels, -torch.log(prob))
        weight = torch.multiply(labels, torch.pow(torch.subtract(1., prob), gamma))
        fl = torch.multiply(alpha, torch.multiply(weight, cross_entropy))
        loss, _ = torch.max(fl, dim=1, keepdim=False, out=None)
        loss = torch.mean(loss)
        """prob = torch.exp(log_prob)
        loss = F.nll_loss(
            ((1 - prob) ** gamma) * log_prob,
            labels,
            reduction='mean'
        )"""
        return (loss, outputs) if return_outputs else loss