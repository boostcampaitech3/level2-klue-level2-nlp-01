from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader
from load_data_copy import *
import pandas as pd
import torch
import torch.nn.functional as F

import pickle as pickle
import numpy as np
import argparse
from tqdm import tqdm

def inference_only_logit(model, tokenized_sent, device, model_name):
  """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
  """
  dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
  model.eval()
  output_prob = []
  if model_name == "klue/roberta-large":
    for i, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            outputs = model(
                input_ids=data['input_ids'].to(device),
                attention_mask=data['attention_mask'].to(device),
                token_type_ids=data['token_type_ids'].to(device)
                )
        logits = outputs[0]
        prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
        output_prob.append(prob)
  elif model_name == "roberta-large" or model_name == 'roberta-base':
    for i, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            outputs = model(
                input_ids=data['input_ids'].to(device),
                attention_mask=data['attention_mask'].to(device),
                #   token_type_ids=data['token_type_ids'].to(device)
                )
        logits = outputs[0]
        prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
        output_prob.append(prob)
  
  return np.concatenate(output_prob, axis=0)

def prob_to_label(output_prob):
    output_pred = []
    for prob in output_prob:
        result = np.argmax(prob, axis=-1)
        output_pred.append(result)
    
    return output_pred

def num_to_label(label):
  """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
  """
  origin_label = []
  with open('dict_num_to_label.pkl', 'rb') as f:
    dict_num_to_label = pickle.load(f)
  for v in label:
    origin_label.append(dict_num_to_label[v])
  
  return origin_label

def load_test_dataset(dataset_dir, tokenizer):
  """
    test dataset을 불러온 후,
    tokenizing 합니다.
  """
  test_dataset = load_data(dataset_dir, 'csv')
  test_label = list(map(int,test_dataset['label'].values))
  # tokenizing dataset
  tokenized_test = tokenized_dataset(test_dataset, tokenizer)
  return test_dataset['id'], tokenized_test, test_label

def get_probs(sbm_dir):
    sbm = pd.read_csv(sbm_dir)
    return np.array(list(map(eval, sbm['probs'])))

def get_ids(sbm_dir):
    sbm = pd.read_csv(sbm_dir)
    return np.array(sbm['id'])

def main(args):
    """
        주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    # probs = get_probs(args.sbm_dir1)
    ids = get_ids(args.sbm_dir1)

    prob1 = get_probs(args.sbm_dir1)
    prob2 = get_probs(args.sbm_dir2)
    prob3 = get_probs(args.sbm_dir3)
    prob4 = get_probs(args.sbm_dir4)
    prob5 = get_probs(args.sbm_dir5)
    prob6 = get_probs(args.sbm_dir6)
    prob7 = get_probs(args.sbm_dir7)
    prob8 = get_probs(args.sbm_dir8)
    prob9 = get_probs(args.sbm_dir9)
    prob10 = get_probs(args.sbm_dir10)
    prob11 = get_probs(args.sbm_dir11)
    prob12 = get_probs(args.sbm_dir12)
    prob13 = get_probs(args.sbm_dir13)
    prob14 = get_probs(args.sbm_dir14)

    # mean_probs = (prob1 + prob2 + prob3 + prob4 + prob5 + prob6 + prob7 + prob8 + prob9 + prob10 + prob11 + prob12 + prob13 + prob14) / 14
    # mean_probs = ((prob1 + prob2) / 2 + prob3 + prob4 + prob5 + prob6 + prob7 + (prob8 + prob9) / 2 + prob10 + prob11 + prob12 + prob13 + prob14) / 12 
    # mean_probs = ((prob1 + prob2)/2 + prob6 +  prob7 + (prob8 + prob9)/2 + prob10 + (prob11 + prob12 + prob13 + prob14)/2 )/7
    # mean_probs = ((prob1 + prob2)/2 + (prob11 + prob12 + prob13 + prob14)/2) / 3
    # mean_probs = ((prob1+prob2)/2 + prob7 + ((prob8+prob9)/2 + prob10)/2 + (prob11+prob12+prob14)/3 + prob13) / 5
    # mean_probs = (prob1 + prob2 + prob3 + prob5 + prob11 + prob12 + prob13 + prob14) / 8
    mean_probs = ((prob1 + prob2 + prob11 + prob12 + prob13 + prob14)/6 + prob7 + ((prob8 + prob9)/2 + prob10)/2)/3
    mean_probs = mean_probs.tolist()

    pred_answer = prob_to_label(mean_probs)
    pred_answer = num_to_label(pred_answer)

    ## make csv file with predicted answer
    #########################################################
    # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
    output = pd.DataFrame({'id':ids,'pred_label':pred_answer,'probs':mean_probs,})

    output.to_csv('./prediction/submission.csv', index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    #### 필수!! ##############################################
    print('---- Finish! ----')
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  # model dir
  parser.add_argument('--sbm_dir1', type=str, default="./submissions2/klue-roberta_bs-32_no-split.csv")
  parser.add_argument('--sbm_dir2', type=str, default="./submissions2/klue-roberta_bs-32_no-split2.csv")
  parser.add_argument('--sbm_dir3', type=str, default="./submissions2/klue-roberta_bs-128_no-split.csv")
  parser.add_argument('--sbm_dir4', type=str, default="./submissions2/klue-roberta_bs-2048_no-split.csv")
  parser.add_argument('--sbm_dir5', type=str, default="./submissions2/klue-roberta_bs64_split.csv")
  parser.add_argument('--sbm_dir6', type=str, default="./submissions2/klue-roberta-large_split_on-aug.csv")
  parser.add_argument('--sbm_dir7', type=str, default="./submissions2/roberta_bs-32_split.csv")
  parser.add_argument('--sbm_dir8', type=str, default="./submissions2/xlm-roberta-base_bs-32_split_on-aug.csv")
  parser.add_argument('--sbm_dir9', type=str, default="./submissions2/xlm-roberta-base_bs-32_split_on-aug2.csv")
  parser.add_argument('--sbm_dir10', type=str, default="./submissions2/xlm-roberta-large_bs-32_split_on-aug.csv")
  parser.add_argument('--sbm_dir11', type=str, default="./teammates/lyj/klue-roberta-large_eda-aug.csv")
  parser.add_argument('--sbm_dir12', type=str, default="./teammates/lyj/klue-roberta-large_no-aug.csv")
  parser.add_argument('--sbm_dir13', type=str, default="./teammates/lyj/klue-roberta-large_pororo.csv")
  parser.add_argument('--sbm_dir14', type=str, default="./teammates/lyj/klue-roberta-large_unknown.csv")

  args = parser.parse_args()
  print(args)
  main(args)
  
