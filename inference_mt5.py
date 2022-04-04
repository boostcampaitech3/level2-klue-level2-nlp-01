from transformers import MT5ForConditionalGeneration, T5Tokenizer
from transformers import MT5ForConditionalGeneration
import utils
import torch
import pickle
import argparse
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import utils
from load_data import *
from tqdm import tqdm

def inference(model, test_data, tokenizer, device):
  """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
  """
  dataloader = DataLoader(test_data, batch_size=32, shuffle=False)

  preds_lst = []
  with torch.no_grad():
    model.eval()
    for data in tqdm(dataloader):
      outputs = model.generate(input_ids=data['input_ids'].to(device),
                               attention_mask=data['attention_mask'].to(device))

      outputs_cpu = outputs.detach().cpu()
      decoded_prediction = tokenizer.batch_decode(outputs_cpu, skip_special_tokens=True)
      preds_lst.extend(decoded_prediction)

  return preds_lst

def label_to_num(preds):
  """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
  """
  new_labels = []
  result = []
  no_key = []
  with open('dict_num_to_label.pkl', 'rb') as f:
    dict_num_to_label = pickle.load(f)
  dict_label_to_num = {v : k for k,v in dict_num_to_label.items()}
  cnt = 0
  for v in preds:
    probs = [0.001] * len(dict_label_to_num)
    if v in dict_label_to_num:
      probs[dict_label_to_num[v]] = 0.971
      new_labels.append(v)
    else:
      probs[dict_label_to_num['no_relation']] = 0.971
      no_key.append(v)
      new_labels.append('no_relation')
      cnt += 1
    result.append(probs)

  return new_labels, result, cnt, no_key

def load_test_dataset(dataset_dir):
  """
    test dataset을 불러온 후,
    tokenizing 합니다.
  """
  test_dataset = load_data(dataset_dir)
  test_label = list(map(int,test_dataset['label'].values))

  return test_dataset['id'], test_dataset, test_label

def main(args):
  """
    주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
  """

  model_args = utils.AttrDict(args.ModelArguments)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  model = MT5ForConditionalGeneration.from_pretrained(model_args.model_dir)
  model.to(device)
  tokenizer = T5Tokenizer.from_pretrained(model_args.MODEL_NAME)

  model.to(device)

  ## load test datset
  test_dataset_dir = model_args.test_data_dir
  test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir)
  Re_test_dataset = RE_Dataset_test_mt5(test_dataset, tokenizer)

  ## predict answer
  pred_answer = inference(model, Re_test_dataset, tokenizer, device) # model에서 class 추론
  new_labels, output_prob, cnt, no_key = label_to_num(pred_answer) # 숫자로 된 class를 원래 문자열 라벨로 변환.
  print(f'no keys : {cnt}')

  ## make csv file with predicted answer
  #########################################################
  # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
  print(f'len(test_id) : {len(test_id)}')
  print(f'len(pred_answer) : {len(pred_answer)}')
  print(f'len(output_prob) : {len(output_prob)}')
  output = pd.DataFrame({'id':test_id,'pred_label':new_labels,'probs':output_prob,})

  sub_path = os.path.join(model_args.model_dir, 'submission.csv')
  output.to_csv(sub_path, index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
  #### 필수!! ##############################################
  print('---- Finish! ----')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', default='./configs/m5t_inference_config.json', help='config.json file')

  args = parser.parse_args()
  config = utils.read_json(args.config)

  parser.set_defaults(**config)
  args = parser.parse_args()
  main(args)
  
