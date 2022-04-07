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

def main(args):
  """
    주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
  """
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  # load tokenizer

  tokenizer1 = AutoTokenizer.from_pretrained("klue/roberta-large")
  tokenizer2 = RobertaTokenizer.from_pretrained("roberta-large")
  tokenizer3 = RobertaTokenizer.from_pretrained("roberta-base")
  

  model1 = AutoModelForSequenceClassification.from_pretrained(args.model_dir1) # klue/roberta-large 
  model2 = AutoModelForSequenceClassification.from_pretrained(args.model_dir2)
  model3 = AutoModelForSequenceClassification.from_pretrained(args.model_dir3)
  model4 = AutoModelForSequenceClassification.from_pretrained(args.model_dir4)
  model5 = AutoModelForSequenceClassification.from_pretrained(args.model_dir5)
  model6 = RobertaForSequenceClassification.from_pretrained(args.model_dir6)
  model7 = AutoModelForSequenceClassification.from_pretrained(args.model_dir7)
  model8 = AutoModelForSequenceClassification.from_pretrained(args.model_dir8)
  model9 = AutoModelForSequenceClassification.from_pretrained(args.model_dir9)
  model10 = AutoModelForSequenceClassification.from_pretrained(args.model_dir10)
  model11 = AutoModelForSequenceClassification.from_pretrained(args.model_dir11)
  

  ## load test datset
  test_dataset_dir = "../dataset/test/test_data.csv"
  test_id1, test_dataset1, test_label1 = load_test_dataset(test_dataset_dir, tokenizer1) # klue/roberta-large
  Re_test_dataset1 = RE_Dataset(test_dataset1 ,test_label1)

  test_id2, test_dataset2, test_label2 = load_test_dataset(test_dataset_dir, tokenizer2) # roberta-large
  Re_test_dataset2 = RE_Dataset(test_dataset2 ,test_label2)

  test_id3, test_dataset3, test_label3 = load_test_dataset(test_dataset_dir, tokenizer3) # roberta-base
  Re_test_dataset3 = RE_Dataset(test_dataset3 ,test_label3)

  model1.to(device)
  ## predict answer
  output_prob1 = inference_only_logit(model1, Re_test_dataset1, device, "klue/roberta-large") # model에서 class 추론
  
  model2.to(device)
  ## predict answer
  output_prob2 = inference_only_logit(model2, Re_test_dataset1, device, "klue/roberta-large") # model에서 class 추론
  
#   model3.to(device)
#   ## predict answer
#   output_prob3 = inference_only_logit(model3, Re_test_dataset1, device, "klue/roberta-large") # model에서 class 추론
  
#   model4.to(device)
#   ## predict answer
#   output_prob4 = inference_only_logit(model4, Re_test_dataset1, device, "klue/roberta-large") # model에서 class 추론
  
#   model5.to(device)
#   ## predict answer
#   output_prob5 = inference_only_logit(model5, Re_test_dataset1, device, "klue/roberta-large") # model에서 class 추론

  model6.to(device)
  ## predict answer
  output_prob6 = inference_only_logit(model6, Re_test_dataset2, device, 'roberta-large') # model에서 class 추론

  model7.to(device)
  ## predict answer
  output_prob7 = inference_only_logit(model7, Re_test_dataset1, device, "klue/roberta-large") # model에서 class 추론

  model8.to(device)
  ## predict answer
  output_prob8 = inference_only_logit(model8, Re_test_dataset1, device, "klue/roberta-large") # model에서 class 추론

  model9.to(device)
  ## predict answer
  output_prob9 = inference_only_logit(model9, Re_test_dataset3, device, "roberta-base") # model에서 class 추론

  model10.to(device)
  ## predict answer
  output_prob10 = inference_only_logit(model10, Re_test_dataset3, device, "roberta-base") # model에서 class 추론

  model11.to(device)
  ## predict answer
  output_prob11 = inference_only_logit(model11, Re_test_dataset2, device, "roberta-large") # model에서 class 추론

  mean_output_prob1_2 = (output_prob1 + output_prob2) / 2
  mean_output_prob7_8 = (output_prob7 + output_prob8) / 2
  mean_output_prob9_10 = (output_prob9 + output_prob10) / 2
  mean_output_prob = (mean_output_prob1_2 + output_prob6 + mean_output_prob7_8 + mean_output_prob9_10 + output_prob11) / 5
#   mean_output_prob = (output_prob1 + output_prob2 + output_prob7  + output_prob8) / 4
  mean_output_prob = mean_output_prob.tolist()

  pred_answer = prob_to_label(mean_output_prob)
  pred_answer = num_to_label(pred_answer)

  ## make csv file with predicted answer
  #########################################################
  # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
  output = pd.DataFrame({'id':test_id1,'pred_label':pred_answer,'probs':mean_output_prob,})

  output.to_csv('./prediction/submission.csv', index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
  #### 필수!! ##############################################
  print('---- Finish! ----')
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  # model dir
  parser.add_argument('--model_dir1', type=str, default="./best_model/roberta_bs-32_no-split")
  parser.add_argument('--model_dir2', type=str, default="./best_model/roberta_bs-32_no-split2")
  parser.add_argument('--model_dir3', type=str, default="./best_model/roberta_bs-128_no-split")
  parser.add_argument('--model_dir4', type=str, default="./best_model/roberta_bs-2048_no-split")
  parser.add_argument('--model_dir5', type=str, default="./best_model/roberta_bs64_split")
  parser.add_argument('--model_dir6', type=str, default="./best_model/non-klue/roberta_bs-32_split")
  parser.add_argument('--model_dir7', type=str, default="./teammates/lyj_klue-roberta_1")
  parser.add_argument('--model_dir8', type=str, default="./teammates/lyj_klue-roberta_2")
  parser.add_argument('--model_dir9', type=str, default="./best_model/xlm-roberta-base_bs-32_split_on-aug")
  parser.add_argument('--model_dir10', type=str, default="./best_model/xlm-roberta-base_bs-32_split_on-aug2")
  parser.add_argument('--model_dir11', type=str, default="./best_model/xlm-roberta-large_bs-32_split_on-aug")
  args = parser.parse_args()
  print(args)
  main(args)
  
