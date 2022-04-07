import pandas as pd
import pickle as pickle
import numpy as np
import argparse
import os
from glob import glob


def get_csv_files(PATH):
  all_csv_files = []
  EXT = "*.csv"
  for path, subdir, files in os.walk(PATH):
    for file in glob(os.path.join(path, EXT)):
      all_csv_files.append(file)
  return all_csv_files


def get_data_frames(csv_files):
  dataframe_lst = []
  for csv in csv_files:
    pd_dataset = pd.read_csv(csv)
    dataframe_lst.append(pd_dataset)
  return dataframe_lst


def get_score_lst(data_frame_lst):
  score_lst = []
  for data_frame in data_frame_lst:
    score_lst.append(list(map(eval, data_frame['probs'].tolist())))
  return score_lst


def get_total_score(score_lst):
  score_total = []
  num_ = len(score_lst)
  for i in zip(*score_lst):
    score_total.append([sum(l) / num_ for l in zip(*i)])
  return score_total


def score_to_label(total_score):
  new_labels = []
  with open('dict_num_to_label.pkl', 'rb') as f:
    dict_num_to_label = pickle.load(f)

  for score in total_score:
    max_index = np.argmax(score)
    new_labels.append(dict_num_to_label[max_index])

  return new_labels


def compute_difference(softvoting_pred_labels, pred_labels_lst):
  sentence = ''
  for index, pred_label in enumerate(pred_labels_lst):
    s_cnt = 0
    d_cnt = 0
    for sf_label, pred_label in zip(softvoting_pred_labels, pred_label):
      if sf_label == pred_label:
        s_cnt += 1
      else:
        d_cnt += 1
    sentence += f'==== Compare difference between softvoting and {index}th submission.csv ==== \n'
    sentence += f'Common {s_cnt} \n'
    sentence += f'Difference {d_cnt} \n'

  return sentence

def main(args):
  """
    주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
  """
  PATH = args.model_dir

  # each scores from submission.csv files
  csv_files = get_csv_files(PATH)
  data_frame_lst = get_data_frames(csv_files)
  test_id = data_frame_lst[0]['id'].to_list()
  score_lst = get_score_lst(data_frame_lst)

  pred_labels_lst = []
  for score in score_lst:
    pred_labels_lst.append(score_to_label(score))

  # soft-voting
  total_score = get_total_score(score_lst)
  softvoting_pred_labels = score_to_label(total_score)

  # Compute between softvoting_pred_labels and pred_labels_lst
  result = compute_difference(softvoting_pred_labels, pred_labels_lst)

  ## make csv file with predicted answer
  #########################################################
  # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
  output = pd.DataFrame({'id': test_id, 'pred_label': softvoting_pred_labels, 'probs': total_score, })
  output_path = os.path.join(PATH, 'soft_voting_submission.csv')
  output.to_csv(output_path, index=False)  # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
  #### 필수!! ##############################################
  print('---- Finish! ----')

  result_path = os.path.join(PATH, 'compare_result.txt')
  with open(result_path, "w") as text_file:
    text_file.write(result)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--model_dir', type=str, default="./best_model/klue-roberta-large-new-tag-5-fold-re")
  args = parser.parse_args()
  print(vars(args))
  main(args)
