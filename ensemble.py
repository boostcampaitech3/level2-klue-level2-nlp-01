# KBS) 아래의 두 링크의 내용을 읽어보고, 그를 토대로 짜보는 앙상블 클래스입니다.
# KBS) http://rasbt.github.io/mlxtend/user_guide/classifier/StackingCVClassifier/
# KBS) https://techblog-history-younghunjo1.tistory.com/103
import numpy as np
import pandas as pd
import torch
import random
import os
from typing import List, Callable, Iterable
from sklearn.model_selection import StratifiedKFold
from utils import *

# KBS) 전체 데이터셋을 우선적으로 train_dataset 및 test_dataset으로 나누고, 여기서 train_dataset을 다시 KFold로 분리한다.
def KFold_split_dataset(dataset: Iterable, num_folds: int=5, split_ratio: float=0.2):
    train_dataset, test_dataset = split_train_valid_stratified(dataset, split_ratio=split_ratio)
    train_label = np.array(label_to_num(train_dataset['label'].values))
    test_label = np.array(label_to_num(test_dataset['label'].values))
    skf = StratifiedKFold(n_splits=num_folds)
    kfold_train_dataset = []
    kfold_valid_dataset = []
    kfold_train_label = []
    kfold_valid_label = []
    for train_idx, valid_idx in skf.split(train_dataset, train_label):
        kfold_train_dataset.append(train_dataset.iloc[train_idx])
        kfold_train_label.append(train_label[train_idx])
        kfold_valid_dataset.append(train_dataset.iloc[valid_idx])
        kfold_valid_label.append(train_label[valid_idx])
    
    return kfold_train_dataset, kfold_train_label, kfold_valid_dataset, kfold_valid_label, test_dataset, test_label

# KBS) 먼저, 1단계로 각 모델에 대해 KFold로 각각의 모델(base classifier)을 개별적으로 학습시킨다.
class StackingEnsembleFirstStep:
    def __init__(self, kfold_train_dataset: List[Iterable], kfold_train_label: List[Iterable],
                       kfold_valid_dataset: List[Iterable], kfold_valid_label: List[Iterable]):
        assert len(kfold_train_dataset) == len(kfold_valid_dataset), 'train_dataset 1개당 valid_dataset 1개를 맞추어주세요.'
        assert len(kfold_train_dataset) == len(kfold_train_label), 'train_dataset과 train_label 개수를 맞추어주세요.'
        assert len(kfold_valid_dataset) == len(kfold_valid_label), 'valid_dataset과 valid_label 개수를 맞추어주세요.'
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.kfold_train_dataset = kfold_train_dataset
        self.kfold_train_label = kfold_train_label
        self.kfold_valid_dataset = kfold_valid_dataset
        self.kfold_valid_label = kfold_valid_label
        self.tokenizer_list = []
        self.model_list = []
        self.training_arguments_list = []
        self.trainerClass_list = []
        self.tokenizedClass_list = []
        self.REClass_list = []
        self.compute_metrics_list = []

    def train(self, seed_value=42):
        self.set_seed(seed_value=seed_value)
        for idx, (train_dataset, train_label, valid_dataset, valid_label) in enumerate(zip(self.kfold_train_dataset, self.kfold_train_label, self.kfold_valid_dataset, self.kfold_valid_label)):
            for idss, (tokenizer, model, training_arguments, trainerClass, tokenizedClass, REClass, compute_metrics) in enumerate(zip(self.tokenizer_list, self.model_list,
                                                                                                                                      self.training_arguments_list, self.trainerClass_list,
                                                                                                                                      self.tokenizedClass_list, self.REClass_list,
                                                                                                                                      self.compute_metrics_list)):
                # tokenizing dataset
                tokenized_train = tokenizedClass(train_dataset, tokenizer)
                tokenized_valid = tokenizedClass(valid_dataset, tokenizer)

                # make dataset for pytorch.
                RE_train_dataset = REClass(tokenized_train, train_label)
                RE_valid_dataset = REClass(tokenized_valid, valid_label)
                model.to(self.device)
                trainer = trainerClass(model=model,                         
                                       args=training_arguments,                 
                                       train_dataset=RE_train_dataset,         
                                       eval_dataset=RE_valid_dataset,            
                                       compute_metrics=compute_metrics)
                trainer.train()
                output_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'step1_models', str(idss) + '번모델', str(idx) + '-KFold')
                os.makedirs(output_path, exist_ok=True)
                model.save_pretrained(output_path)
                tokenizer.save_pretrained(output_path)
                eval_dataloader = trainer.get_eval_dataloader()
                predictions = []
                labels_list = []

                model.eval()

                for _, inputs in enumerate(eval_dataloader):
                    print(inputs)
                    inputs = inputs.to(self.device)
                    labels = inputs.get("labels")
                    outputs = model(**inputs)
                    logits = outputs.get("logits")
                    predictions.append(logits.detach().cpu().numpy())
                    labels_list.append(labels.detach().cpu().numpy())

                predictions = np.concatenate(predictions, dim=0)
                print(labels_list)
                second_step_inputs = {'predictions' : predictions, 'labels' : labels_list}
                with open(os.path.join(output_path, 'predictions.pkl'), 'wb') as f:
                    pickle.dump(predictions, f)

                print('\n\n\n')
                print('-' * 30)
                print(f'{idx}-Fold | {idss}번째 Model 완료!')
                print('-' * 30)

    
    def add_arguments(self, tokenizer, model, training_arguments, trainerClass:Callable, tokenizedClass, REClass, compute_metrics):
        self.tokenizer_list.append(tokenizer)
        self.model_list.append(model)
        self.training_arguments_list.append(training_arguments)
        self.trainerClass_list.append(trainerClass)
        self.tokenizedClass_list.append(tokenizedClass)
        self.REClass_list.append(REClass)
        self.compute_metrics_list.append(compute_metrics)
    
    def set_seed(self, seed_value=42):
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

class StackingEnsembleSecondStep:
    def __init__(self):
        pass

class StackingEnsembleFinalStep:
    def __init__(self):
        pass

class StackingEnsembleClassifier:
    def __init__(self):
        pass

if __name__ == '__main__':
    with open('./train_dataframe.pkl', 'rb') as f:
        dataset = pickle.load(f)
    kfold_train_dataset, kfold_train_label, kfold_valid_dataset, kfold_valid_label, test_dataset, test_label = KFold_split_dataset(dataset, num_folds=5, split_ratio=0.2)
    print(len(kfold_train_dataset[0]), len(kfold_train_label[0]))
    print(len(kfold_valid_dataset[0]), len(kfold_valid_label[0]))
    print(len(test_dataset), len(test_label))
    print('Finished!')