import numpy as np
import pandas as pd
from typing import List, Callable, Iterable
from sklearn.model_selection import StratifiedKFold
from utils import *

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

class StackingEnsembleFirstStep:
    def __init__(self, dataset: Iterable, tokenizer_list: List[Callable], model_list: List[Callable], num_folds:int = 5):
        train_dataset, test_dataset = split_train_valid_stratified(dataset, split_ratio=0.2)
        train_label = label_to_num(train_dataset['label'].values)
        test_label = label_to_num(test_dataset['label'].values)
        skf = StratifiedKFold(n_splits=num_folds)
        kfold_train_dataset = []
        kfold_valid_dataset = []
        kfold_train_label = []
        kfold_valid_label = []
        for train_idx, valid_idx in skf.split(train_dataset, train_label):
            kfold_train_dataset.append(train_dataset[train_idx])
            kfold_train_label.append(train_label[train_idx])
            kfold_valid_dataset.append(train_dataset[valid_idx])
            kfold_valid_label.append(train_label[valid_idx])

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
    #StackingEnsembleFirstStep()
    print('Finished!')