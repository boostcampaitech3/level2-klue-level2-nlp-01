# level2-klue-level2-nlp-01
[NLP] Relation Extraction between entities.
Given data is csv files which have various infos including 'sentence', 'subject_entity', 'object_entity'.

---

input : sentence, sbj_entity, obj_entity
output : a predicted label among 30 labels and probs about each label. (the order of labels should be matched to the order of given dictionary)

---

train :  `python train.py` \(command\)
inference : `python inference.py` \(command\)

## history in leader board
### 22.03.24
- \[micro f1\] 63.2084 \[auprc\] 57.4551
   - klue/roberta-large
   - configs in baseline code

### 22.03.29
- \[micro f1\] 55.8401 \[auprc] 54.0585
   - xlm-roberta-large
   - configs in baseline code

### ~22.04.07
- (public test dataset)
   - \[micro f1\] 73.8665 \[auprc\] 75.8841
- (private test datset)
   - \[micro f1\] 71.6310 \[auprc\] 77.7407
- ensemble(soft-voting)
   - 8 models using klue/roberta-large, roberta-large


## edit
### 22.03.26
- `train.py`
   - add and use `ImbalancedSamplerTrainer`
      - reference [ImbalancedDatasetSampler](https://github.com/ufoym/imbalanced-dataset-sampler)
   - split dataset into train_dataset, val_dataset
   - add 'early-stopping'
      - `callbacks` parameter in trainer
      - (need to know how it works, which metric is used in early-stopping)
- `load_data.py`
   - add function `get_labels` in class `RE_Dataset` (because of ImbalancedDatasetSampler)
   - add function `load_data_and_split`
      - same as function `load_data`, but can split dataset about each labels by val_ratio.

### 22.03.27~22.04.07
- competition end
- `train.py`
   - try to split training into 2 parts
      - classify 'no_relation', 'relation'
      - classify 30 labels
         - because I thought my model often choose weird relation when 'no_relation' is correct.
         - (but this approach has insignificant increase.)
      - add functions
         - `klue_re_micro_f1_2`
         - `compute_metrics`
         - `label_to_num_only_relation`
   - start to use 'wandb'
      - can compare various approach.

- `load_data.py`
   - `custom_preprocessing_dataset`
      - comma base split(`.split(',')`) has problem
         - if subject_entity or object_entity has ',' in it, wrong output comes or error occurs.
         - for example, 'subject_entity' is '12,000'
      - so extract entity in other way.
   - `file_type` parameter in `load_data` and `load_data_and_split`
      - sometimes pickle file is more useful than csv file. so need to extract DataFrame in various file type.

- `inference2.py`
   - 'no_relation, relation' classifier
- `inference3.py`
   - ensemble(soft-voting)
      - get probs of each models, and mean probs
- `inference4.py`
   - inference every model -> inefficient
   - mean probs using submission.csv files

## feedback
### good
- continuous EDA
   - change way of data loading
   - data augmentation
   - some trying like `ImbalancedSampler`, `relation classifier about 'no_relation'` (in spite of failure)
### bad
- should try much more diverse approach like changing models, embedding, tagging, ensemble(like stacking), etc.
- should manage systematically
- should log in more detail, and frequently.
- should use collaborate tools more. 

      