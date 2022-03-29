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
      