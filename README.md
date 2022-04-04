# level2-klue-level2-nlp-01

train

`python train.py --n_splits 5`

inference

`python inference.py --num_of_fold 5`

### load_data.py
- state 추가 (train시에만 augmentation 진행)


load_data(data_dir, state = 'train')

load_data(data_dir, state = 'inference')
