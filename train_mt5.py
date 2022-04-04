import argparse
import utils
import wandb
import sys
from models import MT5SequenceClassification

import os
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import MT5ForConditionalGeneration
from transformers.optimization import Adafactor, AdafactorSchedule
from load_data import *
import pickle


PIPELINE_MAP = {
    "google/mt5-large" : 'MT5SequenceClassification',
}

def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = ['no_relation', 'org:top_members/employees', 'org:members',
        'org:product', 'per:title', 'org:alternate_names',
        'per:employee_of', 'org:place_of_headquarters', 'per:product',
        'org:number_of_employees/members', 'per:children',
        'per:place_of_residence', 'per:alternate_names',
        'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
        'per:spouse', 'org:founded', 'org:political/religious_affiliation',
        'org:member_of', 'per:parents', 'org:dissolved',
        'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
        'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
        'per:religion']
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0

def compute_metrics(pred):
    """ validation을 위한 metrics function """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions

    # calculate accuracy using sklearn's function
    f1 = klue_re_micro_f1(preds, labels)
    auprc = klue_re_auprc(probs, labels)
    acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.

    wandb.log({"micro f1": f1, "auprc": auprc, "acc": acc})

    return {
        'micro f1 score': f1,
        'auprc' : auprc,
        'accuracy': acc,
    }


def klue_re_micro_f1_mt5(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = ['no_relation', 'org:top_members/employees', 'org:members',
        'org:product', 'per:title', 'org:alternate_names',
        'per:employee_of', 'org:place_of_headquarters', 'per:product',
        'org:number_of_employees/members', 'per:children',
        'per:place_of_residence', 'per:alternate_names',
        'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
        'per:spouse', 'org:founded', 'org:political/religious_affiliation',
        'org:member_of', 'per:parents', 'org:dissolved',
        'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
        'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
        'per:religion']
    label_list.remove("no_relation")
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_list) * 100.0

def compute_metrics_mt5(eval_preds):
    eval_preds = eval_preds.data.cpu().numpy()
    print(f'type(eval_preds) : {type(eval_preds)}')
    preds, labels = eval_preds
    print(f'type(preds) : {type(preds)}')
    print(f'type(labels) : {type(labels)}')
    if isinstance(preds, tuple):
        preds = preds[0]
    print(f'preds[:10] : {preds[:10]}')
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    print(f'decoded_preds[:10] : {decoded_preds[:10]}')
    # if data_args.ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    print(f'decoded_labels[:10] : {decoded_labels[:10]}')

    # Some simple post-processing
    # decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    f1 = klue_re_micro_f1_mt5(preds, labels)
    acc = accuracy_score(labels, preds)

    wandb.log({"micro f1 score": f1, "acc": acc})

    return {
        'micro f1 score': f1,
        'accuracy': acc,
    }

def label_to_num(label):
    num_label = []
    with open('dict_label_to_num.pkl', 'rb') as f:
      dict_label_to_num = pickle.load(f)
    for v in label:
      num_label.append(dict_label_to_num[v])

    return num_label

def train(model_args, train_args, data_args):

    # load model and tokenizer
    # MODEL_NAME = "bert-base-uncased"

    MODEL_NAME = model_args.MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # load dataset
    utils.set_seeds(data_args.seed)
    train_dataset = load_data(data_args.data)
    train_dataset, dev_dataset = split_train_valid_stratified(train_dataset, split_ratio=0.2)

    # train_label = label_to_num(train_dataset['label'].values)
    # dev_label = label_to_num(dev_dataset['label'].values)

    # tokenizing dataset
    # tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    # tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset_mt5(train_dataset, tokenizer)
    RE_dev_dataset = RE_Dataset_mt5(dev_dataset, tokenizer)

    dev_save_path = os.path.join(train_args.output_dir, 'RE_dev_dataset.pickle')
    with open(dev_save_path, 'wb') as f:
        pickle.dump(RE_dev_dataset, f, pickle.HIGHEST_PROTOCOL)
    print('saving dev_dataset')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(device)
    # setting model hyperparameter
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    # model_config.num_labels = 30

    model_architecture = getattr(sys.modules[__name__], model_args.architecture)
    model =  model_architecture.from_pretrained(MODEL_NAME, config=model_config)
    print(model.config)
    # print(model.parameters)
    model.to(device)

    # optimizer = Adafactor(model.parameters(),
    #                       scale_parameter=True,
    #                       relative_step=True,
    #                       warmup_init=True,
    #                       lr=None)
    # lr_scheduler = AdafactorSchedule(optimizer)

    # trainer = Trainer(..., optimizers=(optimizer, lr_scheduler))

    # train_args.output_dir = os.path.join(train_args.output_dir, train_args.run_name)
    print(train_args.output_dir)
    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
        **train_args
    )
    trainer = Trainer(
        model=model,                                     # the instantiated 🤗 Transformers model to be trained
        args=training_args,                           # training arguments, defined above
        train_dataset=RE_train_dataset,             # training dataset
        eval_dataset=RE_dev_dataset,                   # evaluation dataset
        # optimizers=(optimizer, lr_scheduler),
        # tokenizer = tokenizer,
        # compute_metrics=compute_metrics_mt5             # define metrics function
    )

    # train model
    trainer.train()
    # model.save_pretrained('./best_model')
    model.save_pretrained(model_args.best_model_dir)

def main(args):
   model_args, train_args, data_args, logging_args = utils.get_arguments(args)
   utils.wandb_login(logging_args)
   train(model_args, train_args, data_args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/m5t_train_config_adam.json', help='config.json file')

    args = parser.parse_args()
    config = utils.read_json(args.config)

    parser.set_defaults(**config)
    args = parser.parse_args()

    main(args)

