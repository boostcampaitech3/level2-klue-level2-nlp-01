import numpy as np
import pandas as pd
from load_data_copy import *
from train_copy import *
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, f1_score

train_dataset = load_data("../dataset/train/train2.csv")
train_label = label_to_num(train_dataset['label'].values)
train_label = np.array(train_label).reshape(-1,1)

model1_output = pd.read_csv("./sub_models/model1.csv")
model2_output = pd.read_csv("./sub_models/model2.csv")
model3_output = pd.read_csv("./sub_models/model3.csv")

model1_output_pred = model1_output['pred']
model1_output_prob = list(map(eval, model1_output['prob']))
model1_output_pred = np.array(model1_output_pred).reshape(-1,1)
model1_output_prob = np.array(model1_output_prob).reshape(-1,30)

model2_output_pred = model2_output['pred']
model2_output_prob = list(map(eval, model2_output['prob']))
model2_output_pred = np.array(model2_output_pred).reshape(-1,1)
model2_output_prob = np.array(model2_output_prob).reshape(-1,30)

model3_output_pred = model1_output['pred']
model3_output_prob = list(map(eval, model3_output['prob']))
model3_output_pred = np.array(model3_output_pred).reshape(-1,1)
model3_output_prob = np.array(model3_output_prob).reshape(-1,30)

model_outputs_pred = np.concatenate([model1_output_pred, model2_output_pred, model3_output_pred], axis=1)
model_outputs_prob = np.concatenate([model1_output_prob + model2_output_prob + model3_output_prob], axis=1)
print(model_outputs_pred.shape)
print(model_outputs_prob.shape)
print(train_label.shape)

# lr_model =  LogisticRegression(C=10)
# lr_model.fit(model_outputs, train_label)
# final_pred = lr_model.predict(model_outputs)
# print(f1_score(train_label, final_pred, average='micro'))

mean_model_outputs_pred = np.mean(model_outputs_pred, axis=1)
mean_model_outputs_pred = np.around(mean_model_outputs_pred).astype(np.int64)
print(mean_model_outputs_pred.shape)
print(f1_score(train_label, mean_model_outputs_pred, average='micro'))

# lr_model = LogisticRegression()
lr_model = LinearRegression()
lr_model.fit(model_outputs_pred, train_label)
final = lr_model.predict(model_outputs_pred)
print(f1_score(train_label, final, average='micro'))