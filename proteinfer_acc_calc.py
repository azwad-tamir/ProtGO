# from copyreg import pickle
import copy
import json
from operator import index

import numpy
import pandas as pd
import re
import numpy as np
from setuptools.dist import sequence
import pickle
import tensorflow as tf
import os
import gzip
import base64
import io
import itertools
import subprocess
import shlex
import tqdm

# from classifier_trainer_MF import temp_labels

# Loading Proteinfer inference data:
proteinfer_pred_df = pd.read_csv('./data/proteinfer_random_test_go_pred.pkl')

# Loading top100 MF GO terms
with open(r"./data/random_train_top100.pkl", "rb") as input_file:
    all_data = pickle.load(input_file)

trian_id = all_data[1]
train_seq = all_data[4]
train_go = all_data[7]
train_go_orig = all_data[10]
# all_data = [bp_id, cc_id, mf_id, bp_seq, cc_seq, mf_seq, bp_go, cc_go, mf_go, bp_go_orig, cc_go_orig, mf_go_orig]

# Making unique go list for hot encoding the go terms in the dataset into labels:
unique_go = list(set([i for row in train_go_orig for i in row]))

########################################################################################################################
## Processing test Dataset
########################################################################################################################
val_df = pd.read_csv("./random_test_df")
val_id = list(val_df['ID'])
val_seq = list(val_df['sequence'])
val_go = list(val_df['GO_terms'])

# Cleaning GO terms:
val_go1 = []
for i in range(len(val_go)):
    if val_go[i] != '[]':
        val_go1.append(val_go[i].replace('[','').replace(']','').replace('\'','').replace(' ', '').split(','))
    else:
        val_go1.append([])

# Keeping only top100 go terms:
val_go2 = []
for i in range(len(val_go1)):
    go_temp = []
    for j in range(len(val_go1[i])):
        if val_go1[i][j] in unique_go:
            go_temp.append(val_go1[i][j])
    val_go2.append(go_temp)

# getting rid of repreated go terms:
val_go3 = []
for i in range(len(val_go2)):
    val_go3.append(list(set(val_go2[i])))

# making numpy encoding for test set:
true_labels = []
for i in range(len(val_go3)):
    temp_array = np.zeros((len(unique_go)))
    for j in range(len(val_go3[i])):
        temp_array[unique_go.index(val_go3[i][j])] = 1

    true_labels.append(temp_array)

########################################################################################################################
## Processing pred Dataset
########################################################################################################################

pred_id = list(proteinfer_pred_df['ID'])
pred_go = list(proteinfer_pred_df['go_terms'])

# Arranging datapoints:
pred_go1 = []
pred_id1 = []
for i in range(len(val_id)):
    pred_id1.append(val_id[i])
    pred_go1.append(pred_go[pred_id.index(val_id[i])])


# Cleaning GO terms:
pred_go2 = []
for i in range(len(pred_go1)):
    if pred_go1[i] != '[]':
        pred_go2.append(pred_go1[i].replace('[','').replace(']','').replace('\'','').replace(' ', '').split(','))
    else:
        pred_go2.append([])

# Keeping only top100 go terms:
pred_go3 = []
for i in range(len(pred_go2)):
    go_temp = []
    for j in range(len(pred_go2[i])):
        if pred_go2[i][j] in unique_go:
            go_temp.append(pred_go2[i][j])
    pred_go3.append(go_temp)

# getting rid of repreated go terms:
pred_go4 = []
for i in range(len(pred_go3)):
    pred_go4.append(list(set(pred_go3[i])))

# making numpy encoding for pred set:
pred_labels = []
for i in range(len(pred_go4)):
    temp_array = np.zeros((len(unique_go)))
    for j in range(len(pred_go4[i])):
        temp_array[unique_go.index(pred_go4[i][j])] = 1

    pred_labels.append(temp_array)

########################################################################################################################
## Computing Accuracy
########################################################################################################################
def compute_metrics(eval_preds):
    # print("PROBLEM!!!")
    # metric = load_metric("accuracy")
    # sdfdf
    # metric = load_metric("accuracy")
    logits, labels = eval_preds
    # logits_main.append(logits)
    # labels_main.append(labels)
    logits1 = (logits>0.5)
    values = (logits1 == labels).flatten()
    acc = float(np.sum(values)/len(values))

    # predictions = np.argmax(logits, axis=-1)
    # labels_real = np.argmax(labels, axis=1)
    # dfdfdfd
    # return metric.compute(predictions=predictions, references=labels_real)
    return {'accuracy': acc}

correct = 0
total = len(unique_go)*len(pred_labels)
for i in range(len(pred_labels)):
    correct += np.sum((pred_labels[i] == true_labels[i]))

print("Proteinfer Accuracy: ", correct/total)

