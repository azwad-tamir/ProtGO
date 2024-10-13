# from copyreg import pickle
import copy
import json

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

# from classifier_trainer_random_BP import all_logits_train, all_labels_train
# from proteinfer_random_acc_calc import f1_score


########################################################################################################################
## Parsing Proteinfer inference results ##
########################################################################################################################

# wget -qN https://storage.googleapis.com/brain-genomics-public/research/proteins/proteinfer/models/zipped_models/noxpd2_cnn_swissprot_go_random_swiss-cnn_for_swissprot_go_random-13731645.tar.gz
# !tar xzf noxpd2_cnn_swissprot_go_random_swiss-cnn_for_swissprot_go_random-13731645.tar.gz
# !wget -qN https://storage.googleapis.com/brain-genomics-public/research/proteins/proteinfer/colab_support/parenthood.json.gz
# !wget -qN https://storage.googleapis.com/brain-genomics-public/research/proteins/proteinfer/blast_baseline/fasta_files/SWISSPROT_RANDOM_GO/eval_test.fasta


def absolute_paths_of_files_in_dir(dir_path):
  files = os.listdir(dir_path)
  return sorted([os.path.join(dir_path, f) for f in files])

def deserialize_inference_result(results_b64):
  """Deserializes an inference result.

  This function is the opposite of serialize_inference_result.

  The full format expected is a
  base-64 encoded ( np compressed array of ( dict of (seq_name: activations))))

  Benefits of this setup:
  - Takes advantage of np compression.
  - Avoids explicit use of pickle (e.g. np.save(allow_pickle)).
  - Is somewhat agnostic to the dictionary contents
    (i.e. you can put whatever you want in the dictionary if we wanted to reuse
    this serialization format)
  - No protos, so no build dependencies for colab.
  - Entries are serialized row-wise, so they're easy to iterate through, and
    it's possible to decode them on the fly.

  Args:
    results_b64: bytes with the above contents.

  Returns:
    tuple of sequence_name, np.ndarray (the activations).

  Raises:
    ValueError if the structured np.array containing the activations doesn't
    have exactly 1 element.
  """

  bytes_io = io.BytesIO(base64.b64decode(results_b64))
  single_pred_dict = dict(np.load(bytes_io))
  if len(single_pred_dict) != 1:
    raise ValueError('Expected exactly one object in the structured np array. '
                     f'Saw {len(single_pred_dict)}')
  sequence_name = list(single_pred_dict.keys())[0]
  activations = list(single_pred_dict.values())[0]
  return sequence_name, activations


def parse_shard(shard_path):
  """Parses file of gzipped, newline-separated inference results.

  The contents of each line are expected to be serialized as in
  `serialize_inference_result` above.

  Args:
    shard_path: file path.

  Yields:
    Tuple of (accession, activation).
  """
  with tf.io.gfile.GFile(shard_path, 'rb') as f:
    with gzip.GzipFile(fileobj=f, mode='rb') as f_gz:
      for line in f_gz:  # Line-by-line.
        yield deserialize_inference_result(line)

run_name = f'go_random_test_ens'
file_shard_names = ['-{:05d}-of-00064.predictions.gz'.format(i) for i in range(64)]
subprocess.check_output(shlex.split(f'mkdir -p ./inference_results/{run_name}/'))

for shard_name in tqdm.tqdm(file_shard_names, position=0, desc="Downloading"):
    subprocess.check_output(
        shlex.split(f'wget https://storage.googleapis.com/brain-genomics-public/research/proteins/proteinfer/swissprot_inference_results/{run_name}/{shard_name} -O ./inference_results/{run_name}/{shard_name}'))

shard_dir_path = f'./inference_results/{run_name}/'
files_to_process = absolute_paths_of_files_in_dir(shard_dir_path)
list_of_shard_results = [parse_shard(f) for f in files_to_process]
final_pd =  pd.DataFrame(list(itertools.chain(*list_of_shard_results)), columns=['sequence_name', 'predictions'])

# with open(r"./data/proteinfer_ens_random_final_pd.pkl", "wb") as output_file:
#     pickle.dump(final_pd, output_file)
#
#
# # Saving proteinfer pred results:
# final_pd.to_csv('./data/proteinfer_ens_clustered_final_pd.csv', index=False)


# # Extracting vocab file as json:
# with open('./data/go_vocab.json') as fp:
#     go_vocab = json.load(fp)

# # Reading final_pd from file:
# final_pd = pd.read_csv("./data/proteinfer_ens_random_final_pd.csv")


# Reading random_vocab file:
with open('./data/go_random_vocab.pkl', 'rb') as fp:
    vocab_random = pickle.load(fp)

vocab_rand = []
for i in range(len(vocab_random)):
    vocab_rand.append(str(vocab_random[i]).replace('b\'GO:', '').replace('\'', ''))

min_decision_threshold = 1e-3
seq_id = list(final_pd['sequence_name'])
pred_scores = np.array(list(final_pd['predictions']))
go_terms = []
for i in range(len(pred_scores)):
    if i%1000 == 0:
        print('Iter: ', i)
    go_temp = []
    for j in range(pred_scores.shape[1]):
        if pred_scores[i,j] > min_decision_threshold:
            go_temp.append(vocab_rand[j])

    go_terms.append(go_temp)

# Saving proteinfer inference resuls as pandas dataframe:
proteinfer_random_pred_dict = {'ID': seq_id, 'go_terms': go_terms}
proteinfer_random_pred_df = pd.DataFrame(proteinfer_random_pred_dict)
proteinfer_random_pred_df.to_csv('./data/proteinfer_ens_clustered_test_go_pred.pkl', index=False)



########################################################################################################################
## Calculating proteinfer accuracy matrices for top100 on the random test dataset:
########################################################################################################################
# Loading proteinfer prediction results:
proteinfer_random_pred_df = pd.read_csv("./data/proteinfer_random_test_go_pred.pkl")
seq_id = list(proteinfer_random_pred_df['ID'])
go_terms_raw = list(proteinfer_random_pred_df['go_terms'])

go_terms = []
for i in range(len(go_terms_raw)):
    if go_terms_raw[i] != '[]':
        # t_go_all.extend(t_go[i].replace('[', '').replace(']', '').replace('\'', '').replace(' ', '').split(','))
        go_terms.append(go_terms_raw[i].replace('[', '').replace(']', '').replace('\'', '').replace(' ', '').split(','))
    else:
        go_terms.append([])

# Loading true labels
true_df = pd.read_csv("./random_test_df")
t_id = list(true_df['ID'])
t_go = list(true_df['GO_terms'])

# Loading top100 aspectwise go terms:
with open('./data/random_train_top100.pkl', 'rb') as fp:
    top100_go_list_main = pickle.load(fp)

bp_id = top100_go_list_main[0]
cc_id = top100_go_list_main[1]
mf_id = top100_go_list_main[2]
bp_seq = top100_go_list_main[3]
cc_seq = top100_go_list_main[4]
mf_seq = top100_go_list_main[5]
bp_go = top100_go_list_main[6]
cc_go = top100_go_list_main[7]
mf_go = top100_go_list_main[8]
bp_go_orig = top100_go_list_main[9]
cc_go_orig = top100_go_list_main[10]
mf_go_orig = top100_go_list_main[11]

bp_top100_go = list(set([i for row in bp_go_orig for i in row]))
cc_top100_go = list(set([i for row in cc_go_orig for i in row]))
mf_top100_go = list(set([i for row in mf_go_orig for i in row]))



t_id1 = []
t_go1 = []
t_go_all = []
for i in range(len(t_go)):
    if t_go[i] != '[]':
        # t_go_all.extend(t_go[i].replace('[', '').replace(']', '').replace('\'', '').replace(' ', '').split(','))
        t_go1.append(t_go[i].replace('[', '').replace(']', '').replace('\'', '').replace(' ', '').split(','))
        # t_go1.append(t_go[i])
        t_id1.append(t_id[i])

# # Calculating top 100 go terms in the true labels
# t_unique_go = list(set(t_go_all))
# freq_dist = np.zeros((len(t_unique_go)))
# for i in range(len(t_go_all)):
#     freq_dist[t_unique_go.index(t_go_all[i])] += 1
#
# freq_arg = np.flip(np.argsort(freq_dist))
# top100_go = []
# for i in range(100):
#     top100_go.append(t_unique_go[freq_arg[i]])



# Keeping only the top 100 go terms in the pred go labels:
top100_pred = []
for i in range(len(go_terms)):
    temp_go = []
    # if i%1000 == 0:
    #     print("index: ", i)
    for j in range(len(go_terms[i])):
        if go_terms[i][j] in bp_top100_go:
            temp_go.append(go_terms[i][j])

    top100_pred.append(temp_go)


# Keeping only the top 100 go terms in the true go labels:
t_top100_go = []
for i in range(len(t_go1)):
    temp_go = []
    # if i%1000 == 0:
    #     print("index: ", i)
    for j in range(len(t_go1[i])):
        if t_go1[i][j] in bp_top100_go:
            temp_go.append(t_go1[i][j])

    t_top100_go.append(temp_go)

true_labels = []
for i in range(len(t_top100_go)):
    true_labels.append(list(set(t_top100_go[i])))

top100_pred1 = []
for i in range(len(top100_pred)):
    top100_pred1.append(list(set(top100_pred[i])))

######
# True_id = t_id1, True_labels = true_labels
# Pred_id = seq_id, Pred_labels = top100_pred
#####
total = 0
correct = 0
temp1 = 0
temp2 = 0 # true label has but absent in predgo
temp3 = 0 # predgo has but absent in true label
temp4 = 0
for i in range(len(t_id1)):
    if t_id1[i] not in seq_id:
        total+=len(true_labels[i])
        temp1+=1
    else:
        pred_go = list(set(top100_pred1[seq_id.index(t_id1[i])]))
        for j in range(len(true_labels[i])):
            total += 1
            if true_labels[i][j] in pred_go:
                correct += 1
            else:
                temp2+=1

        for j in range(len(pred_go)):
            if pred_go[j] not in true_labels[i]:
                total += 1
                temp3+=1

for i in range(len(seq_id)):
    if seq_id[i] not in t_id1:
        total += len(top100_pred1[i])
        temp4+=1

print("Proteinfer Random test Accuracy: ", correct/total)


pred_count = 0
for i in range(len(top100_pred1)):
    pred_count+=len(top100_pred1[i])

print("pred_count:: ", pred_count)

true_count = 0
for i in range(len(true_labels)):
    true_count+=len(true_labels[i])

print("True_count:: ", true_count)



# for i in range(len(t_top100_go)):
#     for j in range(len(t_top100_go[i])):
#         if t_top100_go[i][j] not in top100_go:
#             print(t_top100_go[i][j])



########################################################################################################################
# Creating GO to Aspect mapper:
########################################################################################################################
with open("./GO_to_aspect.txt") as fp:
    lines = fp.readlines()

all_values = []
for i in range(len(lines)):
    values = []
    # line = lines[i].replace('\n','')
    if re.findall("\[Term\]", lines[i].replace('\n','')):
        i+=1
        while(lines[i] != '\n'):
            values.append(lines[i].replace('\n',''))
            i+=1

        all_values.append(values)

id_all = []
name_all = []
aspect_all = []
is_a_all = []
alt_id_all = []
is_obsolete_all = []

# Extracting aspect information from dataset files:
for i in range(len(all_values)):
    id_all.append(all_values[i][0].split(': ', 1)[1])
    is_a = []
    alt_id = []
    is_obsolete = "False"
    for j in range(len(all_values[i])):
        if re.findall("name: (.+)", all_values[i][j]):
            name = re.findall("name: (.+)", all_values[i][j])[0]
        if re.findall("namespace: (.+)", all_values[i][j]):
            aspect = re.findall("namespace: (.+)", all_values[i][j])[0]
        if re.findall("is_a: (.+)", all_values[i][j]):
            is_a.append(re.findall("is_a: (.+)", all_values[i][j])[0])
        if re.findall("alt_id: (.+)", all_values[i][j]):
            alt_id.append(re.findall("alt_id: (.+)", all_values[i][j])[0])
        if re.findall("is_obsolete: (.+)", all_values[i][j]):
            is_obsolete = re.findall("is_obsolete: (.+)", all_values[i][j])[0]

    name_all.append(name)
    aspect_all.append(aspect)
    is_a_all.append(is_a)
    alt_id_all.append(alt_id)
    is_obsolete_all.append(is_obsolete)

asp_types = list(set(aspect_all))
asp_count = np.zeros((3))
non_obs_asp_count = np.zeros((3))
for i in range(len(aspect_all)):
    asp_count[asp_types.index(aspect_all[i])] += 1
    if is_obsolete_all[i] == "False":
        non_obs_asp_count[asp_types.index(aspect_all[i])] += 1

non_obsolete_ids = []
for i in range(len(is_obsolete_all)):
    if is_obsolete_all[i] == "False":
        non_obsolete_ids.append(id_all[i])

########################################################################################################################
### Parsing main datasets:
########################################################################################################################
c_dev = pd.read_csv("./clustered_train_df")
all_go_terms = []
c_dev_go = list(c_dev['GO_terms'])

# go_terms are list of all GO terms in the dataset
go_terms = []
for i in range(len(c_dev_go)):
    if c_dev_go[i] != '[]':
        go_terms.extend(c_dev_go[i].replace('[','').replace(']','').replace('\'','').replace(' ', '').split(','))

# c_go is all GO datapoints in the dataset
c_go = []
for i in range(len(c_dev_go)):
    if c_dev_go[i] != '[]':
        c_go.append(c_dev_go[i].replace('[','').replace(']','').replace('\'','').replace(' ', '').split(','))
    else:
        c_go.append([])

alt_id_flat = []
for i in range(len(alt_id_all)):
    alt_id_flat.extend(alt_id_all[i])

# checking if all go terms in the dataset are either in id_all or alt_id_flat
go_terms1 = list(set(go_terms))
for i in range(len(go_terms1)):
    if (('GO:' + go_terms1[i]) not in id_all) and  (('GO:' + go_terms1[i]) not in alt_id_flat):
        print(i)

# Converting c_go which is a list of datapoint go terms into c_go1 where everything is official GO IDs
c_go1 = []
for i in range(len(c_go)):
    if i%1000 == 0:
        print(i)
    if c_go[i] == []:
        c_go1.append([])
    else:
        temp_go_list = []
        for j in range(len(c_go[i])):
            if ('GO:'+c_go[i][j]) in alt_id_flat:
                for k in range(len(alt_id_all)):
                    if ('GO:'+c_go[i][j]) in alt_id_all[k]:
                        temp_go_list.append(id_all[k])
                        break

            else:
                temp_go_list.append('GO:' + c_go[i][j])

        c_go1.append(temp_go_list)

## Checking for repeated go terms in the dataset:
# x1 = 0
# for i in range(len(c_go1)):
#     if len(c_go1[i]) != len(list(set(c_go1[i]))):
#         x1+=1

# Checking whether all c_go1 terms are made with official GO IDs
c_go1_flat = []
for i in range(len(c_go1)):
    if c_go1[i] != []:
        for j in range(len(c_go1[i])):
            if (c_go1[i][j]) not in id_all:
                print(c_go1[i][j])


# Splitting dataset by Aspects:
c_aspect = []
c_isa = []
c_id = list(c_dev['ID'])
go_dis = np.zeros((len(id_all)))
for i in range(len(c_id)):
    if c_go1 != []:
        for j in range(len(c_go1[i])):
            go_dis[id_all.index(c_go1[i][j])] += 1

go_dis_ord = np.flip(np.sort(go_dis))
go_arg = np.flip(np.argsort(go_dis))

id_bp = []
id_mf = []
id_cc = []
for i in range(len(id_all)):
    if aspect_all[i] == 'biological_process':
        id_bp.append(id_all[i])
    elif aspect_all[i] == 'molecular_function':
        id_mf.append(id_all[i])
    elif aspect_all[i] == 'cellular_component':
        id_cc.append(id_all[i])
    else:
        print(aspect_all[i])

counter_bp = 0
counter_cc = 0
counter_mf = 0
# id_<aspect>_short refers to the top 100 official GO terms from the dataset
id_bp_short = []
id_cc_short = []
id_mf_short = []

for i in range(len(go_arg)):
    if (id_all[go_arg[i]] in id_mf) and counter_mf<100:
        id_mf_short.append(id_all[go_arg[i]])
        counter_mf += 1

    if (id_all[go_arg[i]] in id_cc) and counter_cc<100:
        id_cc_short.append(id_all[go_arg[i]])
        counter_cc += 1

    if (id_all[go_arg[i]] in id_bp) and counter_bp < 100:
        id_bp_short.append(id_all[go_arg[i]])
        counter_bp += 1


# Creating aspect wise datasets with the top 100 GO Terms for each aspect
bp_id = []
cc_id = []
mf_id = []
bp_seq = []
cc_seq = []
mf_seq = []
bp_go = []
cc_go = []
mf_go = []
bp_go_orig = []
cc_go_orig = []
mf_go_orig = []
c_seq = list(c_dev['sequence'])

for i in range(len(c_id)):
    bp_has = 0
    cc_has = 0
    mf_has = 0
    for j in range(len(c_go1[i])):
        if c_go1[i][j] in id_bp_short:
            bp_has+=1
        if c_go1[i][j] in id_cc_short:
            cc_has+=1
        if c_go1[i][j] in id_mf_short:
            mf_has+=1

    if bp_has:
        go_temp = []
        go_temp_orig = []
        bp_id.append(c_id[i])
        bp_seq.append(c_seq[i])
        for j in range(len(c_go1[i])):
            if c_go1[i][j] in id_bp_short:
                go_temp.append(c_go1[i][j])
                go_temp_orig.append(c_go[i][j])

        bp_go.append(go_temp)
        bp_go_orig.append(go_temp_orig)

    if mf_has:
        go_temp = []
        go_temp_orig = []
        mf_id.append(c_id[i])
        mf_seq.append(c_seq[i])
        for j in range(len(c_go1[i])):
            if c_go1[i][j] in id_mf_short:
                go_temp.append(c_go1[i][j])
                go_temp_orig.append(c_go[i][j])

        mf_go.append(go_temp)
        mf_go_orig.append(go_temp_orig)

    if cc_has:
        go_temp = []
        go_temp_orig = []
        cc_id.append(c_id[i])
        cc_seq.append(c_seq[i])
        for j in range(len(c_go1[i])):
            if c_go1[i][j] in id_cc_short:
                go_temp.append(c_go1[i][j])
                go_temp_orig.append(c_go[i][j])

        cc_go.append(go_temp)
        cc_go_orig.append(go_temp_orig)


# go = [item for sublist in bp_go for item in sublist]
# go_unique = list(set(go))
# go_dist = np.zeros((len(go_unique)))
# for i in range(len(go)):
#     go_dist[go_unique.index(go[i])] += 1
#
# print(np.flip(np.sort(go_dist)))


all_data = [bp_id, cc_id, mf_id, bp_seq, cc_seq, mf_seq, bp_go, cc_go, mf_go, bp_go_orig, cc_go_orig, mf_go_orig]

with open(r"./data/clustered_train_top100.pkl", "wb") as output_file:
    pickle.dump(all_data, output_file)


# ########################################################################################################################
# ## Creating official GO_id to original GO_id mapper
# ########################################################################################################################
# official_go_list_all_raw = [i for row in c_go1 for i in row]
# off_go_list_all = []
# for i in range(len(official_go_list_all_raw)):
#     off_go_list_all.append(official_go_list_all_raw[i].replace('GO:',''))
#
# orig_go_list_all = [i for row in c_go for i in row]
# map_orig_list = []
# map_off_list = []
# iter = 0
# for i in range(len(orig_go_list_all)):
#     if orig_go_list_all[i] != off_go_list_all[i]:
#         iter+=1
#         map_orig_list.append(orig_go_list_all[i])
#         map_off_list.append(off_go_list_all[i])
#

## Calculating accuracy scores of ProtGO:
with open(r"./results/r_random_top100_CC.pkl", "rb") as input_file:
    all_pred = pickle.load(input_file)

# [all_logits_train, all_labels_train, all_logits_array, all_labels_array, val_seq_list]
all_logits_array = all_pred[2]
all_labels_array = all_pred[3]
val_seq_list = all_pred[4]

tp = 0
fp = 0
fn = 0
tn = 0
logits1 = (all_logits_array > 0.5)
logits2 = logits1.flatten()
# print("Logits and labels: ", logits1.shape, labels.shape)
labels1 = all_labels_array.flatten()

for i in range(len(logits2)):
    if (labels1[i] == 1) and (logits2[i] == 1):
        tp+=1
    elif (logits2[i] == 1) and (labels1[i] == 0):
        fp+=1
    elif (logits2[i] == 0) and (labels1[i] == 1):
        fn+=1
    else:
        tn+=1

precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1_score = (2*precision*recall)/(precision+recall)

print("Precision: ", precision)
print("Recall: ", recall)
print("F1_score", f1_score)

########################################################################################################################
#### Plotting ROC curves :: Micro average Dataset1 and Dataset2 ########################################################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

## Loading pred results of ProtGO:
with open(r"./results/r_random_top100_BP.pkl", "rb") as input_file:
    all_pred = pickle.load(input_file)

# [all_logits_train, all_labels_train, all_logits_array, all_labels_array, val_seq_list]
all_logits_BP = all_pred[2]
all_labels_BP = all_pred[3]

with open(r"./results/r_random_top100_MF.pkl", "rb") as input_file:
    all_pred = pickle.load(input_file)

# [all_logits_train, all_labels_train, all_logits_array, all_labels_array, val_seq_list]
all_logits_MF = all_pred[2]
all_labels_MF = all_pred[3]

with open(r"./results/r_random_top100_CC.pkl", "rb") as input_file:
    all_pred = pickle.load(input_file)

# [all_logits_train, all_labels_train, all_logits_array, all_labels_array, val_seq_list]
all_logits_CC = all_pred[2]
all_labels_CC = all_pred[3]

# # y_test and y_score: (samples, num_classes)
# fpr_train, tpr_train, _ = roc_curve(y_train.ravel(), score_train.ravel())
# roc_auc_train = auc(fpr_train, tpr_train)


fpr_MF, tpr_MF, _ = roc_curve(all_labels_MF.ravel(), all_logits_MF.ravel())
roc_auc_MF = auc(fpr_MF, tpr_MF)

fpr_CC, tpr_CC, _ = roc_curve(all_labels_CC.ravel(), all_logits_CC.ravel())
roc_auc_CC = auc(fpr_CC, tpr_CC)

fpr_BP, tpr_BP, _ = roc_curve(all_labels_BP.ravel(), all_logits_BP.ravel())
roc_auc_BP = auc(fpr_BP, tpr_BP)

# Plot the ROC curve
plt.figure()
plt.plot(fpr_MF, tpr_MF, color='green', lw=1, linestyle = 'solid', label='MF:: AUC = %0.3f' % roc_auc_MF)
plt.plot(fpr_BP, tpr_BP, color='blue', lw=1, linestyle = 'dotted', label='BP:: AUC = %0.3f' % roc_auc_BP)
plt.plot(fpr_CC, tpr_CC, color='red', lw=1, linestyle = 'dashed', label='CC:: AUC = %0.3f' % roc_auc_CC)
# plt.plot(fpr_s2, tpr_s2, color='black', lw=1, linestyle = 'dashdot',  label='U1:: AUC = %0.3f' % roc_auc_s2)
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle=(0, (5, 1)))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Split ROC Curve')
plt.legend(loc="lower right")
plt.show(block=True)


########################################################################################################################
## Plotting ROC curve :: Individual ####################################################################################
from sklearn.metrics import roc_curve, auc
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

num_samples = all_labels_array.shape[0]
logits1 = (all_logits_array > 0.5)

# Compute ROC curve and ROC area for each class
n_classes = all_labels_array.shape[1]
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(logits1[:, i], all_labels_array[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

colors = ['r', 'b', 'g', 'm', 'navy']
line = ['solid', 'dotted', 'dashed', 'dashdot', (0, (1, 1))]
# Plot of a ROC curve for a specific class
plt.figure()
for i in range(5):
    plt.plot(fpr[i], tpr[i], label=str(i) + ': AUC = %0.3f' % roc_auc[i], color=colors[i], linestyle=line[i])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Dataset 1: U1 split')
    plt.legend(loc="lower right")

plt.plot([0, 1], [0, 1], color='navy', linestyle = (0, (5, 1)))
plt.show(block=True)

count = np.zeros((n_classes))
for i in range(len(target_s1)):
    count[target_s1[i]] += 1


########################################################################################################################
## Sequence Length Analysis ############################################################################################

# Loading true labels
val_df = pd.read_csv("./random_test_df")
val_id = list(val_df['ID'])
val_seq = list(val_df['sequence'])
val_go = list(val_df['GO_terms'])

# Plotting seq len dist histogram:
val_seq_len = []
for i in range(len(val_seq)):
    val_seq_len.append(len(val_seq[i]))

# plt.figure()
# plt.hist(val_seq_len, bins=100, edgecolor = 'black')
# plt.xlim(0, 2500)
# plt.xlabel('Sequence Length')
# plt.ylabel('Frequency')
# plt.title('Distribution of Sequence length in the Dataset')
# plt.show(block=True)


## Loading pred results of ProtGO:
with open(r"./results/r_random_top100_BP.pkl", "rb") as input_file:
    all_pred = pickle.load(input_file)

# [all_logits_train, all_labels_train, all_logits_array, all_labels_array, val_seq_list]
all_logits_BP = all_pred[2]
all_labels_BP = all_pred[3]
seq_BP = all_pred[4]

with open(r"./results/r_random_top100_MF.pkl", "rb") as input_file:
    all_pred = pickle.load(input_file)

# [all_logits_train, all_labels_train, all_logits_array, all_labels_array, val_seq_list]
all_logits_MF = all_pred[2]
all_labels_MF = all_pred[3]
seq_MF = all_pred[4]

with open(r"./results/r_random_top100_CC.pkl", "rb") as input_file:
    all_pred = pickle.load(input_file)

# [all_logits_train, all_labels_train, all_logits_array, all_labels_array, val_seq_list]
all_logits_CC = all_pred[2]
all_labels_CC = all_pred[3]
seq_CC = all_pred[4]

seq_total = list(set(seq_BP + seq_MF + seq_CC))

seq_len_BP = []
for i in range(len(seq_BP)):
    seq_len_BP.append(len(seq_BP[i]))

seq_len_MF = []
for i in range(len(seq_MF)):
    seq_len_MF.append(len(seq_MF[i]))

seq_len_CC = []
for i in range(len(seq_CC)):
    seq_len_CC.append(len(seq_CC[i]))

# Generating BP values:
x_values_BP = []
y_values_BP = []
for i in range(0,1800,100):
    lower = i
    upper = i+100
    target = []
    pred = []
    x_values_BP.append(lower)
    x_values_BP.append(upper-0.1)
    for j in range(len(seq_len_BP)):
        if (seq_len_BP[j] < upper) and (seq_len_BP[j] >= lower):
            target.append(all_labels_BP[j])
            pred.append(all_logits_BP[j])

    logits1 = (np.array(pred)>0.5)
    logits2 = logits1.flatten()
    labels1 = np.array(target).flatten()

    total = 0
    correct = 0
    for i in range(len(logits2)):
        if (logits2[i] == 1) or (labels1[i] == 1):
            total += 1
            if logits2[i] == labels1[i]:
                correct+=1

    acc = correct/total
    y_values_BP.append(acc)
    y_values_BP.append(acc)

# Generating MF values:
x_values_MF = []
y_values_MF = []
for i in range(0,1800,100):
    lower = i
    upper = i+100
    target = []
    pred = []
    x_values_MF.append(lower)
    x_values_MF.append(upper-0.1)
    for j in range(len(seq_len_MF)):
        if (seq_len_MF[j] < upper) and (seq_len_MF[j] >= lower):
            target.append(all_labels_MF[j])
            pred.append(all_logits_MF[j])

    logits1 = (np.array(pred)>0.5)
    logits2 = logits1.flatten()
    labels1 = np.array(target).flatten()

    total = 0
    correct = 0
    for i in range(len(logits2)):
        if (logits2[i] == 1) or (labels1[i] == 1):
            total += 1
            if logits2[i] == labels1[i]:
                correct+=1

    acc = correct/total
    y_values_MF.append(acc)
    y_values_MF.append(acc)

# Generating CC values:
x_values_CC = []
y_values_CC = []
for i in range(0,1800,100):
    lower = i
    upper = i+100
    target = []
    pred = []
    x_values_CC.append(lower)
    x_values_CC.append(upper-0.1)
    for j in range(len(seq_len_CC)):
        if (seq_len_CC[j] < upper) and (seq_len_CC[j] >= lower):
            target.append(all_labels_CC[j])
            pred.append(all_logits_CC[j])

    logits1 = (np.array(pred)>0.5)
    logits2 = logits1.flatten()
    labels1 = np.array(target).flatten()

    total = 0
    correct = 0
    for i in range(len(logits2)):
        if (logits2[i] == 1) or (labels1[i] == 1):
            total += 1
            if logits2[i] == labels1[i]:
                correct+=1

    acc = correct/total
    y_values_CC.append(acc)
    y_values_CC.append(acc)


# Plotting seq len analysis results
plt.figure()
# plt.plot(seq_x_train, seq_y_train, label='Training', color='b', linestyle = 'solid')
plt.plot(x_values_MF, y_values_MF, label='MF', color='r', linestyle = 'solid')
plt.plot(x_values_BP, y_values_BP, label='BP', color='g', linestyle = 'dashed')
plt.plot(x_values_CC, y_values_CC, label='CC', color='k', linestyle = 'dotted')

plt.xlim([0, 1900])
plt.ylim([0.0, 1.05])
plt.xlabel('Sequence Length')
plt.ylabel('Accuracy')
plt.title('Input Sequence length variability of Accuracy')
plt.legend(loc="lower right")
plt.show(block=True)

