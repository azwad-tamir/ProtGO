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
# final_pd.to_csv('./data/proteinfer_ens_random_final_pd.csv', index=False)

# with open('./data/proteinfer_ens_random_final_pd.pkl', 'rb') as fp:
#     final_pd = pickle.load(fp)


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

# min_decision_threshold_list = [5e-3, 7e-3, 1e-2, 2e-2, 5e-2, 1e-1, 0.5e-1]
min_decision_threshold = 0.1
seq_id = list(final_pd['sequence_name'])
pred_scores = np.array(list(final_pd['predictions']))
go_terms = []
for i in range(len(pred_scores)):
    if i%5000 == 0:
        print('Iter: ', i)
    go_temp = []
    for j in range(pred_scores.shape[1]):
        if pred_scores[i,j] > min_decision_threshold:
            go_temp.append(vocab_rand[j])

    go_terms.append(go_temp)

# Saving proteinfer inference resuls as pandas dataframe:
proteinfer_random_pred_dict = {'ID': seq_id, 'go_terms': go_terms}
proteinfer_random_pred_df = pd.DataFrame(proteinfer_random_pred_dict)
proteinfer_random_pred_df.to_csv('./data/proteinfer_ens_random_test_go_pred.pkl', index=False)



# ########################################################################################################################
# ## Calculating proteinfer accuracy matrices for top100 on the random test dataset:
# ########################################################################################################################
# # Loading proteinfer prediction results:
proteinfer_random_pred_df = pd.read_csv("./data/proteinfer_random_test_go_pred.pkl")
seq_id = list(proteinfer_random_pred_df['ID'])
go_terms_raw = list(proteinfer_random_pred_df['go_terms'])


# go_terms_raw = go_terms
go_terms = []
for i in range(len(go_terms_raw)):
    if go_terms_raw[i] != '[]':
        # t_go_all.extend(t_go[i].replace('[', '').replace(']', '').replace('\'', '').replace(' ', '').split(','))
        go_terms.append(go_terms_raw[i].replace('[', '').replace(']', '').replace('\'', '').replace(' ', '').split(','))
        # go_terms.append
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
        if go_terms[i][j] in cc_top100_go:
            temp_go.append(go_terms[i][j])

    top100_pred.append(temp_go)


# Keeping only the top 100 go terms in the true go labels:
t_top100_go = []
for i in range(len(t_go1)):
    temp_go = []
    # if i%1000 == 0:
    #     print("index: ", i)
    for j in range(len(t_go1[i])):
        if t_go1[i][j] in cc_top100_go:
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
temp1 = 0 # true label has but absent in predgo # false negative
temp2 = 0 # true label has but absent in predgo # false negative
temp3 = 0 # predgo has but absent in true label # false positive
temp4 = 0 # predgo has but absent in ture label # false positive
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
        temp4+=len(top100_pred1[i])

fn = temp1 + temp2
fp = temp3 + temp4
tp = correct

precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1_score = (2*precision*recall)/(precision+recall)

acc = correct/total
# print("Min Decision threshold value:: ", min_decision_threshold)
print("Proteinfer BP Random test Accuracy: ", acc)
print("precision: ", precision)
print("Recall: ", recall)
print("F1_score", f1_score)


pred_count = 0
for i in range(len(top100_pred1)):
    pred_count+=len(top100_pred1[i])

# print("pred_count: ", pred_count)

true_count = 0
for i in range(len(true_labels)):
    true_count+=len(true_labels[i])

# print("True_count: ", true_count)
#
# print("temp2: ", temp2)
# print("temp3: ", temp3)
# print("temp4: ", temp4)

