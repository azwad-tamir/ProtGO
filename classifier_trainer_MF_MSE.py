"""
Using BCEwithLogitsLoss
"""

import copy
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import DistilBertModel, BertModel, BertTokenizer, BertForMaskedLM, BertForPreTraining, \
    BertForSequenceClassification, AutoModelForSequenceClassification
import torch
from datasets import load_dataset, Dataset
from transformers import DataCollatorWithPadding, DataCollatorForLanguageModeling
from transformers import TrainingArguments
import pandas as pd
from datasets import load_metric
import random
from torch import nn
from transformers import Trainer
import os
from transformers import AutoConfig, EarlyStoppingCallback
from transformers.models.bert import BertPreTrainedModel
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, BCELoss
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.trainer_utils import IntervalStrategy
import shutil
import time
from sklearn import preprocessing
from collections import Counter
from sklearn.metrics import confusion_matrix, mean_squared_error, classification_report
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score, roc_auc_score, precision_score, recall_score, average_precision_score
import pickle
import pandas as pd

mode = 0
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
random.seed(34)

########################################################################################################################
## Processing ProtGO dataset:    #######################################################################################
########################################################################################################################

with open(r"./data/random_train_top100.pkl", "rb") as input_file:
    all_data = pickle.load(input_file)

trian_id = all_data[0]
train_seq = all_data[3]
train_go = all_data[6]
train_go_orig = all_data[9]
# all_data = [bp_id, cc_id, mf_id, bp_seq, cc_seq, mf_seq, bp_go, cc_go, mf_go, bp_go_orig, cc_go_orig, mf_go_orig]

# Making unique go list for hot encoding the go terms in the dataset into labels:
unique_go = list(set([i for row in train_go for i in row]))

# processing the sequences:
train_seq1 = []
for i in range(len(train_seq)):
    train_seq1.append(' '.join(train_seq[i]))



########################################################################################################################
## Processing validation set:

# Loading true labels
val_df = pd.read_csv("./random_test_df")
val_id = list(val_df['ID'])
val_seq = list(val_df['sequence'])
val_go = list(val_df['GO_terms'])

# processing the sequences:
val_seq1 = []
for i in range(len(val_seq)):
    val_seq1.append(' '.join(val_seq[i]))

# Cleaning GO terms and getting rid of empty go term datapoints:
val_go2 = []
val_seq2 = []
val_id2 = []
for i in range(len(val_go)):
    if val_go[i] != '[]':
        val_go2.append(val_go[i].replace('[','').replace(']','').replace('\'','').replace(' ', '').split(','))
        val_seq2.append(val_seq1[i])
        val_id2.append(val_id[i])

# Creating original to official go mapper
go_map = {}
train_go_orig_flat = [i for row in train_go_orig for i in row]
train_go_flat = [i for row in train_go for i in row]
for i in range(len(train_go_orig_flat)):
    go_map[train_go_orig_flat[i]] = train_go_flat[i]

# Converting val go terms to official ones while only keeping top 100:
val_go3 = []
val_id3 = []
val_seq3 = []
orig_go_unique = list(go_map.keys())
for i in range(len(val_go2)):
    temp_go = []
    for j in range(len(val_go2[i])):
        if val_go2[i][j] in orig_go_unique:
            temp_go.append(go_map[val_go2[i][j]])

    if temp_go != []:
        val_go3.append(list(set(temp_go)))
        val_id3.append(val_id2[i])
        val_seq3.append(val_seq2[i])

'''
Training texts: train_seq1
Training labels: train_go
Training IDs: train_id
Validation texts: val_seq3
validation labels: val_go3
validation IDs: val_id3
'''

########################################################################################################################
## Encoding the training and validation data:

# Getting rid of repeated go terms in one datapoint
train_go1 = []
for i in range(len(train_go)):
    train_go1.append(list(set(train_go[i])))

# Encoding train labels:
train_labels = []
for i in range(len(train_go1)):
    temp_labels = np.zeros((len(unique_go)))
    for j in range(len(train_go1[i])):
        temp_labels[unique_go.index(train_go1[i][j])] += 1

    train_labels.append(temp_labels)

## checking to see if multiple repeated go terms are present in the dataset:
#x=0
# for i in range(len(train_labels)):
#     for j in range(len(train_labels[i])):
#         if train_labels[i][j] > 1:
#             x+=1

# Encoding val labels:
val_labels = []
for i in range(len(val_go3)):
    temp_labels = np.zeros((len(unique_go)))
    for j in range(len(val_go3[i])):
        temp_labels[unique_go.index(val_go3[i][j])] += 1

    val_labels.append(temp_labels)

# # checking to see if multiple repeated go terms are present in the dataset:
# x=0
# for i in range(len(val_labels)):
#     for j in range(len(val_labels[i])):
#         if val_labels[i][j] > 1:
#             x+=1



# # Analyzing GO term frequency distribution
# go = [item for sublist in bp_go for item in sublist]
# go_unique = list(set(go))
# go_dist = np.zeros((len(go_unique)))
# for i in range(len(go)):
#     go_dist[go_unique.index(go[i])] += 1
#
# # print(np.flip(np.sort(go_dist)))
# go_dist_arg = np.flip(np.argsort(go_dist))
# go_top10 = []
# for i in range(10):
#     go_top10.append(go_unique[go_dist_arg[i]])
#
# top10_id = []
# top10_seq = []
# top10_go = []
# for i in range(len(bp_id)):
#     go_temp = []
#     for j in range(len(bp_go[i])):
#         if bp_go[i][j] in go_top10:
#             go_temp.append(bp_go[i][j])
#
#     if go_temp != []:
#         top10_id.append(bp_id[i])
#         top10_seq.append(bp_seq[i])
#         top10_go.append(go_temp)
#
#
#
# top10seq1 = []
# for i in range(len(top10_seq)):
#     top10seq1.append(' '.join(top10_seq[i]))
#
# go_map = list(set([item for sublist in top10_go for item in sublist]))
# go_labels = []
# for i in range(len(top10_id)):
#     label = np.zeros((len(go_map)))
#     # if len(top10_go[i]) != len(list(set(top10_go[i]))):
#     #     print(top10_go[i])
#     for j in range(len(top10_go[i])):
#         label[go_map.index(top10_go[i][j])] = 1
#
#     go_labels.append(label)


# text_total = copy.deepcopy(top10seq1)
# labels_total = copy.deepcopy(go_labels)
#
# text_train, text_test, labels_train, labels_test = train_test_split(text_total, labels_total, test_size=0.1)
# # # text_test1, text_test2, labels_test1, labels_test2 = train_test_split(text_test, labels_test, test_size=0.1)

label_num = len(train_labels[0])

dict_train = {'text':train_seq1,'labels':train_labels}
dict_test = {'text':val_seq3,'labels':val_labels}
########################################################################################################################

# freq = np.sum(labels_total, axis=0)

# logits_all = []
# labels_all = []

########################################################################################################################
############################################### Building Model: ########################################################

class MyBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        if config.intermediate_hidden_size != 0:
            self.intermediate_classifier = nn.Linear(config.hidden_size, config.intermediate_hidden_size)
            self.classifier = nn.Linear(config.intermediate_hidden_size, config.num_labels)
        else:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.config.use_pooler == 1:
            pooled_output = outputs[1]
        elif self.config.use_mean == 1:
            # print(attention_mask.shape)
            # token_embeddings.sum(axis=1) / attention_mask.sum(axis=-1).unsqueeze(-1)
            pooled_output = torch.sum(outputs[0] * attention_mask.unsqueeze(-1), dim=1) / torch.sum(attention_mask, dim=1).unsqueeze(-1)
            # print(pooled_output.shape)
            # pooled_output = torch.mean(outputs[0], dim=1)
        else:
            pooled_output = outputs[0][:, 0]

        pooled_output = self.dropout(pooled_output)
        if config.intermediate_hidden_size != 0:
            intermediate_output = self.intermediate_classifier(pooled_output)
            logits = self.classifier(intermediate_output)
        else:
            logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                # loss_fct = BCEWithLogitsLoss()
                # loss_fct = nn.BCELoss()
                loss_fct = MSELoss()
                # loss_fct = CrossEntropyLoss()
                # print(list(logits.size()))
                # print(list(labels.size()))
                # print(logits.dtype)
                # print(labels.dtype)
                labels = labels.type(torch.float32)
                # print("logits: ", logits.shape)
                # print("labels: ", labels.shape)
                # logits_all.append(logits)
                # labels_all.append(labels)
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# print(loss_fct(torch.tensor([0.4,0.4,0.0,0.0]), torch.tensor([1.0,0.0,0.0,0.0])))
# print(loss_fct(torch.tensor([0.0,1.0,0.0,0.0]), torch.tensor([0.0,1.0,0.0,0.0])))
# print(loss_fct(torch.tensor([1.0,1.0,0.0,0.0]), torch.tensor([0.0,1.0,0.0,0.0])))
# print(loss_fct(torch.tensor([0.5,0.7,-0.1,0.3]), torch.tensor([1.0,1.0,1.0,1.0])))

def tokenize_function(example):

    return tokenizer(example["text"], add_special_tokens=True, truncation=True, max_length=1024)


logits_main = []
labels_main = []
def compute_metrics_old(eval_preds):
    # print("PROBLEM!!!")
    metric = load_metric("accuracy")
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

def compute_metrics(eval_preds):
    # print("PROBLEM!!!")
    metric = load_metric("accuracy")
    # sdfdf
    # metric = load_metric("accuracy")
    logits, labels = eval_preds
    # logits_main.append(logits)
    # labels_main.append(labels)
    logits1 = (logits>0.5)
    logits2 = logits1.flatten()
    # print("Logits and labels: ", logits1.shape, labels.shape)
    labels1 = labels.flatten()
    total = 0
    correct = 0
    for i in range(len(logits2)):
        if (logits2[i] == 1) or (labels1[i] == 1):
            total += 1
            if logits2[i] == labels1[i]:
                correct+=1

    # values = (logits1 == labels).flatten()
    acc = correct/total

    # predictions = np.argmax(logits, axis=-1)
    # labels_real = np.argmax(labels, axis=1)
    # dfdfdfd
    # return metric.compute(predictions=predictions, references=labels_real)
    return {'accuracy': acc}

model_type = "Rostlab/prot_bert_bfd"
# model_type = "Rostlab/prot_bert"
# model_type = "bert-base-cased"
# model_type = "bert-base-uncased"
# model_type = "distilbert-base-uncased"

# do_lower_case = True
do_lower_case = False
tokenizer = BertTokenizer.from_pretrained(model_type, do_lower_case=do_lower_case)
# elif (a==str(0) or a==str(1) or a==str(2) or a==str(3) or a==str(4) or a==str(5) or a==str(6)):
# tokenizer.add_tokens(["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24"])



##########################################################################################################################
# import torch
# from transformers import BertTokenizer, BertModel
#
# tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
# model = BertModel.from_pretrained("bert-base-cased")
#
# print(len(tokenizer))  # 28996
# tokenizer.add_tokens(["NEW_TOKEN1"])
# print(len(tokenizer))  # 28997
#
# model.resize_token_embeddings(len(tokenizer))
# # The new vector is added at the end of the embedding matrix
#
# print(len(model.embeddings.word_embeddings.weight[-1, :]))
# # Randomly generated matrix
#
# model.embeddings.word_embeddings.weight[-1, :] = torch.zeros([model.config.hidden_size])
#
# print(len(model.embeddings.word_embeddings.weight[-1, :]))
# # outputs a vector of zeros of shape [768]

########################################################################################################################

# model = BertForSequenceClassification.from_pretrained("Rostlab/prot_bert_bfd")
config = AutoConfig.from_pretrained(model_type)
# subject = 'secreted'
# subject = 'iamppred'
# subject = 'ampscanner'
# subject = 'iamp2l'
# subject = 'ampep'
# subject = 'secproct'
# subject = 'hemolythic'
# subject = 'hlppredfuse'
# subject = 'rnnamp'
subject = 'EC_num'

config.classifier_dropout = 0
config.hidden_dropout_prob = 0
# config.hidden_size = 1024

mode_hidden_map = {0:0, 1:32, 2:128, 3:1024}
# config.intermediate_hidden_size = 1024
# config.intermediate_hidden_size = 128
# config.intermediate_hidden_size = 32
# config.intermediate_hidden_size = 0
config.intermediate_hidden_size = mode_hidden_map[mode]


########################################################################################################################
##################################################### Training: ##########################################################
# num_epochs = 1
# num_epochs = 10
num_epochs = 5
# num_epochs = 100

config.use_pooler = 0
config.use_mean = 1


# Enabling multilabel classification:
config.num_labels = label_num
config.problem_type = "multi_label_classification"

# freeze_positional = 0
# freeze_non_positional = 0
# freeze_attention = 0
# freeze_layer_norm = 0
# freeze_pooler = 0

# freeze_positional = 1
# freeze_non_positional = 1
# freeze_attention = 1
# freeze_layer_norm = 1
# freeze_pooler = 1

freeze_positional = 1 #0
freeze_non_positional = 1
freeze_attention = 1
freeze_layer_norm = 0
freeze_pooler = 0

transfer = 0
random_init = 0
if random_init:
    transfer = 0
if subject == 'secreted':
    transfer = 0

early_stopping = 1
patience = 10
if early_stopping:
    create_validation_split = 1
else:
    create_validation_split = 0
ten_fold = 0

# monitor_value = ''
# initial_lr = 5e-6
# initial_lr = 5e-5
initial_lr = 5e-4
if subject == 'iamp2l' or subject == 'iamppred':
    batch_size = 32
if subject == 'secreted' or subject == 'ampscanner' or subject == 'hemolythic' or subject == 'hemolythic_2021' or subject == 'hlppredfuse' or subject == 'rnnamp':
    batch_size = 16
if subject == 'ampep' or subject == 'secproct':
    batch_size = 8
if subject == 'secproct':
    batch_size = 4
if subject == 'EC_num':
    batch_size = 1
balanced_loss = 0

# batch_size = 4
losses_all = []
fold_range = [0]
if ten_fold:
    fold_range = range(10)

for fold in fold_range:
    model = None
    dataset = None
    if transfer == 0:
        if random_init:
            model = MyBertForSequenceClassification(config=config)
        else:
            model = MyBertForSequenceClassification.from_pretrained(model_type, config=config)
    if transfer == 1:
        results_df = pd.read_csv('results/training_results.csv')
        results_df = results_df[results_df['subject'] == 'secreted']
        results_df = results_df[results_df['hidden_layer_size'] == config.intermediate_hidden_size]
        results_df = results_df[results_df['usemean'] == config.use_mean]
        results_df = results_df[results_df['usepooler'] == config.use_pooler]
        secreted_model_dir = results_df.iloc[0]['save_dir']
        model = MyBertForSequenceClassification.from_pretrained(secreted_model_dir, config=config)

    # Loading Checkpoint:
    # model = MyBertForSequenceClassification.from_pretrained("./models/0706-172025/checkpoint-4350")

    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # model.to(device)
    # dataset = load_dataset('text', data_files={'train': 'data/uniparc_peptides_spaced.txt'}, cache_dir='data/processed_datasets')
    # dataset = load_dataset('csv', data_files={'train': 'old/input/amp.csv'},
    #                        # cache_dir='data/processed_datasets',
    #                        delimiter=',',
    #                        # task="text-classification"
    #                        )

    # CSV file loader, 2 columns, 'text' and 'labels'. Text is spaced capital sequences.
    if subject == 'iamp2l':
        dataset = load_dataset('csv', data_files={'train': 'data/iamp2l/iamp2l_raw.csv', 'test': 'data/iamp2l/independent/independent.csv'},
                               cache_dir='data/processed_datasets',
                               delimiter=',',
                               # task="text-classification"
                               )
    elif subject == 'iamppred':
        dataset = load_dataset('csv', data_files={'train': 'data/iamppred/old/train.csv', 'test': 'data/iamppred/old/test.csv'},
                               cache_dir='data/processed_datasets',
                               delimiter='\t',
                               # task="text-classification"
                               )
    elif subject == 'secreted':
        dataset = load_dataset('csv', data_files={'train': 'data/swissprot/secreted_all_cleaned_all_train.csv', 'test': 'data/swissprot/secreted_all_cleaned_all_test.csv'},
                               cache_dir='data/processed_datasets',
                               delimiter=',',
                               # task="text-classification"
                               )

    elif subject == 'ampscanner':
        dataset = load_dataset('csv', data_files={'train': 'data/ampscanner/ampscanner_train.csv',
                                                  'test': 'data/ampscanner/ampscanner_test.csv'},
                               cache_dir='data/processed_datasets',
                               delimiter=',',
                               # task="text-classification"
                               )

    elif subject == 'ampep':
        dataset = load_dataset('csv', data_files={'train': 'data/ampep/ampep.csv'},
                               cache_dir='data/processed_datasets',
                               delimiter=',',
                               )

    elif subject == 'secproct':
        dataset = load_dataset('csv', data_files={'train': 'data/secproct/blood_train.csv',
                                                  'test': 'data/secproct/blood_test.csv'},
                               cache_dir='data/processed_datasets',
                               delimiter=',',
                               # task="text-classification"
                               )
    elif subject == 'hemolythic':
        dataset = load_dataset('csv', data_files={'train': 'data/hemolythic/hemolythic_train.csv',
                                                  'test': 'data/hemolythic/hemolythic_test.csv'},
                               cache_dir='data/processed_datasets',
                               delimiter=',',
                               # task="text-classification"
                               )
    elif subject == 'hemolythic_2021':
        dataset = load_dataset('csv', data_files={'train': 'data/hemolythic_2021/hemolythic_2021_train.csv',
                                                  'test': 'data/hemolythic_2021/hemolythic_2021_test.csv'},
                               cache_dir='data/processed_datasets',
                               delimiter=',',
                               # task="text-classification"
                               )
    elif subject == 'hlppredfuse':
        dataset = load_dataset('csv', data_files={'train': 'data/hlppredfuse/hlppredfuse_train.csv',
                                                  'test': 'data/hlppredfuse/hlppredfuse_test.csv'},
                               cache_dir='data/processed_datasets',
                               delimiter=',',
                               # task="text-classification"
                               )
    elif subject == 'rnnamp':
        dataset = load_dataset('csv', data_files={'train': 'data/rnnamp/rnnamp_train.csv',
                                                  'test': 'data/rnnamp/rnnamp_test.csv'},
                               cache_dir='data/processed_datasets',
                               delimiter=',',
                               # task="text-classification"
                               )

    elif subject == 'EC_num':
        # dataset = Dataset.from_pandas(df)
        dataset_train = Dataset.from_dict(dict_train, split='train')
        dataset_test = Dataset.from_dict(dict_test, split='test')



    tokenized_datasets_train = dataset_train.map(tokenize_function, batched=True)
    tokenized_datasets_test = dataset_test.map(tokenize_function, batched=True)

    tokenized_datasets_train.set_format("torch", columns=['input_ids'])
    tokenized_datasets_test.set_format("torch", columns=['input_ids'])


    tokenized_datasets_train = tokenized_datasets_train.remove_columns('text')
    tokenized_datasets_test = tokenized_datasets_test.remove_columns('text')
    # tokenized_datasets_test2 = tokenized_datasets_test2.remove_columns('text')
    # for c in tokenized_datasets.column_names['train']:
    #     if c in ['keyword', 'length']:
    #         tokenized_datasets = tokenized_datasets.remove_columns(c)

    tokenized_datasets_train.set_format("torch")
    tokenized_datasets_test.set_format("torch")
    # tokenized_datasets_test2.set_format("torch")
    # tokenized_datasets_protein.set_format("torch", columns=['input_ids'])
    # print(tokenized_datasets['train'][1])
    # tokenized_datasets = s.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)

    timestr = time.strftime("%m%d-%H%M%S")
    save_dir = 'models_clustered_EC1_new1/' + timestr + '/'
    while os.path.isdir(save_dir):
        timestr = timestr.split('-')[0] + '-' + timestr.split('-')[1][:4] + str(int(timestr.split('-')[1][4:] + random.randint(1,60)))
        save_dir = 'models_clustered_EC1_new1/' + timestr + '/'

    os.makedirs(save_dir, exist_ok=True)

    model.resize_token_embeddings(len(tokenizer))
    training_args = TrainingArguments(num_train_epochs=num_epochs,
                                      output_dir=save_dir,
                                      per_device_train_batch_size=batch_size,
                                      learning_rate=initial_lr,
                                      load_best_model_at_end=True,
                                      evaluation_strategy=IntervalStrategy.EPOCH,
                                      metric_for_best_model='eval_accuracy',
                                      save_total_limit=patience+1,
                                      save_strategy = IntervalStrategy.EPOCH,
                                      # save_strategy = 'no',
                                      # # prediction_loss_only=True,
                                      gradient_accumulation_steps=int(32/batch_size), eval_accumulation_steps=int(32/batch_size),
                                      # fp16=True, fp16_full_eval=True,
                                      per_device_eval_batch_size=batch_size,
                                      # # debug="underflow_overflow"
                                      )

    param_names = []
    for name, param in model.named_parameters():
        param_names.append(name)

    # positional_embedding_params = ['bert.embeddings.word_embeddings.weight', 'bert.embeddings.position_embeddings.weight',
    #      'bert.embeddings.token_type_embeddings.weight', 'bert.embeddings.LayerNorm.weight',
    #      'bert.embeddings.LayerNorm.bias']
    positional_embedding_params = ['bert.embeddings.position_embeddings.weight']
    non_positional_embedding_params = ['bert.embeddings.word_embeddings.weight', 'bert.embeddings.token_type_embeddings.weight']
    pooler_params = ['bert.pooler.dense.weight', 'bert.pooler.dense.bias']
    classifier_params = ['intermediate_classifier.weight', 'intermediate_classifier.bias', 'classifier.weight', 'classifier.bias']
    layer_norm_params = []
    attention_params = []
    for l in param_names:
        if 'LayerNorm' in l:
            layer_norm_params.append(l)
        elif l not in positional_embedding_params+non_positional_embedding_params+pooler_params+classifier_params:
            attention_params.append(l)
    print(len(positional_embedding_params+non_positional_embedding_params+layer_norm_params+attention_params+pooler_params+classifier_params), len(param_names))
    unfrozen_params = []
    unfrozen_params += classifier_params
    if freeze_positional == 0:
        unfrozen_params += positional_embedding_params
    if freeze_non_positional == 0:
        unfrozen_params += non_positional_embedding_params
    if freeze_layer_norm == 0:
        unfrozen_params += layer_norm_params
    if freeze_pooler == 0:
        unfrozen_params += pooler_params
    if freeze_attention == 0:
        unfrozen_params += attention_params

    frozen_counter = 0
    grad_counter = 0
    for name, param in model.named_parameters():
        if name in unfrozen_params:
            param.requires_grad = True
            grad_counter += len(param.flatten())
        else:
            param.requires_grad = False
            frozen_counter += len(param.flatten())

    print('Frozen parameters:', frozen_counter, grad_counter, grad_counter+frozen_counter, grad_counter*100/(grad_counter+frozen_counter))
    callbacks = []
    if early_stopping == 1:
        callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)]

    # config = AutoConfig.from_pretrained("./models_clustered_EC1_EX/0805-134938")
    # model = MyBertForSequenceClassification.from_pretrained("./models_clustered_EC1_EX/0805-134938", config=config)
    # safe_serialization = True
    trainer = None
    trainer = Trainer(
        model,
        training_args,
        # safe_serialization=True,
        train_dataset=tokenized_datasets_train,
        eval_dataset=tokenized_datasets_test,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        data_collator=data_collator,
        tokenizer = tokenizer
    )

    # Load checkpoint model:
    # checkpoint = torch.load("./models/0706-172025/checkpoint-4350/")

    trainer.train()
    dlfjdklf
    # torch.save(model.state_dict(), './model/classifier_trainer_MF.pkl')

    # trainer_filepath = "./model/test100.model"
    # trainer.model.save_pretrained(trainer_filepath)

    # Loading saved model:
    # model.load_state_dict(torch.load("./model/classifier_trainer_MF.pkl", weights_only=True))

    # config = AutoConfig.from_pretrained("./model/test100.model")
    # model = MyBertForSequenceClassification.from_pretrained("./model/test100.model", config=config)


    # dirs = [x[0] for x in os.walk(save_dir)]
    # for d in dirs:
    #     if 'checkpoint' in d:
    #         shutil.rmtree(d, ignore_errors='True')
    # trainer.save_model()



    print(trainer.evaluate())
    # Loading saved model:
    # config = AutoConfig.from_pretrained("./models_main/0504-052935")
    # model = MyBertForSequenceClassification.from_pretrained("./models_main/0504-052935", config=config)

    all_predictions_train = trainer.predict(test_dataset=tokenized_datasets_train)
    all_labels_train = all_predictions_train[1]
    all_logits_train = all_predictions_train[0]
    metrices_train = compute_metrics([all_logits_train, all_labels_train])
    print("Training Accuracy: ", metrices_train['accuracy'])

    all_predictions_eval = trainer.predict(test_dataset=tokenized_datasets_test)
    all_labels_eval = all_predictions_eval[1]
    all_logits_eval = all_predictions_eval[0]
    metrices_eval = compute_metrics([all_logits_eval, all_labels_eval])
    print("Test Accuracy: ", metrices_eval['accuracy'])

    with open('./results/r_top100_MF.pkl', "wb") as output_file:
        pickle.dump([all_logits_train, all_labels_train, all_logits_eval, all_labels_eval])

    # all_predictions_eval = np.argmax(np.array(all_predictions_eval[0]), axis=-1)
    # accuracy_eval = (np.sum(all_predictions_eval == all_labels_eval)) / len(all_labels_eval)

    # all_predictions_eval2 = trainer.predict(test_dataset=tokenized_datasets_test2)
    # all_labels_eval2 = np.array(all_predictions_eval2[1])
    # all_logits_eval2 = all_predictions_eval2[0]
    # all_predictions_eval2 = np.argmax(np.array(all_predictions_eval2[0]), axis=-1)

    # all_predictions_test = trainer.predict(test_dataset=tokenized_datasets['test'])
    # all_labels_test = np.array(all_predictions_test[1])
    # all_logits_test = all_predictions_test[0]
    # all_predictions_test = np.argmax(np.array(all_predictions_test[0]), axis=-1)
    # prediction_df = pd.DataFrame(data=[[all_labels_train, all_predictions_train, all_logits_train,
    #                                    all_labels_eval, all_predictions_eval, all_logits_eval,
    #                                    all_labels_test, all_predictions_test, all_logits_test]], columns=[
    #     'train_labels', 'train_predictions', 'train_logits',
    #     'eval_labels', 'eval_predictions', 'eval_logits',
    #     'test_labels', 'test_predictions', 'test_logits',
    # ])
    # prediction_df.to_csv(save_dir+'predictions.csv', index=False)
    # print(save_dir)
    # print("Eval Accuracies:  \n")
    # print('Acc:', accuracy_score(all_labels_eval, all_predictions_eval),
    #       ', Recall:', recall_score(all_labels_eval, all_predictions_eval, average="weighted"), ', Precision:', precision_score(all_labels_eval, all_predictions_eval, average="weighted"),
    #       ', ROC:', roc_auc_score(all_labels_eval, all_logits_eval[:,1]), ', MCC:', matthews_corrcoef(all_labels_eval, all_predictions_eval),
    #        ', AP:', average_precision_score(all_labels_eval, all_logits_eval[:,1]),
    #       ', F1:', f1_score(all_labels_eval, all_predictions_eval, average="weighted"))
    #
    # print("Training Accuracies:   \n")
    # print('Acc:', accuracy_score(all_labels_train, all_predictions_train),
    #       ', Recall:', recall_score(all_labels_train, all_predictions_train), ', Precision:', precision_score(all_labels_train, all_predictions_train),
    #       ', ROC:', roc_auc_score(all_labels_train, all_logits_train[:,1]), ', MCC:', matthews_corrcoef(all_labels_train, all_predictions_train),
    #        ', AP:', average_precision_score(all_labels_train, all_logits_train[:,1]),
    #       ', F1:', f1_score(all_labels_train, all_predictions_train, average='weighted'))

    # print('Acc:', accuracy_score(all_labels_test, all_predictions_test),
    #       ', Recall:', recall_score(all_labels_test, all_predictions_test), ', Precision:', precision_score(all_labels_test, all_predictions_test),
    #       ', ROC:', roc_auc_score(all_labels_test, all_logits_test[:,1]), ', MCC:', matthews_corrcoef(all_labels_test, all_predictions_test),
    #        ', AP:', average_precision_score(all_labels_test, all_logits_test[:,1]),
    #       ', F1:', f1_score(all_labels_test, all_predictions_test))
    #
    # print(round(accuracy_score(all_labels_test, all_predictions_test), 4),
    #      round(recall_score(all_labels_test, all_predictions_test), 4), round(precision_score(all_labels_test, all_predictions_test), 4),
    #       round(roc_auc_score(all_labels_test, all_logits_test[:,1]), 4), round(matthews_corrcoef(all_labels_test, all_predictions_test), 4))

    # print('F1:', f1_score(all_labels_test, all_predictions_test), ', Acc:', accuracy_score(all_labels_test, all_predictions_test), ', MCC:',
    #       matthews_corrcoef(all_labels_test, all_predictions_test), ', ROC:', roc_auc_score(all_labels_test, all_logits_test[:,1]),
    #       ', Precision:', precision_score(all_labels_test, all_predictions_test), ', Recall:',
    #       recall_score(all_labels_test, all_predictions_test), ', AP:', average_precision_score(all_labels_test, all_logits_test[:,1]))

    # df = pd.DataFrame(data=[[f1_score(all_labels_eval, all_predictions_eval), accuracy_score(all_labels_eval, all_predictions_eval),
    #                    matthews_corrcoef(all_labels_eval, all_predictions_eval), roc_auc_score(all_labels_eval, all_logits_eval[:,1]),
    #                    precision_score(all_labels_eval, all_predictions_eval),
    #                    recall_score(all_labels_eval, all_predictions_eval), average_precision_score(all_labels_eval, all_logits_eval[:,1]),
    #                    f1_score(all_labels_test, all_predictions_test), accuracy_score(all_labels_test, all_predictions_test),
    #                    matthews_corrcoef(all_labels_test, all_predictions_test), roc_auc_score(all_labels_test, all_logits_test[:,1]),
    #                    precision_score(all_labels_test, all_predictions_test),
    #                    recall_score(all_labels_test, all_predictions_test), average_precision_score(all_labels_test, all_logits_test[:,1]),
    #                    subject, config.use_pooler, config.use_mean, config.intermediate_hidden_size, batch_size,
    #                    freeze_positional, patience, freeze_non_positional, freeze_attention,
    #                    freeze_layer_norm, freeze_pooler, initial_lr, transfer, num_epochs, save_dir, early_stopping, create_validation_split, fold, ten_fold]], columns=[
    #                    'f1_eval', 'accuracy_eval', 'mcc_eval', 'roc_auc_eval', 'precision_eval', 'recall_eval', 'average_precision_score_eval',
    #                    'f1_test', 'accuracy_test', 'mcc_test', 'roc_auc_test', 'precision_test', 'recall_test', 'average_precision_score_test',
    #                    'subject', 'usepooler', 'usemean', 'hidden_layer_size', 'batchsize', 'frozenpositional', 'patience',
    #                    'frozennonpositional', 'frozenattention', 'frozenlayernorm', 'frozenpooler', 'lr', 'transfer', 'num_epochs', 'save_dir',
    #                    'early_stopping', 'create_validation_split', 'fold', 'ten_fold'])
    # if not os.path.isfile('results/training_results.csv'):
    #    df.to_csv('results/training_results.csv', index=False)
    # else: # else it exists so append without writing the header
    #    df.to_csv('results/training_results.csv', index=False, mode='a', header=False)

# fresh_model = MyBertForSequenceClassification.from_pretrained("Rostlab/prot_bert_bfd", config=config)
# new_model = trainer.model
# new_model.to(device='cpu')
# new_model_state_dict = new_model.state_dict()
# for name, param in fresh_model.named_parameters():
#     # if 'attn' in name or 'mlp' in name:
#     new_param = new_model_state_dict[name]
#     if torch.abs(param.data - new_param.data).sum() > 1e-8:
#         print(f'{name} was modified')
#     else:
#         print(f'{name} is unchanged')


# results_df = pd.read_csv('results/training_results.csv')
# results_df = results_df[results_df['subject'] == 'ampep']
# results_df = results_df[results_df['hidden_layer_size'] == 1024]
#
# all_predictions_test = []
# all_logits_test = []
# all_labels_test = []
# for dir in results_df['save_dir']:
#     pred_df = pd.read_csv(dir+'predictions.csv')
#     all_predictions_test.extend(np.array(pred_df['test_predictions']))
#     all_logits_test.extend(np.array(pred_df['test_logits']))
#     all_labels_test.extend(np.array(pred_df['test_labels']))
#
# print(matthews_corrcoef(all_labels_test, all_predictions_test))
#
# print('F1:', f1_score(all_labels_test, all_predictions_test), ', Acc:', accuracy_score(all_labels_test, all_predictions_test), ', MCC:',
#       matthews_corrcoef(all_labels_test, all_predictions_test), ', ROC:', roc_auc_score(all_labels_test, all_logits_test[:,1]),
#       ', Precision:', precision_score(all_labels_test, all_predictions_test), ', Recall:',
#       recall_score(all_labels_test, all_predictions_test), ', AP:', average_precision_score(all_labels_test, all_logits_test[:,1]))
# print(np.mean(np.array(results_df['mcc_test'])))
#
# alpha = dataset_train['text']
# alpha_all = []
# for a in alpha:
#     for b in a:
#         if b not in alpha_all:
#             alpha_all.append(b)
#
#
# data = tokenized_datasets_test['input_ids']
# seq_data = []
# for proteins in data:
#     seq_data.append(proteins.numpy())
#
# # animo_acids = []
# for a in seq_data:
#     for b in a:
#         if int(b) not in animo_acids:
#             animo_acids.append(int(b))
#
#
# data_temp = tokenized_datasets_temp['input_ids']
# seq_data_temp = []
# for proteins in data_temp:
#     seq_data_temp.append(proteins.numpy())
#
# animo_acids_temp = []
# for a in seq_data_temp:
#     for b in a:
#         if int(b) not in animo_acids_temp:
#             animo_acids_temp.append(int(b))
#
# # 22,26
# print(tokenizer.convert_ids_to_tokens(data_temp[0]))
# y = '3 ' + 'A B C '*512
# text_temp = ['1 A B C D E F G H I J K L M N O P Q R S T U V W X Y Z', '2 L Z A F', y]
# dict_temp = {'text':text_temp,'labels':labels_test[0:3]}
# dataset_temp = Dataset.from_dict(dict_temp, split='train')
# all_data = tokenize_function(dataset_test)
# tokenized_datasets_test = copy.deepcopy(dataset_test)
# tokenized_datasets_test = tokenized_datasets_test.add_column('input_ids', all_data[0])
# tokenized_datasets_test = tokenized_datasets_test.add_column('token_type_ids', all_data[2])
# tokenized_datasets_test = tokenized_datasets_test.add_column('attention_mask', all_data[1])
#
# all_data = tokenize_function(dataset_train)
# tokenized_datasets_train = copy.deepcopy(dataset_train)
# tokenized_datasets_train = tokenized_datasets_train.add_column('input_ids', all_data[0])
# tokenized_datasets_train = tokenized_datasets_train.add_column('token_type_ids', all_data[2])
# tokenized_datasets_train = tokenized_datasets_train.add_column('attention_mask', all_data[1])
#
#
# # tokenized_datasets_temp = dataset_temp.map(tokenize_function, batched=True)
#
# tokenized_datasets_train.set_format("torch", columns=['input_ids'])
# tokenized_datasets_train = tokenized_datasets_train.remove_columns('text')
# tokenized_datasets_train.set_format("torch")
#
# print(tokenizer.convert_ids_to_tokens(tokenized_datasets_temp['input_ids'][0]))
# # ds = load_dataset("cosmos_qa", split="train")
#
# new_column = ["foo"] * len(tokenized_datasets_temp)
# tokenized_datasets_temp = tokenized_datasets_temp.add_column("new_column", new_column)


# import pickle
#
# # Open the file in binary mode
# with open('features.pkl', 'rb') as file:
#     # Call load method to deserialze
#     features_temp = pickle.load(file)
#
# # Open the file in binary mode
# with open('batch.pkl', 'rb') as file:
#     # Call load method to deserialze
#     batch_temp = pickle.load(file)
#
# x = batch_temp['labels']
# x1 = batch_temp['input_ids']
# x2 = batch_temp['token_type_ids']
# x3 = batch_temp['attention_mask']
#
# for i in range(0,len(x2[1])):
#     if x3[2][i] == 0:
#         break
# # Precsion and recall:
# print("\n\nENSEMBLE MODEL:  ")
# print(classification_report(test_pred_flat, test_pred_label_flat,digits=4))
# # Confusion Matrix:
# y_true = test_pred_label_flat
# y_pred = test_pred_flat
# data = confusion_matrix(y_true, y_pred)
# df_cm = pd.DataFrame(data, columns=np.unique(y_true), index = np.unique(y_true))
# df_cm.index.name = 'Actual'
# df_cm.columns.name = 'Predicted'
# plt.figure(figsize = (10,7))
# sn.set(font_scale=1.4)#for label size
# sn.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16}, fmt='d')# font size
# plt.title("Confusion matrix for Ensemble model")
# plt.show()