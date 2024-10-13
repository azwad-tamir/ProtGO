
import copy
import json

import numpy
import pandas as pd
import re
import numpy as np
from setuptools.dist import sequence
import pickle
import os
import gzip
import base64
import io
import itertools
import subprocess
import shlex
import tqdm

with open(r"./data/random_train_top100.pkl", "rb") as input_file:
    all_data = pickle.load(input_file)

trian_id = all_data[0]
train_seq = all_data[3]
train_go = all_data[6]
train_go_orig = all_data[9]

# Loading true labels
val_df = pd.read_csv("./random_test_df")
val_id = list(val_df['ID'])
val_seq = list(val_df['sequence'])
val_go = list(val_df['GO_terms'])