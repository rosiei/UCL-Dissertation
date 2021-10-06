#https://mccormickml.com/2019/09/19/XLNet-fine-tuning/
from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import pickle
import time
import math

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
# from keras.preprocessing.sequence import pad_sequences
from sklearn import metrics

# from keras.preprocessing.sequence import pad_sequences


from transformers import XLNetModel, XLNetTokenizer, XLNetForSequenceClassification,XLNetConfig
from transformers import AdamW

import torch.autograd as autograd
from scipy import stats



class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, note=""):
        """Constructs a InputExample.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.note = note

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, guid):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.guid = guid
        

class MAProcessor(object):
    
    def get_direct_control_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "ma_train.pkl")), "missed_micro") + self._create_examples(self._read_tsv(os.path.join(data_dir, "nonma_large_train.pkl")), "clean") + self._create_examples(self._read_tsv(os.path.join(data_dir, "toxic_train.pkl")), "hateful")
    
    def get_dirctr_corrected_train_examples(self, data_dir, correction_dir, correction_size):
        """See base class."""
        sorted_ex_for_correction = self._read_tsv(os.path.join(correction_dir, "sorted_ex_for_correction.pkl"))
        correct_idx_list = sorted_ex_for_correction[:correction_size]
        examples = self._create_examples(self._read_tsv(os.path.join(data_dir, "ma_train.pkl")), "missed_micro") + self._create_examples(self._read_tsv(os.path.join(data_dir, "nonma_large_train.pkl")), "clean") + self._create_examples(self._read_tsv(os.path.join(data_dir, "toxic_train.pkl")), "hateful")
        new_examples = []
        for i, example in enumerate(examples):
            if i in correct_idx_list:
                new_examples.append(InputExample(guid=i, text_a=example.text_a, text_b=None, label="1")) # flip the label
            else:
                new_examples.append(InputExample(guid=i, text_a=example.text_a, text_b=None, label=example.label))
        return new_examples
    
    def get_dirctr_checked_train_examples(self, data_dir, correction_dir, correction_size):
        """See base class."""
        sorted_ex_for_correction = self._read_tsv(os.path.join(correction_dir, "sorted_ex_for_correction.pkl"))
        correct_idx_list = sorted_ex_for_correction[:correction_size]
        len_ma_train = len(self._read_tsv(os.path.join(data_dir, "ma_train.pkl")))
        examples = self._create_examples(self._read_tsv(os.path.join(data_dir, "ma_train.pkl")), "missed_micro") + self._create_examples(self._read_tsv(os.path.join(data_dir, "nonma_large_train.pkl")), "clean") + self._create_examples(self._read_tsv(os.path.join(data_dir, "toxic_train.pkl")), "hateful")
        new_examples = []
        for i, example in enumerate(examples):
            if i in correct_idx_list and i < len_ma_train: # only flip the true microaggressions
                new_examples.append(InputExample(guid=i, text_a=example.text_a, text_b=None, label="1")) # flip the label
            else:
                new_examples.append(InputExample(guid=i, text_a=example.text_a, text_b=None, label=example.label))
        return new_examples
    
    def get_dirctr_gold_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "ma_train.pkl")), "gold_micro") + self._create_examples(self._read_tsv(os.path.join(data_dir, "nonma_large_train.pkl")), "clean") + self._create_examples(self._read_tsv(os.path.join(data_dir, "toxic_train.pkl")), "hateful")
    
    def get_direct_control_test_examples(self, data_dir):
        """See base class."""
        ma_test_ex = self._create_examples(self._read_tsv(os.path.join(data_dir, "ma_test.pkl")), "gold_micro")
        clean_test_ex = self._create_examples(self._read_tsv(os.path.join(data_dir, "nonma_large_test.pkl")), "clean")
        toxic_test_ex = self._create_examples(self._read_tsv(os.path.join(data_dir, "toxic_test.pkl")), "hateful")
        return (ma_test_ex, clean_test_ex, toxic_test_ex)
    
    def get_adv_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "ma_adv.pkl")), "missed_micro")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = i
            text_a = line[0]
            if set_type == "hateful":
                label = "1"
            elif set_type == "missed_micro":
                label = "0"
            elif set_type == "gold_micro":
                label = "1"
            elif set_type == "clean":
                label = "0"
            else:
                raise ValueError("Check your set type")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples
    
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "rb") as f:
            pairs = pickle.load(f)
            return pairs

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    SEG_ID_A   = 0
    SEG_ID_B   = 1
    SEG_ID_CLS = 2
    SEG_ID_SEP = 3
    SEG_ID_PAD = 4
    
    UNK_ID = tokenizer.encode("<unk>")[0]
    CLS_ID = tokenizer.encode("<cls>")[0]
    SEP_ID = tokenizer.encode("<sep>")[0]
    MASK_ID = tokenizer.encode("<mask>")[0]
    EOD_ID = tokenizer.encode("<eod>")[0]
    for (ex_index, example) in enumerate(examples):
        # Tokenize sentence to token id list
        tokens_a = tokenizer.encode(example.text_a)
        
        # Trim the len of text
        if(len(tokens_a)>max_seq_length-2):
            tokens_a = tokens_a[:max_seq_length -2]
        # tokens_b = None
        # if example.text_b:
        #     tokens_b = tokenizer.tokenize(example.text_b)    
            
        tokens = []
        segment_ids = []
        
        # try
        # tokens = tokens_a + ["[SEP]"] + ["[CLS]"]
        # segment_ids = [0] * len(tokens)
        
        # if tokens_b:
        #     tokens += tokens_b + ["[SEP]"]
        #     segment_ids += [1] * (len(tokens_b) + 1)

        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(SEG_ID_A) 
        
            
        # Add <sep> token 
        tokens.append(SEP_ID)
        segment_ids.append(SEG_ID_A)
        
        
        # Add <cls> token
        tokens.append(CLS_ID)
        segment_ids.append(SEG_ID_CLS)
        
        input_ids = tokens
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
    
        # Zero-pad up to the sequence length at fornt
        if len(input_ids) < max_seq_length:
            delta_len = max_seq_length - len(input_ids)
            input_ids = [1] * delta_len + input_ids
            input_mask = [0] * delta_len + input_mask
            segment_ids = [SEG_ID_PAD] * delta_len + segment_ids

  
    
    # full_input_ids.append(input_ids)
    # full_input_masks.append(input_mask)
    # full_segment_ids.append(segment_ids)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
    # https://medium.com/swlh/using-xlnet-for-sentiment-classification-cfa948e65e85
        # encodings = tokenizer.encode_plus(example.text_a, add_special_tokens=True, max_length=max_seq_length, return_tensors='pt', return_token_type_ids=False, return_attention_mask=True, pad_to_max_length=True)
        # input_ids = pad_sequences(encodings['input_ids'], maxlen=max_seq_length, dtype="long", truncating="post", padding="post")
        # # input_ids = input_ids.astype(dtype = 'int64')
        # # input_ids = torch.tensor(input_ids) 
        # attention_mask = pad_sequences(encodings['attention_mask'], maxlen=max_seq_length, dtype=torch.Tensor ,truncating="post",padding="post")
        # # attention_mask = attention_mask.astype(dtype = 'int64')
        # # attention_mask = torch.tensor(attention_mask)    
    
        label_id = label_map[example.label]
    
        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              guid=example.guid))
    # features = torch.tensor(features)    
    print (len(features))
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def accuracy(out, label_ids):
    # axis-0: seqs in batch; axis-1: potential labels of seq
    outputs = np.argmax(out, axis=1)
    matched = outputs == label_ids
    num_correct = np.sum(matched)
    num_total = len(label_ids)
    return num_correct, num_total
    
#https://www.kaggle.com/tanulsingh077/deep-learning-for-nlp-zero-to-transformers-bert    
def roc_auc(predictions,target):
    '''
    This methods returns the AUC Score when given the Predictions
    and Labels
    '''
    
    fpr, tpr, thresholds = metrics.roc_curve(target, predictions)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc
def save_Conf_mat():    
    for matrix in confusion_matrices:
        fig = plt.figure()
        plt.matshow(cm)
        plt.title('Problem 1: Confusion Matrix Digit Recognition')
        plt.colorbar()
        plt.ylabel('True Label')
        plt.xlabel('Predicated Label')
        plt.savefig('confusion_matrix'+str(learning_values.pop())+'.jpg')
