#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Fine-tuning the library models for sequence classification."""


import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import datasets
from datasets import Dataset
import pandas as pd
import time
import glob
from pathlib import Path

import numpy as np
import tensorflow as tf

import random

from transformers import (
    AutoConfig,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizer,
    TFAutoModelForSequenceClassification,
    TFTrainer,
    TFTrainingArguments,
)

from transformers.utils import logging as hf_logging

from tf_wrapper import TFTrainerWrapper

def set_seeds(seed):
    if seed:
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        tf.random.set_seed(seed)
        np.random.seed(seed)

hf_logging.set_verbosity_info()
hf_logging.enable_default_handler()
hf_logging.enable_explicit_format()


'''
   Refer to https://github.com/google-research/bert/blob/master/run_classifier.py
   and https://github.com/huggingface/transformers/blob/master/examples/tensorflow/text-classification/run_text_classification.py
'''
class DatasetProcessor(object):
    
    """Base class for data converters for sequence classification data sets."""
    def __init__(self,
                 label_column_id,
                 text_column_id,
                 max_seq_length,
                 tokenizer,
                 to_add_cls = False,
                 to_add_sep = False,
                 positive_label = '',
                 use_balanced_neg = False,
                 no_neg_for_train_dev = False,
                 max_neg_scale = 2):
        
        self.label_column_id = label_column_id
        self.text_column_id  = text_column_id
        
        self.to_add_cls = to_add_cls
        self.to_add_sep = to_add_sep
        
        self.positive_label = positive_label
        
        self.use_balanced_neg = use_balanced_neg
        self.max_neg_scale = max_neg_scale
        self.no_neg_for_train_dev = no_neg_for_train_dev
        
        self.label_name = 'label'
        self.text_name  = 'text'
        
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
        self.input_names = self.tokenizer.model_input_names
        
        #print('>>>>>>>>>>>>>>>>self.tokenizer.model_input_names', self.tokenizer.model_input_names)
        
        self.transformed_ds = {}
    
    def _gen_train(self):
        label2id = self.get_label2id()
        
        for ex in self.transformed_ds['train']:
            d = {k: v for k, v in ex.items() if k in self.input_names}
            label = label2id[ex[self.label_name]]
            #print('>>>>>>>>>>>>>>>d.keys()', d.keys())
            #print('>>>>>>>>>>>>>>>label', label)
            yield (d, label)
        
    def _gen_eval(self):
        label2id = self.get_label2id()
        
        for ex in self.transformed_ds['dev']:
            d = {k: v for k, v in ex.items() if k in self.input_names}
            label = label2id[ex[self.label_name]]
            yield (d, label)
        
    def _gen_test(self):
        label2id = self.get_label2id()
        
        for ex in self.transformed_ds['test']:
            d = {k: v for k, v in ex.items() if k in self.input_names}
            if ex[self.label_name] != '':
                label = label2id[ex[self.label_name]]
            else:
                label = label2id['None']
            yield (d, label)
        
    def _get_dataset(self, 
                     data_file, 
                     set_type,
                     has_header = True):
        
        features = datasets.Features(
                {self.label_name: datasets.Value('string'),
                 self.text_name: datasets.Value('string')})
        
        data_dict = {}
        try:
            if has_header:
                data_df = pd.read_csv(data_file, sep='\t', dtype=str).fillna(np.str_(''))
            else:
                data_df = pd.read_csv(data_file, sep='\t', header=None, dtype=str).fillna(np.str_(''))
            data_dict[self.label_name]   = [self._map_label(label) for label in data_df.iloc[:,self.label_column_id]]
            data_dict[self.text_name]    = data_df.iloc[:,self.text_column_id]
        
        except:
            data_dict[self.label_name] = []
            data_dict[self.text_name] = []
        
        if set_type == 'train':
            if self.no_neg_for_train_dev:
                subset = []
                neg_labels = self.get_negative_labels()
                for i, label in enumerate(data_dict[self.label_name]):
                    if label not in neg_labels:
                        subset.append(i)
                        
                data_dict[self.label_name] = [data_dict[self.label_name][index] for index in subset]
                data_dict[self.text_name] = [data_dict[self.text_name][index] for index in subset]
            elif self.use_balanced_neg:
                num_neg = 0.
                for _neg_label in self.get_negative_labels():
                    num_neg += float(data_dict[self.label_name].count(_neg_label))
                num_non_neg = float(len(data_dict[self.label_name])) - num_neg
                neg_scale = int(round(num_neg / num_non_neg))
                neg_scale = 1 if neg_scale < 1 else neg_scale
                neg_scale = int(neg_scale)
                subset = []
                neg_labels = self.get_negative_labels()
                for i, label in enumerate(data_dict[self.label_name]):
                    if label in neg_labels:
                        _r = random.randint(1, neg_scale)
                        if _r <= self.max_neg_scale:
                            subset.append(i)
                    else:
                        subset.append(i)
        
                data_dict[self.label_name] = [data_dict[self.label_name][index] for index in subset]
                data_dict[self.text_name] = [data_dict[self.text_name][index] for index in subset]
        elif set_type == 'dev':
            if self.no_neg_for_train_dev:
                subset = []
                neg_labels = self.get_negative_labels()
                for i, label in enumerate(data_dict[self.label_name]):
                    if label not in neg_labels:
                        subset.append(i)
            
                data_dict[self.label_name] = [data_dict[self.label_name][index] for index in subset]
                data_dict[self.text_name] = [data_dict[self.text_name][index] for index in subset]
        
        if self.to_add_cls:
            text_list = data_dict[self.text_name]
            for i in range(len(text_list)):
                text_list[i] = '[CLS] ' + text_list[i]
        
        if self.to_add_sep:
            text_list = data_dict[self.text_name]
            for i in range(len(text_list)):
                text_list[i] = text_list[i] + ' [SEP]'
        
        data_dataset = Dataset.from_dict(data_dict, features=features)
        
        self.transformed_ds[set_type] = data_dataset.map(
            lambda example: self.tokenizer.batch_encode_plus(
                example[self.text_name], 
                truncation     = True, 
                max_length     = self.max_seq_length, 
                padding        = "max_length",
                stride         = 128
            ),
            batched=True,
        )
                
        if set_type == 'train':    
            data_ds = (
                tf.data.Dataset.from_generator(
                    self._gen_train,
                    ({k: tf.int32 for k in self.input_names}, tf.int64),
                    ({k: tf.TensorShape([None]) for k in self.input_names}, tf.TensorShape([])),
                )
            )
        elif set_type == 'dev':    
            data_ds = (
                tf.data.Dataset.from_generator(
                    self._gen_eval,
                    ({k: tf.int32 for k in self.input_names}, tf.int64),
                    ({k: tf.TensorShape([None]) for k in self.input_names}, tf.TensorShape([])),
                )
            )
        elif set_type == 'test':    
            data_ds = (
                tf.data.Dataset.from_generator(
                    self._gen_test,
                    ({k: tf.int32 for k in self.input_names}, tf.int64),
                    ({k: tf.TensorShape([None]) for k in self.input_names}, tf.TensorShape([])),
                )
            )

        data_ds = data_ds.apply(tf.data.experimental.assert_cardinality(len(data_dataset)))
        
        return data_ds

    def get_train_dataset(self, data_dir):
        return self._get_dataset(os.path.join(data_dir, "train.tsv"), "train", False)
    
    def get_dev_dataset(self, data_dir):
        return self._get_dataset(os.path.join(data_dir, "dev.tsv"), "dev", False)
    
    def get_test_dataset(self, data_dir):
        return self._get_dataset(os.path.join(data_dir, "test.tsv"), "test")    
    
    def get_train_dataset_by_name(self, file_name, has_header=False):
        return self._get_dataset(file_name, "train", has_header)
    
    def get_dev_dataset_by_name(self, file_name, has_header=False):
        return self._get_dataset(file_name, "dev", has_header)
    
    def get_test_dataset_by_name(self, file_name, has_header=False):
        return self._get_dataset(file_name, "test", has_header)
        
    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()
        
    def get_negative_labels(self):
        raise NotImplementedError()
        
    def get_label2id(self):
        label2id = {}
        for i, label in enumerate(self.get_labels()):
            mapped_id = self._map_label(label)
            if mapped_id not in label2id:
                label2id[mapped_id] = len(label2id)
        return label2id
        
    @classmethod
    def get_entity_type_dict(cls):
        raise NotImplementedError()
    
    def get_entity_type_list(self):
        return sorted([entity_type for entity_type in self.get_entity_type_dict().keys()])

    def get_entity_indices_by_types(self, text_a):
        entity_type_dict = self.get_entity_type_dict()
        all_indices = {}
        i_wo_empty_string = -1
        for i, token in enumerate(text_a.split(' ')):
            if token != '':
                i_wo_empty_string += 1
            if token in entity_type_dict:
                if token not in all_indices:
                    all_indices[token] = []
                #all_indices[token].append(i)
                all_indices[token].append(i_wo_empty_string)
        return all_indices
    
    def get_entity_types_in_text(self, text_a):
        entity_type_dict = self.get_entity_type_dict()
        entity_types_in_text = set()
        for i, token in enumerate(text_a.split(' ')):
            if token in entity_type_dict:
                entity_types_in_text.add(token)
        return entity_types_in_text
        
    def _map_label(self, label):
        
        # if positive_label is not None, means you are training a model for one vs the rest labels which will be negative label
        if self.positive_label != '':
            if self.positive_label == label:
                return label
            else:
                return self.get_negative_label()
        return label
    
    @classmethod
    def get_special_tags(cls):
        return None
   

class BioRExProcessor(DatasetProcessor):
    
    def __init__(self,
                 label_column_id = 9,
                 text_column_id  = 7,
                 max_seq_length  = 512,
                 tokenizer       = None,
                 to_add_cls      = False,
                 to_add_sep      = False,
                 positive_label  = '',
                 use_balanced_neg = False,
                 no_neg_for_train_dev = False,
                 max_neg_scale = 2):
        
        self.neg_labels = set(['None', 'None-CID', 'None-PPIm', 'None-AIMED', 'None-DDI', 'None-BC7', 'None-DISGENET', 'None-EMU_BC', 'None-EMU_PC', 'None-HPRD50', 'None-PHARMGKB'])
        
        super().__init__(
                label_column_id = label_column_id,
                text_column_id  = text_column_id,
                max_seq_length  = max_seq_length,
                tokenizer       = tokenizer,
                to_add_cls      = to_add_cls,
                to_add_sep      = to_add_sep,
                positive_label  = positive_label,
                use_balanced_neg= use_balanced_neg,
                no_neg_for_train_dev=no_neg_for_train_dev,
                max_neg_scale   = max_neg_scale)
    
    def get_labels(self):
        """See base class."""
        return ['None',
                'Association',
                'Bind',
                'Comparison',
                'Conversion',
                'Cotreatment',
                'Drug_Interaction',
                'Negative_Correlation',
                'Positive_Correlation',
                'None-CID',
                'CID',
                'None-PPIm',
                'PPIm',
                'None-AIMED',
                'None-DDI',
                'None-BC7',
                'None-phargkb',
                'None-GDA',
                'None-DISGENET', 
                'None-EMU_BC', 
                'None-EMU_PC', 
                'None-HPRD50', 
                'None-PHARMGKB',
                'ACTIVATOR',
                'AGONIST',
                'AGONIST-ACTIVATOR',
                'AGONIST-INHIBITOR',
                'ANTAGONIST',
                'DIRECT-REGULATOR',
                'INDIRECT-DOWNREGULATOR',
                'INDIRECT-UPREGULATOR',
                'INHIBITOR',
                'PART-OF',
                'PRODUCT-OF',
                'SUBSTRATE',
                'SUBSTRATE_PRODUCT-OF',
                'mechanism',
                'int',
                'effect',
                'advise',
                'AIMED-Association',
                'HPRD-Association',
                'EUADR-Association',
                'None-EUADR',
                'Indirect_conversion',
                'Non_conversion']
            
    @classmethod
    def get_entity_type_dict(cls):
        return {'@GeneOrGeneProductSrc$':0, 
                '@DiseaseOrPhenotypicFeatureSrc$':0,
                '@ChemicalEntitySrc$':0,    
                '@ChemicalSrc$':0,
                '@GeneSrc$':0,
                '@proteinSrc$':0,
                '@drugSrc$':0,
                '@CHEMICALSrc$': 0,
                '@SequenceVariantSrc$': 0,
                '@DiseaseSrc$': 0,
                '@/GeneOrGeneProductSrc$':0, 
                '@/DiseaseOrPhenotypicFeatureSrc$':0,
                '@/ChemicalEntitySrc$':0,
                '@/ChemicalSrc$':0,
                '@/GeneSrc$':0,
                '@/proteinSrc$':0,
                '@/drugSrc$':0,
                '@/CHEMICALSrc$': 0,
                '@/SequenceVariantSrc$': 0,
                '@/DiseaseSrc$': 0,
                '@GeneOrGeneProductTgt$':1,
                '@DiseaseOrPhenotypicFeatureTgt$':1,
                '@ChemicalEntityTgt$':1,
                '@DiseaseTgt$':1,
                '@GeneTgt$':1,
                '@proteinTgt$':1,
                '@drugTgt$':1,
                '@GENETgt$': 1,
                '@SequenceVariantTgt$': 1,
                '@/GeneOrGeneProductTgt$':1,
                '@/DiseaseOrPhenotypicFeatureTgt$':1,
                '@/ChemicalEntityTgt$':1,
                '@/DiseaseTgt$':1,
                '@/GeneTgt$':1,
                '@/proteinTgt$':1,
                '@/drugTgt$':1,
                '@/GENETgt$': 1,
                '@/SequenceVariantTgt$': 1,
                '@/DiseaseTgt$': 1,}
        
    def get_negative_labels(self):
        return self.neg_labels
    
    @classmethod
    def get_special_tags(cls):
        return ['[CID]', '[Litcoin]', '[BC6PM]', '[AIMED]', '[DDI]', '[Drugprot]', '[DISGENET]', '[BC]', '[PC]', '[PubtatorAPI]', '[EUADR]']

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    
    task_name: str = field(metadata={"help": "The name of the task"})
    
    in_data_dir: str = field(default=None, metadata={"help": "The path of the dataset files"})
    label_column_id: int = field(default=None, metadata={"help": "Which column contains the label"})
    text_column_id: int = field(default=None, metadata={"help": "Which column contains the text"})
    
    positive_label: Optional[str] = field(default="", metadata={"help": "If you specify a positive_label, the other positive labels will be assigned the negative label. dafault=''"})
    selected_label_for_evaluating_dev: Optional[str] = field(default=None, metadata={"help": "The labels are used for evaluating dev.tsv and save the best performance model. dafault=None"})
    
    use_balanced_neg: Optional[bool] = field(default=False, metadata={"help": "Whether to balance the numbers of negative and non-negative instances in train? dafault=False"})
    no_neg_for_train_dev: Optional[bool] = field(default=False, metadata={"help": "No to use negative instances in train and dev dafault=False"})
    max_neg_scale: Optional[int] = field(default=2, metadata={"help": "The times of negative instances over the other instances. It is used only if use_balanced_neg == True. dafault=2"})
    
    train_file: Optional[str] = field(default=None, metadata={"help": "The path of the train file"})
    dev_file: Optional[str] = field(default=None, metadata={"help": "The path of the dev file"})
    test_file: Optional[str] = field(default=None, metadata={"help": "The path of the test file"})
        
    in_test_data_dir: Optional[str] = field(default=None, metadata={"help": "Input test tsv dir"})
    in_data_dir: Optional[str] = field(default=None, metadata={"help": "Input pubtator or bioc xml dir"})
    out_pubtator_data_dir: Optional[str] = field(default=None, metadata={"help": "Output pred pubtator dir"})
    out_bioc_xml_data_dir: Optional[str] = field(default=None, metadata={"help": "Output pred biox xml dir"})
    out_tsv_data_dir: Optional[str] = field(default=None, metadata={"help": "Output pred tsv dir"})
    
    data_format: Optional[str] = field(default="pubtator", metadata={"help": "Input data format"})
    test_has_header: Optional[bool] = field(default=False, metadata={"help": "If test_file has header, default=False"})
    to_add_cls: Optional[bool] = field(default=False, metadata={"help": "Add [CLS] token to each instance, default=False"})
    to_add_sep: Optional[bool] = field(default=False, metadata={"help": "Append [SEP] token to each instance, default=False"})
    to_add_tag_as_special_token: Optional[bool] = field(default=False, metadata={"help": "Add @YOUR_TAG$ as special token, default=False"})
    
    do_predict_mul_files: Optional[bool] = field(default=False, metadata={"help": "Predict multiple .tsv, default=False"})
    
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast: bool = field(default=False, metadata={"help": "Set this flag to use fast tokenization."})
    # If you want to tweak more attributes on your tokenizer, you should do it in a distinct script,
    # or just modify its tokenizer_config.json.
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    
    hidden_dropout_prob: Optional[float] = field(
        default=None,
        metadata={"help": "If you specify hidden_dropout_prob, it won't use the hidden_dropout_prob of config.json"},
    )

def main():
    
    processors = {
        "biorex": BioRExProcessor,
    }
    
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TFTrainingArguments))
        
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    
    set_seeds(training_args.seed)
    
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(
        f"n_replicas: {training_args.n_replicas}, distributed training: {bool(training_args.n_replicas > 1)}, "
        f"16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    
    task_name = data_args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))


    if data_args.to_add_tag_as_special_token:
        new_special_tokens = list(processors[task_name].get_entity_type_dict().keys())
        new_special_tokens.sort()
        if not processors[task_name].get_special_tags():
            new_special_tokens.extend(processors[task_name].get_special_tags())
    else:
        new_special_tokens = []
    
    if training_args.do_train:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir = model_args.cache_dir,
            additional_special_tokens = new_special_tokens,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir = model_args.cache_dir,
        )

    #print('>>>>>>>>>>>>main tokenizer.model_input_names', tokenizer.model_input_names)
    processor = None
    if data_args.label_column_id != None and data_args.text_column_id != None:
        processor = processors[task_name](
                label_column_id = data_args.label_column_id,
                text_column_id  = data_args.text_column_id,
                max_seq_length  = data_args.max_seq_length,
                tokenizer       = tokenizer,
                to_add_cls      = data_args.to_add_cls,
                to_add_sep      = data_args.to_add_sep,
                positive_label  = data_args.positive_label,
                use_balanced_neg= data_args.use_balanced_neg,
                no_neg_for_train_dev = data_args.no_neg_for_train_dev,
                max_neg_scale   = data_args.max_neg_scale)
    else:
        processor = processors[task_name](
                max_seq_length  = data_args.max_seq_length,
                tokenizer       = tokenizer,
                to_add_cls      = data_args.to_add_cls,
                to_add_sep      = data_args.to_add_sep,
                positive_label  = data_args.positive_label,
                use_balanced_neg= data_args.use_balanced_neg,
                no_neg_for_train_dev = data_args.no_neg_for_train_dev,
                max_neg_scale   = data_args.max_neg_scale)
            
    label2id      = processor.get_label2id()
    id2label      = {id: label for label, id in label2id.items()}
    print('=======================>label2id', label2id)
    print('=======================>positive_label', data_args.positive_label)
    print('=======================>use_balanced_neg', data_args.use_balanced_neg)
    print('=======================>max_neg_scale', data_args.max_neg_scale)
    
    if data_args.selected_label_for_evaluating_dev != None and data_args.selected_label_for_evaluating_dev != '':
        selected_label_ids_for_evaluating_dev = np.array([label2id[label] for label in data_args.selected_label_for_evaluating_dev.split('|')])
    else:
        selected_label_ids_for_evaluating_dev = np.array([])

    # if has multiple neg labels, we have to use compute_metrics_with_labels(), so we assign selected_label_ids_for_evaluating_dev
    if len(processor.get_negative_labels()) > 1:
        pos_label_ids = []
        for id, label in id2label.items():
            if label not in processor.get_negative_labels():
                pos_label_ids.append(id)
        selected_label_ids_for_evaluating_dev = np.array(pos_label_ids)
        logger.info(f"pos_label_ids")
        logger.info(pos_label_ids)
    else:
        for neg_label in processor.get_negative_labels():
            neg_label_id  = label2id[neg_label]
            break
    
    with training_args.strategy.scope():
        
        config = None
        
        if training_args.do_train:
            config = AutoConfig.from_pretrained(
                model_args.config_name if model_args.config_name else model_args.model_name_or_path,
                num_labels      = len(label2id),
                label2id        = label2id,
                id2label        = id2label,
                finetuning_task = "text-classification",
                cache_dir       = model_args.cache_dir,
            )
            if model_args.hidden_dropout_prob:
                config.hidden_dropout_prob = model_args.hidden_dropout_prob
        
        if config:
            model = TFAutoModelForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                from_pt    = True if any(fname.endswith('.bin') for fname in os.listdir(model_args.model_name_or_path)) else False,
                config     = config,
                cache_dir  = model_args.cache_dir,
                ignore_mismatched_sizes=True
                )
        else:
            model = TFAutoModelForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                from_pt    = True if any(fname.endswith('.bin') for fname in os.listdir(model_args.model_name_or_path)) else False,
                cache_dir  = model_args.cache_dir)
        #    ignore_mismatched_sizes=True
        #)
        
        #if training_args.do_train:
        #    model.resize_token_embeddings(len(tokenizer))
        
        model.resize_token_embeddings(len(tokenizer))
    
    def compute_metrics(p: EvalPrediction) -> Dict:
        preds = np.argmax(p.predictions, axis=1)
        
        np_array_non_neg_label_id = p.label_ids != neg_label_id
        np_array_compared_result  = p.label_ids == preds
        np_array_tp               = np_array_compared_result * np_array_non_neg_label_id
        np_array_tp               = p.label_ids * np_array_tp
        np_array_tp_wo_neg = np.delete(np_array_tp, np.where(np_array_tp == neg_label_id))
        np_array_pred_pos  = np.delete(preds,       np.where(preds       == neg_label_id))
        np_array_gold_pos  = np.delete(p.label_ids, np.where(p.label_ids == neg_label_id))
        
        np_f_tp       = np.float(np_array_tp_wo_neg.shape[0])
        np_f_pred_pos = np.float(np_array_pred_pos.shape[0])
        np_f_gold_pos = np.float(np_array_gold_pos.shape[0])
        
        precision = np_f_tp / np_f_pred_pos if np_f_pred_pos != 0. else 0.
        recall    = np_f_tp / np_f_gold_pos if np_f_gold_pos != 0. else 0.
        f1        = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0. else 0.
        
        logger.info(f"tp_debug")
        logger.info(np_array_tp)
        logger.info(f"pred_debug")
        logger.info(preds)
        logger.info(f"gold_debug")
        logger.info(p.label_ids)
        logger.info(f"neg_label_id")
        logger.info(neg_label_id)
        
        return {"f1":          f1,
                'precision':   precision,
                'recall':      recall,
                'tp':          np_f_tp,
                'fp':          np_f_pred_pos - np_f_tp,
                'fn':          np_f_gold_pos - np_f_tp}

    def compute_metrics_with_labels(p: EvalPrediction) -> Dict:
        preds = np.argmax(p.predictions, axis=1)
        
        # non-selected labels are considered as don't care (negative label)
        np_array_non_neg_label_id = np.isin(p.label_ids, selected_label_ids_for_evaluating_dev)
        np_array_compared_result  = p.label_ids == preds
        np_array_tp               = np_array_compared_result * np_array_non_neg_label_id
        np_array_tp               = p.label_ids * np_array_tp
        
        
        np_array_tp_wo_neg = np.delete(np_array_tp, np.where(np.invert(np.isin(np_array_tp, selected_label_ids_for_evaluating_dev))))
        np_array_pred_pos  = np.delete(preds,       np.where(np.invert(np.isin(preds, selected_label_ids_for_evaluating_dev))))
        np_array_gold_pos  = np.delete(p.label_ids, np.where(np.invert(np.isin(p.label_ids, selected_label_ids_for_evaluating_dev))))
        
        np_f_tp       = np.float(np_array_tp_wo_neg.shape[0])
        np_f_pred_pos = np.float(np_array_pred_pos.shape[0])
        np_f_gold_pos = np.float(np_array_gold_pos.shape[0])
        
        precision = np_f_tp / np_f_pred_pos if np_f_pred_pos != 0. else 0.
        recall    = np_f_tp / np_f_gold_pos if np_f_gold_pos != 0. else 0.
        f1        = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0. else 0.
        
        logger.info(f"tp_debug")
        logger.info(np_array_tp)
        logger.info(f"pred_debug")
        logger.info(preds)
        logger.info(f"gold_debug")
        logger.info(p.label_ids)
        
        return {"f1":          f1,
                'precision':   precision,
                'recall':      recall,
                'tp':          np_f_tp,
                'fp':          np_f_pred_pos - np_f_tp,
                'fn':          np_f_gold_pos - np_f_tp}

    # Initialize our Trainer
    

    # Training and evaluating
    results = {}
    #learned_model = model    
    
    if training_args.do_train:
        
        if not data_args.train_file:
            train_dataset = processor.get_train_dataset(data_args.in_data_dir)
        else:
            train_dataset = processor.get_train_dataset_by_name(data_args.train_file)
        
        if not data_args.dev_file:
            eval_dataset  = processor.get_dev_dataset(data_args.in_data_dir)
        else:
            eval_dataset = processor.get_dev_dataset_by_name(data_args.dev_file)
        
        if len(processor.get_negative_labels()) > 1:
            # if has multiple neg labels, we have to use compute_metrics_with_labels()
            learner = TFTrainerWrapper(
                model            = model,
                args             = training_args,
                train_dataset    = train_dataset,
                eval_dataset     = eval_dataset,
                compute_metrics  = compute_metrics_with_labels,
                main_metric_name = 'f1'
            )
        elif data_args.selected_label_for_evaluating_dev == None or data_args.selected_label_for_evaluating_dev == '':
            learner = TFTrainerWrapper(
                model            = model,
                args             = training_args,
                train_dataset    = train_dataset,
                eval_dataset     = eval_dataset,
                compute_metrics  = compute_metrics,
                main_metric_name = 'f1'
            )
        else:
            learner = TFTrainerWrapper(
                model            = model,
                args             = training_args,
                train_dataset    = train_dataset,
                eval_dataset     = eval_dataset,
                compute_metrics  = compute_metrics_with_labels,
                main_metric_name = 'f1'
            )
        learner.train(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)   
        
        model = TFAutoModelForSequenceClassification.from_pretrained(
            training_args.output_dir,
            from_pt    = True if any(fname.endswith('.bin') for fname in os.listdir(training_args.output_dir)) else False,
            config     = config,
            cache_dir  = model_args.cache_dir)
        #
        #    ignore_mismatched_sizes=True
        #)
        # loading the best model
        '''learner = TFTrainerWrapper(
            model            = model,
            args             = training_args,
            train_dataset    = train_dataset,
            eval_dataset     = eval_dataset,
            compute_metrics  = compute_metrics,
            main_metric_name = 'f1'
        )'''

        if not os.path.exists(training_args.output_dir):
            os.makedirs(training_args.output_dir)
        
        batch_eval_dataset = eval_dataset.batch(training_args.eval_batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        
        predictions = model.predict(batch_eval_dataset)["logits"]
        #predictions = model(eval_dataset)
        #predictions = np.argmax(predictions, axis=1)
        output_predict_file = os.path.join(training_args.output_dir, "eval_results.tsv")
        with open(output_predict_file, "w") as writer:
            for index, item in enumerate(predictions):
                writer.write('\t'.join(map(str, item)) + '\n')
    
    if training_args.do_predict:
        if not os.path.exists(training_args.output_dir):
            os.makedirs(training_args.output_dir)
        '''
        if not learner:
            learner = TFTrainerWrapper(
                model            = model,
                args             = training_args,
                train_dataset    = test_dataset,
                eval_dataset     = test_dataset,
                compute_metrics  = compute_metrics,
                main_metric_name = 'f1'
            )'''
        
        if not data_args.test_file:
            test_dataset  = processor.get_test_dataset(data_args.in_data_dir)
        else:
            test_dataset  = processor.get_test_dataset_by_name(data_args.test_file, data_args.test_has_header)
        
        batch_test_dataset = test_dataset.batch(training_args.eval_batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        predictions = model.predict(batch_test_dataset)["logits"]
        #predictions = model(test_dataset)
        #predictions = np.argmax(predictions, axis=1)
        output_predict_file = os.path.join(training_args.output_dir, "test_results.tsv")
        with open(output_predict_file, "w") as writer:
            for index, item in enumerate(predictions):
                writer.write('\t'.join(map(str, item)) + '\n')
                #writer.write(str(id2label[item]) + '\n')
                
    return results


if __name__ == "__main__":
    main()
