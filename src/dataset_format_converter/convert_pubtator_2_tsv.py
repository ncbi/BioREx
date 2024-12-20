# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 13:27:17 2021

@author: laip2
"""

from document import PubtatorDocument, TextInstance
from annotation import AnnotationInfo
import os
import random
import glob
from pathlib import Path

from collections import defaultdict

import re
import numpy as np

import sys
import utils
      
import optparse

import json
import drugprot_loader

import codecs

parser = optparse.OptionParser()
 
parser.add_option('--exp_option',                       action="store",
                     dest="exp_option",                 help="dataset name", default="")

parser.add_option('--in_main_train_files',              action="store",
                     dest="in_main_train_files",        help="input tsv files seperated by \';\'", default="")

parser.add_option('--in_main_test_files',               action="store",
                     dest="in_main_test_files",         help="input tsv files seperated by \';\'", default="")

parser.add_option('--in_other_train_files',             action="store",
                     dest="in_other_train_files",       help="input tsv files seperated by \';\'", default="")

parser.add_option('--in_other_test_files',              action="store",
                     dest="in_other_test_files",        help="input tsv files seperated by \';\'", default="")

parser.add_option('--out_train_tsv_file',               action="store",
                     dest="out_train_tsv_file",         help="output combined tsv file", default="")

parser.add_option('--out_test_tsv_file',                action="store",
                     dest="out_test_tsv_file",          help="output combined tsv file", default="")

parser.add_option('--out_data_dir',                     action="store",
                     dest="out_data_dir",               help="output processed dir", default="")

parser.add_option('--normalize_pair_2_rel_type',        action="store",
                     dest="normalize_pair_2_rel_type",  help="normalize pair to type eg:'proteinSrc|proteinTgt|Relation|Association;drugSrc|drugTgt|effect|Association'", default="")

parser.add_option('--normalize_pair_2_pair',            action="store",
                     dest="normalize_pair_2_pair",      help="normalize pair to new pair eg:'chemSrc|proteinTgt|ChemicalSrc|GeneTgt'", default="")

parser.add_option('--normalize_tag_2_tag',              action="store",
                     dest="normalize_tag_2_tag",        help="normalize tag to new tag eg:'[AIMED]|[LitCoin]'", default="")

parser.add_option('--num_train_biored',                 action="store", type="int",
                     dest="num_train_biored",           help="the maximum number of biored doc used for training (default is -1:all)", default=-1)

parser.add_option('--in_test_pubtator_file',            action="store", 
                     dest="in_test_pubtator_file",      help="use pubtator NER/NEL results", default="")

parser.add_option('--only_co_occurrence_sent',          action="store_true",
                     dest="only_co_occurrence_sent",    help="to use only co-occurrence sentences", default=False)

parser.add_option('--has_novelty',                      action="store_true",
                     dest="has_novelty",                help="to append novelity to last column or not", default=False)

parser.add_option('--to_mask_src_and_tgt',              action="store_true",
                     dest="to_mask_src_and_tgt",        help="to mask src and tgo or not", default=False)


parser.add_option('--in_pubtator_file',                 action="store",
                     dest="in_pubtator_file",           help="input pubtator file for 'pubtator_2_tsv'", default="")

parser.add_option('--in_pubtator_dir',                  action="store",
                     dest="in_pubtator_dir",            help="input pubtator dir for 'pubtator_2_tsv'", default="")

parser.add_option('--out_tsv_file',                     action="store",
                     dest="out_tsv_file",               help="output tsv file for 'pubtator_2_tsv'", default="")

parser.add_option('--out_tsv_dir',                      action="store",
                     dest="out_tsv_dir",                help="output tsv dir for 'pubtator_2_tsv'", default="")

parser.add_option('--to_sentence_level',                action="store_true",
                     dest="to_sentence_level",          help="convert to sentence level dataset", default=False)

parser.add_option('--to_remove_question',               action="store_true",
                     dest="to_remove_question",         help="remove question text between [CLS] and [SEP]", default=False)

parser.add_option('--to_merge_neg_2_none',              action="store_true",
                     dest="to_merge_neg_2_none",        help="merge all negative instances into 'None' label", default=False)

def add_annotations_2_text_instances(text_instances, annotations):
    offset = 0
    for text_instance in text_instances:
        text_instance.offset = offset
        offset += len(text_instance.text) + 1
        
    for annotation in annotations:
        can_be_mapped_to_text_instance = False
                
        for i, text_instance in enumerate(text_instances):
            if text_instance.offset <= annotation.position and annotation.position + annotation.length <= text_instance.offset + len(text_instance.text):
                
                annotation.position = annotation.position - text_instance.offset
                text_instance.annotations.append(annotation)
                can_be_mapped_to_text_instance = True
                break
        if not can_be_mapped_to_text_instance:
            print(annotation.text)
            print(annotation.position)
            print(annotation.length)
            print(annotation, 'cannot be mapped to original text')
            raise
    
def load_pubtator_into_documents(in_pubtator_file, 
                                 normalized_type_dict = {},
                                 re_id_spliter_str = r'\,',
                                 pmid_2_index_2_groupID_dict = None):
    
    documents = []
    
    with open(in_pubtator_file, 'r', encoding='utf8') as pub_reader:
        
        pmid = ''
        
        document = None
        
        annotations = []
        text_instances = []
        relation_pairs = {}
        index2normalized_id = {}
        id2index = {}
        
        for line in pub_reader:
            line = line.rstrip()
            
            if line == '':
                
                document = PubtatorDocument(pmid)
                #print(pmid)
                add_annotations_2_text_instances(text_instances, annotations)
                document.text_instances = text_instances
                document.relation_pairs = relation_pairs
                documents.append(document)
                
                annotations = []
                text_instances = []
                relation_pairs = {}
                id2index = {}
                index2normalized_id = {}
                continue
            
            tks = line.split('|')
            
            
            if len(tks) > 1 and (tks[1] == 't' or tks[1] == 'a'):
                #2234245	250	270	audiovisual toxicity	Disease	D014786|D006311
                pmid = tks[0]
                x = TextInstance(tks[2])
                text_instances.append(x)
            else:
                _tks = line.split('\t')
                if len(_tks) == 6:
                    start = int(_tks[1])
                    end = int(_tks[2])
                    index = _tks[1] + '|' + _tks[2]
                    text = _tks[3]
                    ne_type = _tks[4]
                    ne_type = re.sub(r'\s*\(.*?\)\s*$', '', ne_type)
                    orig_ne_type = ne_type
                    if ne_type in normalized_type_dict:
                        ne_type = normalized_type_dict[ne_type]
                    
                    _anno = AnnotationInfo(start, end-start, text, ne_type)
                    
                    #2234245	250	270	audiovisual toxicity	Disease	D014786|D006311
                    ids = [x for x in re.split(re_id_spliter_str, _tks[5])]
                    
                    # if annotation has groupID then update its id
                    if orig_ne_type == 'SequenceVariant':
                        if pmid_2_index_2_groupID_dict != None and index in pmid_2_index_2_groupID_dict[pmid]:
                            index2normalized_id[index] = pmid_2_index_2_groupID_dict[pmid][index][0] # pmid_2_tmvarID_2_groupID_dict[pmid][_id] => (var_id, gene_id)
                            _anno.corresponding_gene_id = pmid_2_index_2_groupID_dict[pmid][index][1]
                    for i, _id in enumerate(ids):
                        if pmid_2_index_2_groupID_dict != None and index in pmid_2_index_2_groupID_dict[pmid]:
                            id2index[ids[i]] = index
                            ids[i] = pmid_2_index_2_groupID_dict[pmid][index][0] # pmid_2_tmvarID_2_groupID_dict[pmid][_id] => (var_id, gene_id)
                            _anno.corresponding_gene_id = pmid_2_index_2_groupID_dict[pmid][index][1]
                        else:
                            #ids[i] = re.sub('\s*\(.*?\)\s*$', '', _id)
                            ids[i] = _id
                        
                    
                    _anno.orig_ne_type = orig_ne_type
                    _anno.ids = set(ids)
                    annotations.append(_anno)
                elif len(_tks) == 4 or len(_tks) == 5:
                    
                    id1 = _tks[2]
                    id2 = _tks[3]
                    
                    if pmid_2_index_2_groupID_dict != None and (id1 in id2index) and (id2index[id1] in index2normalized_id):
                        id1 = index2normalized_id[id2index[id1]] # pmid_2_tmvarID_2_groupID_dict[pmid][_id] => (var_id, gene_id)
                    if pmid_2_index_2_groupID_dict != None and (id2 in id2index) and (id2index[id2] in index2normalized_id):
                        id2 = index2normalized_id[id2index[id2]] # pmid_2_tmvarID_2_groupID_dict[pmid][_id] => (var_id, gene_id)
                    rel_type = _tks[1]
                    relation_pairs[(id1, id2)] = rel_type
                    
        if len(text_instances) != 0:
            document = PubtatorDocument(pmid)
            add_annotations_2_text_instances(text_instances, annotations)
            document.text_instances = text_instances
            document.relation_pairs = relation_pairs
            documents.append(document)
    
    return documents
    
   
def get_out_neighbors_list(text_instance):
    
    out_neighbors_list = []
    out_neighbors_head_list = []
    
    invert_heads = {}
        
    
    for current_idx, (head, head_idx) in enumerate(zip(text_instance.head,
                                                       text_instance.head_indexes)):
        if head_idx not in invert_heads:
            invert_heads[head_idx] = []
        
        _edge = {}
        _edge['label'] = head
        _edge['toIndex'] = current_idx
        
        invert_heads[head_idx].append(_edge)
    
    for current_idx, (head, head_idx) in enumerate(zip(
                                         text_instance.head,
                                         text_instance.head_indexes)):
        neighbors = []
        neighbors_head = []
                    
        if current_idx in invert_heads:
            for edge in invert_heads[current_idx]:
                neighbors.append(edge['toIndex'])
                neighbors_head.append(edge['label'])
        
        out_neighbors_list.append(neighbors)
        out_neighbors_head_list.append(neighbors_head)
        
    return out_neighbors_list, out_neighbors_head_list



def convert_pubtator_to_tsv_file(
        in_pubtator_file,
        out_tsv_file,
        has_end_tag,
        task_tag,        
        spacy_model,
        normalized_type_dict,
        src_ne_type = '',
        tgt_ne_type = '',
        src_tgt_pairs = set(),
        gene_tag = 'Gene',
        re_id_spliter_str = r'\,',
        neg_label = 'None',
        pos_label = '',
        selected_doc_ids = None,
        only_co_occurrence_sent = False,
        to_mask_src_and_tgt = False,
        to_sentence_level = False):
            
    all_documents = load_pubtator_into_documents(
            in_pubtator_file     = in_pubtator_file, 
            normalized_type_dict = normalized_type_dict,
            re_id_spliter_str    = re_id_spliter_str)
    
    if selected_doc_ids != None:
        selected_docs = []
        for doc in all_documents:
            if doc.id in selected_doc_ids:
                selected_docs.append(doc)
        all_documents = selected_docs
    
    if not to_sentence_level:
        #utils.dump_documents_2_bert_gt_format(
        utils.dump_documents_2_bert_format(
            all_documents, 
            out_tsv_file, 
            src_ne_type,
            tgt_ne_type,
            src_tgt_pairs,
            has_end_tag = has_end_tag,
            only_co_occurrence_sent = only_co_occurrence_sent,
            task_tag    = task_tag,
            neg_label   = neg_label,
            has_ne_type = True,
            to_mask_src_and_tgt = to_mask_src_and_tgt,
            pos_label   = pos_label)
    else:
        utils.tokenize_documents_by_spacy(all_documents, spacy_model)
        utils.dump_documents_2_bert_gt_format_by_sent_level(
            all_documents          = all_documents,
            out_bert_file          = out_tsv_file, 
            src_ne_type            = src_ne_type,
            tgt_ne_type            = tgt_ne_type,
            src_tgt_pairs          = src_tgt_pairs,
            is_test_set            = False,
            to_mask_src_and_tgt    = to_mask_src_and_tgt,
            to_insert_src_and_tgt_at_left = True,
            has_end_tag            = has_end_tag,
            task_tag               = task_tag,
            neg_label              = neg_label,
            add_ne_type            = True,
            pos_label   = pos_label)
        
def convert_pubtator_to_tsv_file_with_mask_NEs(
        in_pubtator_file,
        out_tsv_file,
        has_end_tag,
        task_tag,        
        spacy_model,
        normalized_type_dict,
        src_ne_type = '',
        tgt_ne_type = '',
        src_tgt_pairs = set(),
        gene_tag = 'Gene',
        re_id_spliter_str = r'\,',
        neg_label = 'None',
        pos_label = '',
        selected_doc_ids = None,
        only_co_occurrence_sent = False,
        to_mask_src_and_tgt = True,
        to_sentence_level = False):
            
    all_documents = load_pubtator_into_documents(
            in_pubtator_file     = in_pubtator_file, 
            normalized_type_dict = normalized_type_dict,
            re_id_spliter_str    = re_id_spliter_str)
    
    if selected_doc_ids != None:
        selected_docs = []
        for doc in all_documents:
            if doc.id in selected_doc_ids:
                selected_docs.append(doc)
        all_documents = selected_docs
    
    utils.tokenize_documents_by_spacy(all_documents, spacy_model)
    
    # below function allow pair in the same token
    utils.dump_documents_2_bert_format(
        all_documents, 
        out_tsv_file, 
        src_ne_type,
        tgt_ne_type,
        src_tgt_pairs,
        has_end_tag = has_end_tag,
        only_co_occurrence_sent = only_co_occurrence_sent,
        task_tag    = task_tag,
        neg_label   = neg_label,
        has_ne_type = True,
        to_mask_src_and_tgt = to_mask_src_and_tgt,
        pos_label   = pos_label)

def gen_cdr_dataset(
        in_cdr_dir,
        out_cdr_dir,
        spacy_model,
        has_end_tag,
        re_id_spliter_str,
        normalized_type_dict,
        task_tag,
        neg_label):
    
    in_train_pubtator_file = in_cdr_dir + 'CDR_TrainingSet.PubTator.txt'
    in_dev_pubtator_file   = in_cdr_dir + 'CDR_DevelopmentSet.PubTator.txt'
    in_test_pubtator_file  = in_cdr_dir + 'CDR_TestSet.PubTator.txt'    
    
    if not os.path.exists(out_cdr_dir):
        os.makedirs(out_cdr_dir)    
    
    out_train_tsv_file = out_cdr_dir + 'train.tsv'
    out_dev_tsv_file   = out_cdr_dir + 'dev.tsv'
    out_test_tsv_file  = out_cdr_dir + 'test.tsv'
    
    src_ne_type = 'Chemical'
    tgt_ne_type = 'Disease'
    
    convert_pubtator_to_tsv_file(
        in_pubtator_file = in_train_pubtator_file,
        out_tsv_file     = out_train_tsv_file,
        src_ne_type      = src_ne_type,
        tgt_ne_type      = tgt_ne_type,
        has_end_tag      = has_end_tag,
        task_tag         = task_tag,
        re_id_spliter_str= re_id_spliter_str,
        normalized_type_dict = normalized_type_dict,
        spacy_model      = spacy_model,
        neg_label        = neg_label)    
    
    convert_pubtator_to_tsv_file(
        in_pubtator_file = in_dev_pubtator_file,
        out_tsv_file     = out_dev_tsv_file,
        src_ne_type      = src_ne_type,
        tgt_ne_type      = tgt_ne_type,
        has_end_tag      = has_end_tag,
        task_tag         = task_tag,
        re_id_spliter_str= re_id_spliter_str,
        normalized_type_dict = normalized_type_dict,
        spacy_model      = spacy_model,
        neg_label        = neg_label)    
    
    convert_pubtator_to_tsv_file(
        in_pubtator_file = in_test_pubtator_file,
        out_tsv_file     = out_test_tsv_file,
        src_ne_type      = src_ne_type,
        tgt_ne_type      = tgt_ne_type,
        has_end_tag      = has_end_tag,
        task_tag         = task_tag,
        re_id_spliter_str= re_id_spliter_str,
        normalized_type_dict = normalized_type_dict,
        spacy_model      = spacy_model,
        neg_label        = neg_label)
    
def gen_aimed_dataset(
        in_data_dir,
        out_data_dir,
        spacy_model,
        has_end_tag,
        re_id_spliter_str,
        normalized_type_dict,
        task_tag,
        neg_label):
    
    in_pubtator_file = in_data_dir + 'aimed_bioc.PubTator'
    
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)        
    
    out_tsv_file = out_data_dir + 'train.tsv'
    
    src_ne_type = 'protein'
    tgt_ne_type = 'protein'
    
    convert_pubtator_to_tsv_file(
        in_pubtator_file = in_pubtator_file,
        out_tsv_file     = out_tsv_file,
        src_ne_type      = src_ne_type,
        tgt_ne_type      = tgt_ne_type,
        has_end_tag      = has_end_tag,
        task_tag         = task_tag,
        re_id_spliter_str= re_id_spliter_str,
        normalized_type_dict = normalized_type_dict,
        spacy_model      = spacy_model,
        neg_label        = neg_label,
        pos_label        = 'AIMED-Association')
    

def gen_aimed_sent_dataset(
        in_data_dir,
        out_data_dir,
        spacy_model,
        has_end_tag,
        re_id_spliter_str,
        normalized_type_dict,
        task_tag,
        neg_label):
    
    in_pubtator_file = in_data_dir + 'aimed_bioc.Sen.PubTator'
    
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)        
    
    out_tsv_file = out_data_dir + 'train.tsv'
    
    src_ne_type = 'Gene'
    tgt_ne_type = 'Gene'
    
    convert_pubtator_to_tsv_file_with_mask_NEs(
        in_pubtator_file = in_pubtator_file,
        out_tsv_file     = out_tsv_file,
        src_ne_type      = src_ne_type,
        tgt_ne_type      = tgt_ne_type,
        has_end_tag      = has_end_tag,
        task_tag         = task_tag,
        re_id_spliter_str= re_id_spliter_str,
        normalized_type_dict = normalized_type_dict,
        spacy_model      = spacy_model,
        neg_label        = neg_label,
        pos_label        = 'AIMED-Association')
    
def gen_hprd50_dataset(
        in_pubtator_file,
        out_data_dir,
        spacy_model,
        has_end_tag,
        re_id_spliter_str,
        normalized_type_dict,
        task_tag,
        neg_label):
    
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)        
    
    out_tsv_file = out_data_dir + 'train.tsv'
    
    src_ne_type = 'Gene'
    tgt_ne_type = 'Gene'
    
    convert_pubtator_to_tsv_file_with_mask_NEs(
        in_pubtator_file = in_pubtator_file,
        out_tsv_file     = out_tsv_file,
        src_ne_type      = src_ne_type,
        tgt_ne_type      = tgt_ne_type,
        has_end_tag      = has_end_tag,
        task_tag         = task_tag,
        re_id_spliter_str= re_id_spliter_str,
        normalized_type_dict = normalized_type_dict,
        spacy_model      = spacy_model,
        neg_label        = neg_label,
        pos_label        = 'HPRD-Association')
    
def gen_ddi_dataset(
        in_data_dir,
        out_data_dir,
        spacy_model,
        has_end_tag,
        re_id_spliter_str,
        normalized_type_dict,
        task_tag,
        neg_label):
    
    in_train_pubtator_file = in_data_dir + 'DDI.Train.PubTator'
    in_test_pubtator_file  = in_data_dir + 'DDI.Test.PubTator'
    
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)        
    
    out_train_tsv_file = out_data_dir + 'train.tsv'
    out_test_tsv_file = out_data_dir + 'test.tsv'
    
    src_ne_type = 'drug'
    tgt_ne_type = 'drug'
    
    convert_pubtator_to_tsv_file(
        in_pubtator_file = in_train_pubtator_file,
        out_tsv_file     = out_train_tsv_file,
        src_ne_type      = src_ne_type,
        tgt_ne_type      = tgt_ne_type,
        has_end_tag      = has_end_tag,
        task_tag         = task_tag,
        re_id_spliter_str= re_id_spliter_str,
        normalized_type_dict = normalized_type_dict,
        spacy_model      = spacy_model,
        neg_label        = neg_label)
    
    convert_pubtator_to_tsv_file(
        in_pubtator_file = in_test_pubtator_file,
        out_tsv_file     = out_test_tsv_file,
        src_ne_type      = src_ne_type,
        tgt_ne_type      = tgt_ne_type,
        has_end_tag      = has_end_tag,
        task_tag         = task_tag,
        re_id_spliter_str= re_id_spliter_str,
        normalized_type_dict = normalized_type_dict,
        spacy_model      = spacy_model,
        neg_label        = neg_label)
    
def gen_litcoin_dataset(
        in_data_dir,
        out_data_dir,
        spacy_model,
        has_end_tag,
        re_id_spliter_str,
        normalized_type_dict,
        task_tag,
        neg_label = 'None'):
    
    in_train_pubtator_file = in_data_dir + 'Train.PubTator'
    in_dev_pubtator_file   = in_data_dir + 'Dev.PubTator'
    in_test_pubtator_file  = in_data_dir + 'Test.PubTator'
    
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)        
    
    out_train_tsv_file = out_data_dir + 'train.tsv'
    out_dev_tsv_file   = out_data_dir + 'dev.tsv'
    out_test_tsv_file  = out_data_dir + 'test.tsv'
    
    src_tgt_pairs = set(
        [('ChemicalEntity', 'ChemicalEntity'),
         ('ChemicalEntity', 'DiseaseOrPhenotypicFeature'),
         ('ChemicalEntity', 'GeneOrGeneProduct'),
         ('DiseaseOrPhenotypicFeature', 'GeneOrGeneProduct'),
         ('GeneOrGeneProduct', 'GeneOrGeneProduct')])
    
    convert_pubtator_to_tsv_file(
        in_pubtator_file = in_train_pubtator_file,
        out_tsv_file     = out_train_tsv_file,
        src_tgt_pairs    = src_tgt_pairs,
        has_end_tag      = has_end_tag,
        task_tag         = task_tag,
        re_id_spliter_str= re_id_spliter_str,
        normalized_type_dict = normalized_type_dict,
        spacy_model      = spacy_model,
        neg_label        = neg_label)    
    
    convert_pubtator_to_tsv_file(
        in_pubtator_file = in_dev_pubtator_file,
        out_tsv_file     = out_dev_tsv_file,
        src_tgt_pairs    = src_tgt_pairs,
        has_end_tag      = has_end_tag,
        task_tag         = task_tag,
        re_id_spliter_str= re_id_spliter_str,
        normalized_type_dict = normalized_type_dict,
        spacy_model      = spacy_model,
        neg_label        = neg_label)    
    
    convert_pubtator_to_tsv_file(
        in_pubtator_file = in_test_pubtator_file,
        out_tsv_file     = out_test_tsv_file,
        src_tgt_pairs    = src_tgt_pairs,
        has_end_tag      = has_end_tag,
        task_tag         = task_tag,
        re_id_spliter_str= re_id_spliter_str,
        normalized_type_dict = normalized_type_dict,
        spacy_model      = spacy_model,
        neg_label        = neg_label)
    

def dump_tsv_file(        
        in_main_sets,
        in_others_sets,
        out_tsv_file,
        litcoin_pmids,
        normalize_pair_2_rel_type='',
        normalize_pair_2_pair='',
        normalize_tag_2_tag='',
        num_train_biored=-1,
        to_remove_question=False,
        to_merge_neg_2_none=False):
    
    class NormalizePair:
        
        def __init__(self,
                     src,
                     tgt,
                     orig_rel,
                     new_rel):
            self.src = src
            self.tgt = tgt
            self.orig_rel = orig_rel
            self.new_rel = new_rel
            
    class Pair2Pair:
        
        def __init__(self,
                     src,
                     tgt,
                     new_src,
                     new_tgt):
            self.src = src
            self.tgt = tgt
            self.new_src = new_src
            self.new_tgt = new_tgt
    
    class Tag2Tag:
        
        def __init__(self,
                     tag,
                     new_tag):
            self.tag = tag
            self.new_tag = new_tag
    
    def __remove_question(line):
        
        tks = line.split('\t')
        
        tks[7] = tks[7].split('[SEP]')[-1].rstrip().strip()
        
        return '\t'.join(tks)
    
    normalize_pair_2_rel_type_list = []
    normalize_pair_2_pair_list = []
    normalize_tag_2_tag_list = []
    
    if normalize_pair_2_rel_type != '':
        for _normalize_pair_2_rel_type in normalize_pair_2_rel_type.split(';'):
            if _normalize_pair_2_rel_type != '':
                tks = _normalize_pair_2_rel_type.split('|')
                src = tks[0]
                tgt = tks[1]
                orig_rel = tks[2]
                new_rel = tks[3]
                normalize_pair_2_rel_type_list.append(
                    NormalizePair(src, tgt, orig_rel, new_rel))
            
    if normalize_pair_2_pair != '':
        for _normalize_pair_2_pair in normalize_pair_2_pair.split(';'):
            if _normalize_pair_2_pair != '':
                tks = _normalize_pair_2_pair.split('|')
                src = tks[0]
                tgt = tks[1]
                new_src = tks[2]
                new_tgt = tks[3]
                normalize_pair_2_pair_list.append(
                    Pair2Pair(src, tgt, new_src, new_tgt))
            
    if normalize_tag_2_tag != '':
        for _normalize_tag_2_tag in normalize_tag_2_tag.split(';'):
            if _normalize_tag_2_tag != '':
                tks = _normalize_tag_2_tag.split('|')
                tag = tks[0]
                new_tag = tks[1]
                normalize_tag_2_tag_list.append(
                    Tag2Tag(tag, new_tag))
                
    with open(out_tsv_file, 'w', encoding='utf8') as writer:
        for in_tsv_file in in_main_sets.split(';'): 
            if in_tsv_file == '':
                continue
            in_tsv_file = in_tsv_file.split('|')[0]
            pmids = set()
            with open(in_tsv_file, 'r', encoding='utf8') as reader:
                for line in reader:
                    line = line.rstrip()
                    if line.rstrip() == '':
                         continue
                    pmid = line.split('\t')[0]
                    
                    pmids.add(pmid)
                    if num_train_biored != -1 and len(pmids) > num_train_biored:
                        break
                    for _normalize_pair_2_rel_type in normalize_pair_2_rel_type_list:
                        if _normalize_pair_2_rel_type.src in line:
                            if _normalize_pair_2_rel_type.tgt in line:
                                if _normalize_pair_2_rel_type.orig_rel == '*':
                                    _tks = line.split('\t')
                                    label = _tks[-1]
                                    if to_merge_neg_2_none and _tks[-1].startswith('None'):
                                        _tks[-1] = 'None'
                                    if not label.startswith('None'):
                                        _tks[-1] = _normalize_pair_2_rel_type.new_rel
                                    line = '\t'.join(_tks)
                                    
                                else:
                                    _tks = line.split('\t')
                                    if to_merge_neg_2_none and _tks[-1].startswith('None'):
                                        _tks[-1] = 'None'
                                    if _tks[-1] == _normalize_pair_2_rel_type.orig_rel:
                                        _tks[-1] = _normalize_pair_2_rel_type.new_rel
                                    line = '\t'.join(_tks)
                                    
                    for _normalize_pair_2_pair in normalize_pair_2_pair_list:
                        _tks = line.split('\t')
                        if (_normalize_pair_2_pair.src in _tks[7]) and (_normalize_pair_2_pair.tgt in _tks[7]):
                            _tks[7] = _tks[7].replace(_normalize_pair_2_pair.src, _normalize_pair_2_pair.new_src)
                            _tks[7] = _tks[7].replace(_normalize_pair_2_pair.tgt, _normalize_pair_2_pair.new_tgt)
                            line = '\t'.join(_tks)
                                    
                    for _normalize_tag_2_tag in normalize_tag_2_tag_list:
                            line = line.replace(_normalize_tag_2_tag.tag, _normalize_tag_2_tag.new_tag)
                
                    if to_remove_question:
                        line = __remove_question(line)
                
                    writer.write(line + '\n')
        
        for in_tsv_file in in_others_sets.split(';'):
            if in_tsv_file == '':
                continue
            tks = in_tsv_file.split('|')
            ignored_label = ''
            if len(tks) > 1:
                ignored_label = tks[1]
                in_tsv_file = tks[0]
            with open(in_tsv_file, 'r', encoding='utf8') as reader:
                for line in reader:
                    line = line.rstrip()
                    tks = line.rstrip().split('\t')
                    label = tks[-1]      
                    pmid = tks[0]
                    # remove cdr overlaps with litcoin
                    if pmid in litcoin_pmids:
                        continue
                    if line == '':
                        continue
                    if label != ignored_label:
                        for _normalize_pair_2_rel_type in normalize_pair_2_rel_type_list:
                            if _normalize_pair_2_rel_type.src in line:
                                if _normalize_pair_2_rel_type.tgt in line:
                                    if _normalize_pair_2_rel_type.orig_rel == '*':
                                        _tks = line.split('\t')
                                        label = _tks[-1]
                                        if to_merge_neg_2_none and label.startswith('None'):
                                            _tks[-1] = 'None'
                                        if not label.startswith('None'):
                                            _tks[-1] = _normalize_pair_2_rel_type.new_rel
                                        
                                        line = '\t'.join(_tks)
                                        
                                    else:
                                        _tks = line.split('\t')
                                        if to_merge_neg_2_none and _tks[-1].startswith('None'):
                                            _tks[-1] = 'None'
                                        if _tks[-1] == _normalize_pair_2_rel_type.orig_rel:
                                            _tks[-1] = _normalize_pair_2_rel_type.new_rel
                                        line = '\t'.join(_tks)
                        
                        for _normalize_pair_2_pair in normalize_pair_2_pair_list:
                            _tks = line.split('\t')
                            if (_normalize_pair_2_pair.src in _tks[7]) and (_normalize_pair_2_pair.tgt in _tks[7]):
                                _tks[7] = _tks[7].replace(_normalize_pair_2_pair.src, _normalize_pair_2_pair.new_src)
                                _tks[7] = _tks[7].replace(_normalize_pair_2_pair.tgt, _normalize_pair_2_pair.new_tgt)
                                line = '\t'.join(_tks)
                                
                        for _normalize_tag_2_tag in normalize_tag_2_tag_list:
                            line = line.replace(_normalize_tag_2_tag.tag, _normalize_tag_2_tag.new_tag)
                        
                        if to_remove_question:
                            line = __remove_question(line)
                        
                        writer.write(line + '\n')
        

def combine_tsv_files(
        in_main_train_files,
        in_main_test_files,
        in_other_train_files, 
        in_other_test_files, 
        out_train_tsv_file,
        out_test_tsv_file,
        normalize_pair_2_rel_type,
        normalize_pair_2_pair,
        normalize_tag_2_tag,
        num_train_biored=-1,
        to_remove_question=False,
        to_merge_neg_2_none=False):
    
    if out_train_tsv_file != "":
        if not os.path.exists(os.path.dirname(out_train_tsv_file)):
            os.makedirs(os.path.dirname(out_train_tsv_file))
    if out_test_tsv_file != "":
        if not os.path.exists(os.path.dirname(out_test_tsv_file)):
            os.makedirs(os.path.dirname(out_test_tsv_file))
        
    main_pmids = set()

    for in_tsv_file in in_main_train_files.split(';'):
        if in_tsv_file == '':
            continue
        in_tsv_file = in_tsv_file.split('|')[0]
        with open(in_tsv_file, 'r', encoding='utf8') as reader:
            for line in reader:
                tks = line.rstrip().split('\t')
                pmid = tks[0]
                main_pmids.add(pmid)
        
    for in_tsv_file in in_main_test_files.split(';'):
        if in_tsv_file == '':
            continue
        in_tsv_file = in_tsv_file.split('|')[0]
        with open(in_tsv_file, 'r', encoding='utf8') as reader:
            for line in reader:
                tks = line.rstrip().split('\t')
                pmid = tks[0]
                main_pmids.add(pmid)
    
    if out_train_tsv_file != "":
        dump_tsv_file(    
            in_main_train_files,
            in_other_train_files,
            out_train_tsv_file,
            main_pmids,
            normalize_pair_2_rel_type,
            normalize_pair_2_pair,
            normalize_tag_2_tag,
            num_train_biored,
            to_remove_question,
            to_merge_neg_2_none)
        
    if out_test_tsv_file != "":
        dump_tsv_file(
            in_main_test_files,
            in_other_test_files,
            out_test_tsv_file,
            main_pmids,
            normalize_pair_2_rel_type,
            normalize_pair_2_pair,
            normalize_tag_2_tag,
            -1,
            to_remove_question,
            to_merge_neg_2_none)

def __load_pmid_2_index_2_groupID_dict(in_tmvar_file):
    
    pmid_2_index_2_groupID_dict = defaultdict(dict)
    with open(in_tmvar_file, 'r', encoding='utf8') as tmvar_reader:
        
        for line in tmvar_reader:
            
            tks = line.rstrip().split('\t')
            
            if len(tks) > 5:
                pmid = tks[0]
                # tmVar:p|SUB|P|75|A;HGVS:p.P75A;VariantGroup:0;CorrespondingGene:3175;RS#:74805019;CA#:7570459
                group_id = ''
                tmVar_id = ''
                rs_id    = ''
                gene_id  = ''
                index = tks[1] + '|' + tks[2]
                for var_id in tks[5].split(';'):
                    if var_id.startswith('VariantGroup'):
                        group_id = var_id
                    elif var_id.startswith('tmVar'):
                        tmVar_id = var_id[6:]
                    elif var_id.startswith('RS#:'):
                        rs_id = 'rs' + var_id[4:]
                    elif var_id.startswith('CorrespondingGene'):
                        gene_id = var_id[18:]
                
                if group_id != '':
                    pmid_2_index_2_groupID_dict[pmid][index] = (group_id, gene_id)
                '''
                if group_id != '' and tmVar_id != '':
                    pmid_2_index_2_groupID_dict[pmid][tmVar_id] = (group_id, gene_id)
                if group_id != '' and tmVar_id != '' and rs_id != '':
                    pmid_2_index_2_groupID_dict[pmid][rs_id] = (group_id, gene_id)'''
    
    return pmid_2_index_2_groupID_dict

def __update_pmid_2_tmvarID_2_groupID_dict(
        in_gene_var_file,
        pmid_2_tmvarID_2_groupID_dict):
    
    with open(in_gene_var_file, 'r', encoding='utf8') as tsv_reader:
        
        for line in tsv_reader:
            
            tks = line.split('\t')
            if len(tks) == 3:
                pmid = tks[0]
                var_id = tks[1]
                gene_id = tks[2]
                
                if var_id not in pmid_2_tmvarID_2_groupID_dict[pmid]:
                    pmid_2_tmvarID_2_groupID_dict[pmid][var_id] = (var_id, gene_id)

def __load_dgv_relations(
        all_documents,
        pmid_2_index_2_groupID_dict):
    
    
    gene_tag = 'GeneOrGeneProduct'
    variant_tag = 'SequenceVariant'
    disease_tag = 'DiseaseOrPhenotypicFeature'
    
    for document in all_documents:
        
        pmid = document.id
        document.nary_relations = {}
        
        gene_disease_pairs = set()
        variant_disease_pairs = {}
        variant_gene_pairs = {}
        
        # mapping id to NE type
        id2type_dict = {}
        
        document.variant_gene_pairs = set()
                
        for text_instance in document.text_instances:
            for ann in text_instance.annotations:
                index = str(text_instance.offset + ann.position) + '|' + str(text_instance.offset + ann.position + ann.length)
                for id in ann.ids:
                    id2type_dict[id] = ann.ne_type
                    # after load_pubtator_into_documents, 'SequenceVariant' will be 'GeneOrGeneProduct'
                    # now we map it back
                    if index in pmid_2_index_2_groupID_dict[pmid]:
                        id2type_dict[id] = variant_tag
                        variant_gene_pairs[id] = pmid_2_index_2_groupID_dict[pmid][index][1]
                        if pmid_2_index_2_groupID_dict[pmid][index][1] != '':
                            gene_id = pmid_2_index_2_groupID_dict[pmid][index][1]
                            document.variant_gene_pairs.add((id, gene_id))                    
                        
        for relation_pair, rel_type in document.relation_pairs.items():
            # if id is in id2type_dict, means that it is from variant instead of gene, so just ignore it
            
            id1 = relation_pair[0]
            id2 = relation_pair[1]
            if (id1 not in id2type_dict) or (id2 not in id2type_dict):
                continue
            id1_type = id2type_dict[id1]
            id2_type = id2type_dict[id2]
            if id1_type == gene_tag and id2_type == disease_tag:
                gene_disease_pairs.add((id1, id2))
            elif id1_type == disease_tag and id2_type == gene_tag:
                gene_disease_pairs.add((id2, id1))
            elif id1_type == variant_tag and id2_type == disease_tag:
                variant_disease_pairs[(id1, id2)] = rel_type
            elif id1_type == disease_tag and id2_type == variant_tag:
                variant_disease_pairs[(id2, id1)] = rel_type
        
        
        
        for variant_disease_pair, rel_type in variant_disease_pairs.items():
            variant_id = variant_disease_pair[0]
            disease_id = variant_disease_pair[1]
            if variant_id not in variant_gene_pairs:
                continue
            gene_id    = variant_gene_pairs[variant_id]
            document.nary_relations[(disease_id, gene_id, variant_id)] = rel_type
            
def dump_pharmgkb_dataset(
            in_pubtator_file,
            out_data_dir,
            spacy_model,
            has_end_tag,
            re_id_spliter_str,
            normalized_type_dict,
            task_tag,
            neg_label):
            
    out_tsv_file = out_data_dir + 'train.tsv'
    
    src_ne_type = 'Chemical'
    tgt_ne_type = 'Gene'
        
    pmid_2_index_2_groupID_dict = __load_pmid_2_index_2_groupID_dict(in_pubtator_file)
    #__update_pmid_2_index_2_groupID_dict(in_gene_var_file,
    #                                       pmid_2_index_2_groupID_dict)
    
    all_documents = load_pubtator_into_documents(
            in_pubtator_file     = in_pubtator_file, 
            normalized_type_dict = normalized_type_dict,
            re_id_spliter_str    = re_id_spliter_str,
            pmid_2_index_2_groupID_dict = pmid_2_index_2_groupID_dict)

    __load_dgv_relations(
            all_documents,
            pmid_2_index_2_groupID_dict)
    

    utils.tokenize_documents_by_spacy(all_documents, spacy_model)
    
    # do not use n-ary, but tag variants for g-d pair
    utils.dump_documents_2_bert_gt_format(
        all_documents, 
        out_tsv_file, 
        src_ne_type,
        tgt_ne_type,
        has_end_tag = has_end_tag,
        task_tag    = task_tag,
        has_dgv     = False,
        use_corresponding_gene_id = True,
        has_ne_type = True,
        neg_label   = neg_label)
    
def dump_emu_bc_dataset(
            in_pubtator_file,
            out_data_dir,
            spacy_model,
            has_end_tag,
            re_id_spliter_str,
            normalized_type_dict,
            task_tag,
            neg_label):
            
    out_tsv_file = out_data_dir + 'train.tsv'
        
    src_ne_type = 'Disease'
    tgt_ne_type = 'Gene'
        
    pmid_2_index_2_groupID_dict = __load_pmid_2_index_2_groupID_dict(in_pubtator_file)
    #__update_pmid_2_index_2_groupID_dict(in_gene_var_file,
    #                                       pmid_2_index_2_groupID_dict)
    
    all_documents = load_pubtator_into_documents(
            in_pubtator_file     = in_pubtator_file, 
            normalized_type_dict = normalized_type_dict,
            re_id_spliter_str    = re_id_spliter_str,
            pmid_2_index_2_groupID_dict = pmid_2_index_2_groupID_dict)

    __load_dgv_relations(
            all_documents,
            pmid_2_index_2_groupID_dict)
    

    utils.tokenize_documents_by_spacy(all_documents, spacy_model)
    
    # do not use n-ary, but tag variants for g-d pair
    utils.dump_documents_2_bert_gt_format(
        all_documents, 
        out_tsv_file, 
        src_ne_type,
        tgt_ne_type,
        has_end_tag = True,
        task_tag    = task_tag,
        has_dgv     = False,
        use_corresponding_gene_id = True,
        has_ne_type = True,
        neg_label   = neg_label)
    
def dump_emu_pc_dataset(
            in_pubtator_file,
            out_data_dir,
            spacy_model,
            has_end_tag,
            re_id_spliter_str,
            normalized_type_dict,
            task_tag,
            neg_label):
            
    out_tsv_file = out_data_dir + 'train.tsv'
        
    src_ne_type = 'Disease'
    tgt_ne_type = 'Gene'
        
    pmid_2_index_2_groupID_dict = __load_pmid_2_index_2_groupID_dict(in_pubtator_file)
    #__update_pmid_2_index_2_groupID_dict(in_gene_var_file,
    #                                       pmid_2_index_2_groupID_dict)
    
    all_documents = load_pubtator_into_documents(
            in_pubtator_file     = in_pubtator_file, 
            normalized_type_dict = normalized_type_dict,
            re_id_spliter_str    = re_id_spliter_str,
            pmid_2_index_2_groupID_dict = pmid_2_index_2_groupID_dict)

    __load_dgv_relations(
            all_documents,
            pmid_2_index_2_groupID_dict)
    

    utils.tokenize_documents_by_spacy(all_documents, spacy_model)
    
    # do not use n-ary, but tag variants for g-d pair
    utils.dump_documents_2_bert_gt_format(
        all_documents, 
        out_tsv_file, 
        src_ne_type,
        tgt_ne_type,
        has_end_tag = True,
        task_tag    = task_tag,
        has_dgv     = False,
        use_corresponding_gene_id = True,
        has_ne_type = True,
        neg_label   = neg_label)

def dump_disgenet_dataset(
            in_pubtator_file,
            out_data_dir,
            spacy_model,
            has_end_tag,
            re_id_spliter_str,
            normalized_type_dict,
            task_tag,
            neg_label):
            
    out_tsv_file = out_data_dir + 'train.tsv'
        
    src_ne_type = 'Disease'
    tgt_ne_type = 'Gene'
        
    pmid_2_index_2_groupID_dict = __load_pmid_2_index_2_groupID_dict(in_pubtator_file)
    #__update_pmid_2_index_2_groupID_dict(in_gene_var_file,
    #                                       pmid_2_index_2_groupID_dict)
    
    all_documents = load_pubtator_into_documents(
            in_pubtator_file     = in_pubtator_file, 
            normalized_type_dict = normalized_type_dict,
            re_id_spliter_str    = re_id_spliter_str,
            pmid_2_index_2_groupID_dict = pmid_2_index_2_groupID_dict)

    __load_dgv_relations(
            all_documents,
            pmid_2_index_2_groupID_dict)
    

    utils.tokenize_documents_by_spacy(all_documents, spacy_model)
    
    # do not use n-ary, but tag variants for g-d pair
    utils.dump_documents_2_bert_gt_format(
        all_documents, 
        out_tsv_file, 
        src_ne_type,
        tgt_ne_type,
        has_end_tag = True,
        task_tag    = task_tag,
        has_dgv     = False,
        use_corresponding_gene_id = True,
        has_ne_type = True,
        neg_label   = neg_label)

def dump_litcoinx_with_dgv_dataset(
            in_pubtator_file,
            out_tsv_file,
            in_tmvar_file,
            in_gene_var_file,
            spacy_model,
            re_id_spliter_str,
            normalized_type_dict,
            task_tag):
        
    src_ne_type = 'Any'
    tgt_ne_type = 'Any'
    
    pmid_2_index_2_groupID_dict = __load_pmid_2_index_2_groupID_dict(in_tmvar_file)
    #__update_pmid_2_index_2_groupID_dict(in_gene_var_file,
    #                                       pmid_2_index_2_groupID_dict)
    
    all_documents = load_pubtator_into_documents(
            in_pubtator_file     = in_pubtator_file, 
            normalized_type_dict = normalized_type_dict,
            re_id_spliter_str    = re_id_spliter_str,
            pmid_2_index_2_groupID_dict = pmid_2_index_2_groupID_dict)

    __load_dgv_relations(
            all_documents,
            pmid_2_index_2_groupID_dict)
    

    utils.tokenize_documents_by_spacy(all_documents, spacy_model)
    utils.dump_documents_2_bert_gt_format(
        all_documents, 
        out_tsv_file, 
        src_ne_type,
        tgt_ne_type,
        has_end_tag = True,
        task_tag    = task_tag,
        has_dgv     = False,
        use_corresponding_gene_id = True)

def  gen_phargkb_dataset(
            in_pubtator_file,
            out_data_dir,
            spacy_model,
            has_end_tag,
            re_id_spliter_str,
            task_tag,
            normalized_type_dict = dict):
    
    
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)        
    
    out_tsv_file = out_data_dir + 'train.tsv'
    src_ne_type  = 'Chemical'
    tgt_ne_type  = 'Mutation'
    
    all_documents = load_pubtator_into_documents(
            in_pubtator_file     = in_pubtator_file, 
            normalized_type_dict = normalized_type_dict,
            re_id_spliter_str    = re_id_spliter_str)
   

    utils.tokenize_documents_by_spacy(all_documents, spacy_model)
    utils.dump_documents_2_bert_gt_format(
        all_documents, 
        out_tsv_file, 
        src_ne_type,
        tgt_ne_type,
        has_end_tag = True,
        task_tag    = task_tag)
    
def gen_drugprot_dataset(
        in_abs_tsv_file,
        in_ann_tsv_file,
        out_bert_gt_file,
        task_tag,
        neg_label,
        in_rel_tsv_file = '',
        is_test_set = False,
        normalized_type_dict = None):
    
    random.seed(1234)
    
    spacy_model = 'en_core_sci_md'
    
    if not os.path.exists(os.path.dirname(out_bert_gt_file)):
        os.makedirs(os.path.dirname(out_bert_gt_file))
    
    all_documents = drugprot_loader.load_drugprot_into_document_list(
        in_abs_tsv_file = in_abs_tsv_file,
        in_ann_tsv_file = in_ann_tsv_file,
        spacy_model     = spacy_model,
        normalized_type_dict = normalized_type_dict,
        in_rel_tsv_file = in_rel_tsv_file)
    
    utils.dump_documents_2_bert_gt_format_by_sent_level(
        all_documents          = all_documents,
        out_bert_file          = out_bert_gt_file, 
        src_ne_type            = 'CHEMICAL',
        tgt_ne_type            = 'GENE',
        is_test_set            = is_test_set,
        to_mask_src_and_tgt    = False,
        to_insert_src_and_tgt_at_left = True,
        has_end_tag            = True,
        task_tag               = task_tag,
        neg_label              = neg_label,
        add_ne_type            = True)

if __name__ == '__main__':
    
    options, args      = parser.parse_args()
    
    exp_option            = options.exp_option
    in_main_train_files   = options.in_main_train_files
    in_main_test_files    = options.in_main_test_files
    in_other_train_files  = options.in_other_train_files
    in_other_test_files   = options.in_other_test_files    
    out_train_tsv_file    = options.out_train_tsv_file
    out_test_tsv_file     = options.out_test_tsv_file
    
    in_test_pubtator_file     = options.in_test_pubtator_file
    normalize_pair_2_rel_type = options.normalize_pair_2_rel_type
    normalize_pair_2_pair     = options.normalize_pair_2_pair
    normalize_tag_2_tag       = options.normalize_tag_2_tag
    num_train_biored          = options.num_train_biored
    only_co_occurrence_sent   = options.only_co_occurrence_sent
    to_mask_src_and_tgt       = options.to_mask_src_and_tgt
    out_data_dir              = options.out_data_dir
    to_remove_question        = options.to_remove_question
    to_merge_neg_2_none       = options.to_merge_neg_2_none
    
    spacy_model = 'en_core_sci_md'
    normalized_type_dict = {}
    random.seed(1234)
        
    # standard train and dev
    if exp_option == 'cdr':
        in_cdr_dir       = 'datasets/cdr/'
        out_cdr_dir      = 'datasets/cdr/processed/'
        has_end_tag      = True
        re_id_spliter_str= r'\|'
        task_tag    = '[CID]'
        neg_label   = 'None-CID'
        
        gen_cdr_dataset(
            in_cdr_dir  = in_cdr_dir,
            out_cdr_dir = out_cdr_dir,
            spacy_model = spacy_model,
            has_end_tag = has_end_tag,
            re_id_spliter_str = re_id_spliter_str,
            normalized_type_dict = normalized_type_dict,
            task_tag = task_tag,
            neg_label = neg_label)
        
    elif exp_option == 'biored':
        in_data_dir       = 'datasets/ncbi_relation/'
        out_data_dir      = 'datasets/ncbi_relation/processed/'
        has_end_tag       = True
        re_id_spliter_str= r'[\,\;]'
        normalized_type_dict = {'SequenceVariant':'GeneOrGeneProduct'}
        task_tag    = '[Litcoin]'
        
        gen_litcoin_dataset(
            in_data_dir  = in_data_dir,
            out_data_dir = out_data_dir,
            spacy_model  = spacy_model,
            has_end_tag  = has_end_tag,
            re_id_spliter_str = re_id_spliter_str,
            normalized_type_dict = normalized_type_dict,
            task_tag = task_tag)
            
    elif exp_option == 'biored_pred':
        in_test_pubtator_file = options.in_pubtator_file
        out_test_tsv_file     = options.out_tsv_file
        has_end_tag           = True
        re_id_spliter_str     = r'[\,\;]'
        normalized_type_dict  = {'SequenceVariant':'GeneOrGeneProduct'}
        task_tag    = '[Litcoin]'
        
        src_tgt_pairs = set(
            [('ChemicalEntity', 'ChemicalEntity'),
             ('ChemicalEntity', 'DiseaseOrPhenotypicFeature'),
             ('ChemicalEntity', 'GeneOrGeneProduct'),
             ('DiseaseOrPhenotypicFeature', 'GeneOrGeneProduct'),
             ('GeneOrGeneProduct', 'GeneOrGeneProduct')])
        
        convert_pubtator_to_tsv_file(
            in_pubtator_file = in_test_pubtator_file,
            out_tsv_file     = out_test_tsv_file,
            src_tgt_pairs    = src_tgt_pairs,
            has_end_tag      = has_end_tag,
            task_tag         = task_tag,
            re_id_spliter_str= re_id_spliter_str,
            normalized_type_dict = normalized_type_dict,
            spacy_model      = spacy_model)
    
    elif exp_option == 'ddi':
        in_data_dir       = 'datasets/ddi/'
        out_data_dir      = 'datasets/ddi/processed/'
        has_end_tag       = True
        re_id_spliter_str= r'\,'        
        normalized_type_dict = {'drug':'drug', 
                                'brand':'drug', 
                                'drug_n':'drug',
                                'group':'drug'}
        task_tag    = '[DDI]'
        neg_label   = 'None-DDI'
        
        gen_ddi_dataset(
            in_data_dir  = in_data_dir,
            out_data_dir = out_data_dir,
            spacy_model  = spacy_model,
            has_end_tag  = has_end_tag,
            re_id_spliter_str = re_id_spliter_str,
            normalized_type_dict = normalized_type_dict,
            task_tag = task_tag,
            neg_label = neg_label)
                
    elif exp_option == 'drugprot':        
               
        in_train_abs_tsv_file  = 'datasets/drugprot/drugprot-gs-training-development/training/drugprot_training_abstracs.tsv'
        in_train_ann_tsv_file  = 'datasets/drugprot/drugprot-gs-training-development/training/drugprot_training_entities.tsv'
        in_train_rel_tsv_file  = 'datasets/drugprot/drugprot-gs-training-development/training/drugprot_training_relations.tsv'
        out_train_bert_gt_file = 'datasets/drugprot/drugprot-gs-training-development/processed/train.tsv'
        
        in_dev_abs_tsv_file  = 'datasets/drugprot/drugprot-gs-training-development/drugprot_development_abstracs.tsv'
        in_dev_ann_tsv_file  = 'datasets/drugprot/drugprot-gs-training-development/drugprot_development_entities.tsv'
        in_dev_rel_tsv_file  = 'datasets/drugprot/drugprot-gs-training-development/drugprot_development_relations.tsv'
        out_dev_bert_gt_file = 'datasets/drugprot/drugprot-gs-training-development/processed/dev.tsv'
        
        in_test_abs_tsv_file  = 'datasets/drugprot/drugprot-gs-training-development/test-background/test_background_abstracts.tsv'
        in_test_ann_tsv_file  = 'datasets/drugprot/drugprot-gs-training-development/test-background/test_background_entities.tsv'
        out_test_bert_gt_file = 'datasets/drugprot/drugprot-gs-training-development/processed/test.tsv'
                        
        task_tag    = '[Drugprot]'
        neg_label   = 'None-BC7'
        
        gen_drugprot_dataset(
            in_abs_tsv_file     = in_train_abs_tsv_file,
            in_ann_tsv_file     = in_train_ann_tsv_file,
            out_bert_gt_file    = out_train_bert_gt_file,
            in_rel_tsv_file     = in_train_rel_tsv_file,
            is_test_set         = False,
            task_tag            = task_tag,
            neg_label           = neg_label)
        
        gen_drugprot_dataset(
            in_abs_tsv_file     = in_dev_abs_tsv_file,
            in_ann_tsv_file     = in_dev_ann_tsv_file,
            out_bert_gt_file    = out_dev_bert_gt_file,
            in_rel_tsv_file     = in_dev_rel_tsv_file,
            is_test_set         = False,
            task_tag            = task_tag,
            neg_label           = neg_label)
        
    elif exp_option == 'aimed':              
        
        in_data_dir       = 'datasets/aimed/'
        out_data_dir      = 'datasets/aimed/processed/'
        has_end_tag       = True
        re_id_spliter_str= r'\,'
        task_tag    = '[AIMED]'
        neg_label   = 'None-AIMED'
        
        gen_aimed_sent_dataset(
            in_data_dir  = in_data_dir,
            out_data_dir = out_data_dir,
            spacy_model  = spacy_model,
            has_end_tag  = has_end_tag,
            re_id_spliter_str = re_id_spliter_str,
            normalized_type_dict = normalized_type_dict,
            task_tag = task_tag,
            neg_label = neg_label)
        
    elif exp_option == 'pharmgkb':
        
        in_pubtator_file  = 'datasets/pharmgkb/PharmGKB.PubTator'
        out_data_dir      = 'datasets/pharmgkb/processed/'
        has_end_tag       = True
        re_id_spliter_str= r'\;'
        task_tag    = '[PHARMGKB]'
        neg_label   = 'None-PHARMGKB'
        
        dump_pharmgkb_dataset(
            in_pubtator_file     = in_pubtator_file,
            out_data_dir         = out_data_dir,
            spacy_model          = spacy_model,
            has_end_tag          = has_end_tag,
            re_id_spliter_str    = re_id_spliter_str,
            normalized_type_dict = normalized_type_dict,
            task_tag             = task_tag,
            neg_label            = neg_label)  
        
    elif exp_option == 'emu_pc':
        
        in_pubtator_file  = 'datasets/emu_pc/PCa_db_novel_51.PubTator'
        out_data_dir      = 'datasets/emu_pc/processed/'
        has_end_tag       = True
        re_id_spliter_str= r'\;'
        task_tag    = '[EMU_PC]'
        neg_label   = 'None-EMU_PC'
        dump_emu_pc_dataset(
            in_pubtator_file     = in_pubtator_file,
            out_data_dir         = out_data_dir,
            spacy_model          = spacy_model,
            has_end_tag          = has_end_tag,
            re_id_spliter_str    = re_id_spliter_str,
            normalized_type_dict = normalized_type_dict,
            task_tag             = task_tag,
            neg_label            = neg_label)  
        
    elif exp_option == 'emu_bc':
        
        in_pubtator_file  = 'datasets/emu_bc/BCa_db_novel_128.PubTator'
        out_data_dir      = 'datasets/emu_bc/processed/'
        has_end_tag       = True
        re_id_spliter_str= r'\;'
        task_tag    = '[EMU_BC]'
        neg_label   = 'None-EMU_BC'
        dump_emu_bc_dataset(
            in_pubtator_file     = in_pubtator_file,
            out_data_dir         = out_data_dir,
            spacy_model          = spacy_model,
            has_end_tag          = has_end_tag,
            re_id_spliter_str    = re_id_spliter_str,
            normalized_type_dict = normalized_type_dict,
            task_tag             = task_tag,
            neg_label            = neg_label)
        
    elif exp_option == 'disgenet':
        
        in_pubtator_file  = 'datasets/disgenet/DisGeNET.PubTator'
        out_data_dir      = 'datasets/disgenet/processed/'
        has_end_tag       = True
        re_id_spliter_str= r'\;'
        task_tag    = '[DISGENET]'    
        neg_label   = 'None-DISGENET'
        
        dump_disgenet_dataset(
            in_pubtator_file     = in_pubtator_file,
            out_data_dir         = out_data_dir,
            spacy_model          = spacy_model,
            has_end_tag          = has_end_tag,
            re_id_spliter_str    = re_id_spliter_str,
            normalized_type_dict = normalized_type_dict,
            task_tag             = task_tag,
            neg_label            = neg_label)

    elif exp_option == 'hprd50':
        
        #in_pubtator_file  = 'datasets/hprd50/HPRD50.PubTator'
        in_pubtator_file  = 'datasets/hprd50/hprd50_bioc.PubTator'
        out_data_dir      = 'datasets/hprd50/processed/'
        has_end_tag       = True
        re_id_spliter_str= r'\|'
        task_tag    = '[HPRD50]'
        neg_label   = 'None-HPRD50'
        
        gen_hprd50_dataset(
            in_pubtator_file  = in_pubtator_file,
            out_data_dir = out_data_dir,
            spacy_model  = spacy_model,
            has_end_tag  = has_end_tag,
            re_id_spliter_str = re_id_spliter_str,
            normalized_type_dict = normalized_type_dict,
            task_tag = task_tag,
            neg_label = neg_label)
        
    elif exp_option == 'combine_sets': 
        
        combine_tsv_files(
                in_main_train_files    = in_main_train_files,
                in_main_test_files     = in_main_test_files,
                in_other_train_files   = in_other_train_files, 
                in_other_test_files    = in_other_test_files, 
                out_train_tsv_file     = out_train_tsv_file,
                out_test_tsv_file      = out_test_tsv_file,
                normalize_pair_2_rel_type = normalize_pair_2_rel_type,
                normalize_pair_2_pair     = normalize_pair_2_pair,
                normalize_tag_2_tag       = normalize_tag_2_tag,
                num_train_biored          = num_train_biored,
                to_remove_question        = to_remove_question,
                to_merge_neg_2_none       = to_merge_neg_2_none)
                
        