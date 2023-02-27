# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 21:41:35 2021

@author: laip2
"""

import pandas as pd
from collections import defaultdict
import numpy as np
import sys
sys.path.append('src/dataset_format_converter')
import os.path
import re
import distutils.util
from convert_pubtator_2_tsv import load_pubtator_into_documents
from sentence_spliter import tokenize_documents_by_brat

import optparse

parser = optparse.OptionParser()

parser.add_option('--exp_option',                     action="store",
                     dest="exp_option",               help="biored: train on 400 train and evaluate on 100 dev; \
                                                      biorex: train on biorex train and evaluate on biored 100 test", default="")

parser.add_option('--in_gold_tsv_file',               action="store",
                     dest="in_gold_tsv_file",         help="input gold tsv file",
                                                      default="")

parser.add_option('--in_pred_tsv_file',               action="store",
                     dest="in_pred_tsv_file",         help="input prediction tsv file",
                                                      default="")

parser.add_option('--out_bin_result_file',            action="store",
                     dest="out_bin_result_file",      help="binary classification scores file",
                                                      default="")

parser.add_option('--out_result_file',                action="store",
                     dest="out_result_file",          help="multi-class classification scores file",
                                                      default="")

parser.add_option('--out_pred_pubtator_file',         action="store",
                     dest="out_pred_pubtator_file",   help="output prediction pubtator file",
                                                      default="")

parser.add_option('--biored_eval_filter',             action="store",
                     dest="biored_eval_filter",       help="ne_type pair for run_simple_eval",
                                                      default="")

class RelInfo:
    
    def __init__(self, id1, id2, rel_type, score):
        self.id1   = id1
        self.id2   = id2
        self.rel_type = rel_type
        self.score = score


    
def __softmax(vector):
	e = np.exp(vector)
	return e / e.sum()

def add_relation_pairs_dict(
        in_gold_tsv_file, 
        in_pred_tsv_file,
        pmid_2_rel_pairs_dict,
        labels,
        label = 'label',
        label_index = -2):
    
    #print('in_gold_tsv_file', in_gold_tsv_file)
    #print('in_pred_tsv_file', in_pred_tsv_file)
    testdf = pd.read_csv(in_gold_tsv_file, sep="\t", index_col=0)
    try:
        preddf = pd.read_csv(in_pred_tsv_file, sep="\t", header=None)
    except:
        return
    
    pred = [preddf.iloc[i].tolist() for i in preddf.index]
    pred_class = [np.argmax(v) for v in pred]
    pred_scores = [np.max(__softmax(v)) for v in pred]
    
    
    test_labels = None
    test_is_in_same_sents = None
    
    try:
        test_labels           = testdf[label]
        index_list            = testdf["pmid"]
        id1_list              = testdf["id1"]
        id2_list              = testdf["id2"]
    except:
        testdf = pd.read_csv(in_gold_tsv_file, sep="\t", header=None)
        test_labels           = testdf.iloc[:,label_index]
        index_list            = testdf.iloc[:,0]
        id1_list              = testdf.iloc[:,3]
        id2_list              = testdf.iloc[:,4]
    
    
    for _pred, _gold, index, id1, id2, score in zip(pred_class, test_labels, index_list, id1_list, id2_list, pred_scores):
        pred_label = labels[_pred]
        if not pred_label.startswith('None'):
            #print(index, pred_label)
            sindex = str(index)
            if sindex not in pmid_2_rel_pairs_dict:
                pmid_2_rel_pairs_dict[sindex] = set()
                #print(sindex)
            pmid_2_rel_pairs_dict[sindex].add(RelInfo(str(id1), str(id2), pred_label, score))
            
def add_relation_pairs_dict_by_key(
        in_gold_tsv_file, 
        in_pred_tsv_file,
        pmid_2_rel_pairs_dict,
        labels,
        label = 'label',
        label_index = -2):
    
    #print('in_gold_tsv_file', in_gold_tsv_file)
    #print('in_pred_tsv_file', in_pred_tsv_file)
    try:    
        testdf = pd.read_csv(in_gold_tsv_file, sep="\t", index_col=0)
        preddf = pd.read_csv(in_pred_tsv_file, sep="\t", header=None)
    except:
        return
    
    pred = [preddf.iloc[i].tolist() for i in preddf.index]
    pred_class = [np.argmax(v) for v in pred]
    pred_scores = [np.max(__softmax(v)) for v in pred]
    
    
    test_labels = None
    test_is_in_same_sents = None
    
    try:
        test_labels           = testdf[label]
        index_list            = testdf["pmid"]
        id1_list              = testdf["id1"]
        id2_list              = testdf["id2"]
    except:
        testdf = pd.read_csv(in_gold_tsv_file, sep="\t", header=None)
        test_labels           = testdf.iloc[:,label_index]
        index_list            = testdf.iloc[:,0]
        id1_list              = testdf.iloc[:,3]
        id2_list              = testdf.iloc[:,4]
    
    
    for _pred, _gold, index, id1, id2, score in zip(pred_class, test_labels, index_list, id1_list, id2_list, pred_scores):
        pred_label = labels[_pred]
        if not pred_label.startswith('None'):
            #print(index, pred_label)
            sindex = str(index)
            if sindex not in pmid_2_rel_pairs_dict:
                pmid_2_rel_pairs_dict[sindex] = {}
                #print(sindex)
            pmid_2_rel_pairs_dict[sindex][(str(id1), str(id2))] = RelInfo(str(id1), str(id2), pred_label, score)
            
def add_gold_relation_pairs_dict(
        in_gold_tsv_file,
        pmid_2_rel_pairs_dict,
        label = 'label',
        label_index = -2):
    
    #print('in_gold_tsv_file', in_gold_tsv_file)
    #print('in_pred_tsv_file', in_pred_tsv_file)
    testdf = pd.read_csv(in_gold_tsv_file, sep="\t", index_col=0)
    
    try:
        test_labels           = testdf[label]
        index_list            = testdf["pmid"]
        id1_list              = testdf["id1"]
        id2_list              = testdf["id2"]
    except:
        testdf = pd.read_csv(in_gold_tsv_file, sep="\t", header=None)
        test_labels           = testdf.iloc[:,label_index]
        index_list            = testdf.iloc[:,0]
        id1_list              = testdf.iloc[:,3]
        id2_list              = testdf.iloc[:,4]
    
    for _gold, index, id1, id2 in zip(test_labels, index_list, id1_list, id2_list):
        if _gold != 'None':
            #print(index, pred_label)
            sindex = str(index)
            if sindex not in pmid_2_rel_pairs_dict:
                pmid_2_rel_pairs_dict[sindex] = set()
                #print(sindex)
            pmid_2_rel_pairs_dict[sindex].add((str(id1), str(id2), _gold))
            
def dump_pred_2_pubtator_file(in_pubtator_file, 
                              in_gold_tsv_file,
                              in_pred_tsv_file,
                              out_pubtator_pred_file,
                              labels,
                              pmid_2_index_2_groupID_dict = None):
    
    pmid_2_rel_pairs_dict = {}
    add_relation_pairs_dict(
            in_gold_tsv_file,
            in_pred_tsv_file,
            pmid_2_rel_pairs_dict,
            labels = labels)
    
    # refine pmid_2_rel_pairs_dict accodring to nary_relations
    for pmid in pmid_2_rel_pairs_dict.keys():
        
        rel_pairs = pmid_2_rel_pairs_dict[pmid]
        
        update_me_dict = {}
        for i, rel_pair in enumerate(rel_pairs):
            
            src_id   = rel_pair.id1
            tgt_id   = rel_pair.id2
            rel_type = rel_pair.rel_type
            score    = rel_pair.score
            if '@' in src_id:
                # gene_id + '@' + variant_id
                src_id1 = src_id.split('@')[0]
                src_id2 = src_id.split('@')[1]
                if rel_type == 'Positive_Correlation':
                    update_me_dict[(src_id1, tgt_id)] = 'Association'
                    #update_me_dict[(src_id2, tgt_id)] = rel_type
                #update_me_dict[(src_id1, tgt_id)] = 'Association'
                update_me_dict[(src_id2, tgt_id)] = (rel_type, score)
                
        for update_me, (new_rel_type, new_rel_score) in update_me_dict.items():
            
            update_id1 = update_me[0]
            update_id2 = update_me[1]
            
            updated = False
            
            for rel_pair in rel_pairs:
                
                src_id        = rel_pair.id1
                tgt_id        = rel_pair.id2
                old_rel_type  = rel_pair.label
                old_rel_score = rel_pair.score
                
                if update_id1 == src_id and update_id2 == tgt_id:
                    old_rel = RelInfo(src_id, tgt_id, old_rel_type, old_rel_score)
                    new_rel = RelInfo(src_id, tgt_id, new_rel_type, new_rel_score)
                    updated = True
                    break
                elif update_id2 == src_id and update_id1 == tgt_id:
                    old_rel = RelInfo(src_id, tgt_id, old_rel_type, old_rel_score)
                    new_rel = RelInfo(src_id, tgt_id, new_rel_type, new_rel_score)
                    updated = True
                    break
            
            if updated:
                for pair in pmid_2_rel_pairs_dict[pmid]:
                    if pair.id1 == old_rel.id1 and pair.id2 == old_rel.id2:
                        pair.rel_type  = new_rel.rel_type
                        pair.rel_score = new_rel.rel_score
            else:
                pmid_2_rel_pairs_dict[pmid].add(RelInfo(update_id1, update_id2, new_rel_type, new_rel_score))
                
                    
    
    pred_writer = open(out_pubtator_pred_file, 'w', encoding='utf8')
        
    id2ne_type_dict = {}
    id2index = {}
            
    with open(in_pubtator_file, 'r', encoding='utf8') as pub_reader:
        
        pmid = ''
        
        for line in pub_reader:
            line = line.rstrip()
            
            if line == '':
                #print('len(pmid_2_rel_pairs_dict)', len(pmid_2_rel_pairs_dict))
                #print('len(pmid_2_rel_pairs_dict[pmid])', len(pmid_2_rel_pairs_dict[pmid]))
                
                if pmid in pmid_2_rel_pairs_dict:
                    #print('len(pmid_2_rel_pairs_dict[pmid])', len(pmid_2_rel_pairs_dict[pmid]))
                    for rel in pmid_2_rel_pairs_dict[pmid]:
                        
                        id1 = rel.id1
                        id2 = rel.id2
                        if (pmid_2_index_2_groupID_dict != None) and (rel.id1 in id2index) and (id2index[rel.id1] in pmid_2_index_2_groupID_dict[pmid]):
                            id1 = pmid_2_index_2_groupID_dict[pmid][id2index[rel.id1]][0]
                        if (pmid_2_index_2_groupID_dict != None) and (rel.id2 in id2index) and (id2index[rel.id2] in pmid_2_index_2_groupID_dict[pmid]):
                            id2 = pmid_2_index_2_groupID_dict[pmid][id2index[rel.id2]][0]
                        
                        pred_writer.write(pmid + 
                                  '\t' + rel.rel_type + 
                                  '\t' + id1 + 
                                  '\t' + id2 + 
                                  '\t' + str(rel.score) + '\n')
                    pmid = ''        
                pred_writer.write('\n')
                id2ne_type_dict = {}
                id2index = {}
                        
            else:
                tks = line.split('|')
                if len(tks) > 1 and (tks[1] == 't' or tks[1] == 'a'):
                    #2234245	250	270	audiovisual toxicity	Disease	D014786|D006311
                    pmid = tks[0]
                    
                    pred_writer.write(line + '\n')
                else:
                    tks = line.split('\t')
                    pmid = tks[0]
                    if len(tks) == 6:
                        #print(line)
                        
                        id = tks[5]
                        start = tks[1]
                        end = tks[2]
                        index = start + '|' + end
                        
                        if (pmid_2_index_2_groupID_dict != None) and (pmid in pmid_2_index_2_groupID_dict) and (index in pmid_2_index_2_groupID_dict[pmid]):
                            tks[5] = pmid_2_index_2_groupID_dict[pmid][index][0]
                        
                        line = '\t'.join(tks)
                        pred_writer.write(line + '\n')
                        #id = re.sub('\s*\(.*?\)\s*$', '', tks[5])
                        ne_type = tks[4]
                        
                        if '@' in id:
                            id = id.split('@')[1]
                        
                        '''if ne_type == 'SequenceVariant':
                            ne_type = 'GeneOrGeneProduct'''
                        id2ne_type_dict[id] = ne_type
                        
                        id2index[id] = index
        if pmid != '' and pmid in pmid_2_rel_pairs_dict:
            for rel in pmid_2_rel_pairs_dict[pmid]:
                
                pred_writer.write(pmid + 
                                  '\t' + rel.rel_type + 
                                  '\t' + rel.id1 + 
                                  '\t' + rel.id2 + 
                                  '\t' + rel.score + '\n')
                
    pred_writer.close()
    
def _is_five_considered_ne_pairs(ne1_type, ne2_type):
    
    if ne1_type > ne2_type:
        tmp = ne2_type
        ne2_type = ne1_type
        ne1_type = tmp
    
    #'cc', 'cd', 'cg', 'dg', 'gg'
    if ne1_type == 'ChemicalEntity' and ne2_type == 'ChemicalEntity':
        return True
    elif ne1_type == 'ChemicalEntity' and ne2_type == 'DiseaseOrPhenotypicFeature':
        return True
    elif ne1_type == 'ChemicalEntity' and ne2_type == 'GeneOrGeneProduct':
        return True
    elif ne1_type == 'DiseaseOrPhenotypicFeature' and ne2_type == 'GeneOrGeneProduct':
        return True
    elif ne1_type == 'GeneOrGeneProduct' and ne2_type == 'GeneOrGeneProduct':
        return True
    
    return False

def retrive_original_relation_pairs_dict(in_pubtator_file, 
                                to_binary=False,
                                has_novelty=False,
                                re_id_spliter_str=r',',
                                pmid_2_index_2_groupID_dict=None,
                                only_five_ne_pairs=False,
                                in_homo_id_tsv_file='',
                                only_load_has_id=False,
                                use_original_id_for_eval=False):
    
    pmid_2_rel_pairs_dict = {}
    pmid_2_id2ne_type_dict = {}
    
    rel_pairs_set = {}
    id2ne_type_dict = {}
    
    print(in_pubtator_file)
    
    entrez_id_2_homo_entrez_id = defaultdict()
    if in_homo_id_tsv_file != '':
        homo_id_2_entrez_id = {}
        with open(in_homo_id_tsv_file, 'r', encoding='utf8') as homo_reader:
            homo_reader.readline()
            for line in homo_reader:
                line = line.rstrip()
                tks = line.split('\t')
                entrez_id = tks[0]
                homo_id   = tks[1]
                if homo_id not in homo_id_2_entrez_id:
                    homo_id_2_entrez_id[homo_id] = entrez_id
                entrez_id_2_homo_entrez_id[entrez_id] = homo_id_2_entrez_id[homo_id]
    
    with open(in_pubtator_file, 'r', encoding='utf8') as pub_reader:
        
        pmid = ''
        id2index = {}
        
        for line in pub_reader:
            line = line.rstrip()
            
            if line == '':
                #print('len(rel_pairs_set)', pmid, len(rel_pairs_set))
                pmid_2_rel_pairs_dict[pmid] = rel_pairs_set
                pmid_2_id2ne_type_dict[pmid] = id2ne_type_dict
                rel_pairs_set = {}
                id2ne_type_dict = {}
                id2index = {}
            else:
                tks = line.split('|')
                if len(tks) > 1 and (tks[1] == 't' or tks[1] == 'a'):
                    #2234245	250	270	audiovisual toxicity	Disease	D014786|D006311
                    pmid = tks[0]
                else:
                    tks = line.split('\t')
                    
                    if len(tks) == 5 or len(tks) == 4:
                        rel_type = tks[1]
                        for id1 in re.split(re_id_spliter_str, tks[2]):
                            for id2 in re.split(re_id_spliter_str, tks[3]):
                                #id1      = re.sub('\s*\(.*?\)\s*$', '', tks[2])
                                #id2      = re.sub('\s*\(.*?\)\s*$', '', tks[3])
                                
                                if only_load_has_id and (len(tks) == 5) and (not tks[-1].startswith('HasID')):
                                    continue
                                
                                if '@' in id1:
                                    id1 = id1.split('@')[1]
                                if '@' in id2:
                                    id2 = id2.split('@')[1]
                                #print(line)                             
                                if id1 in entrez_id_2_homo_entrez_id:
                                    id1 = entrez_id_2_homo_entrez_id[id1]
                                if id2 in entrez_id_2_homo_entrez_id:
                                    id2 = entrez_id_2_homo_entrez_id[id2]
                                if (id1 not in id2ne_type_dict) or (id2 not in id2ne_type_dict):
                                    print(line)
                                ne_type1 = id2ne_type_dict[id1]
                                ne_type2 = id2ne_type_dict[id2]
                                
                                if (pmid_2_index_2_groupID_dict != None) and (pmid in pmid_2_index_2_groupID_dict) and (id2index[id1] in pmid_2_index_2_groupID_dict[pmid]):
                                    id1 = pmid_2_index_2_groupID_dict[pmid][id2index[id1]][0]
                                if (pmid_2_index_2_groupID_dict != None) and (pmid in pmid_2_index_2_groupID_dict) and (id2index[id2] in pmid_2_index_2_groupID_dict[pmid]):
                                    id2 = pmid_2_index_2_groupID_dict[pmid][id2index[id2]][0]
                                                                
                                if has_novelty and len(tks) != 4:
                                    novelty  = tks[4]
                                else:
                                    novelty  = 'None'
                                
                                if to_binary:
                                    rel_type = 'Association'
                                    
                                if has_novelty:
                                    rel_type += '|' + novelty
                                #rel_pairs_set.add((id2, id1, rel_type))
                                if ne_type1 == 'DiseaseOrPhenotypicFeature' and ne_type2 == 'DiseaseOrPhenotypicFeature':
                                    continue
                                elif ne_type1 == 'GeneOrGeneProduct' and ne_type2 == 'SequenceVariant':
                                    continue
                                elif ne_type2 == 'GeneOrGeneProduct' and ne_type1 == 'SequenceVariant':
                                    continue
                                elif ne_type2 == 'GeneOrGeneProduct' and ne_type1 == 'OrganismTaxon':
                                    continue
                                elif ne_type2 == 'OrganismTaxon' and ne_type1 == 'GeneOrGeneProduct':
                                    continue
                                
                                if only_five_ne_pairs and (not _is_five_considered_ne_pairs(ne_type1, ne_type2)):
                                    continue
                                if use_original_id_for_eval:
                                    orig_id2 = tks[-1].split('|')[2]
                                    orig_id1 = tks[-1].split('|')[1]
                                    if orig_id2 < orig_id1:
                                        tmp = orig_id1
                                        orig_id1 = orig_id2
                                        orig_id2 = tmp
                                    
                                    if str(id2) < str(id1):
                                        rel_pairs_set[(str(id2), str(id1))] = (orig_id1, orig_id2)
                                    else:
                                        rel_pairs_set[(str(id1), str(id2))] = (orig_id1, orig_id2)
                    else:
                        #id = re.sub('\s*\(.*?\)\s*$', '', tks[5])
                        id = tks[5]
                        index = tks[1] + '|' + tks[2]
                        
                        
                        if '@' in id:
                            id = id.split('@')[1]
                        
                        if id in entrez_id_2_homo_entrez_id:
                            id = entrez_id_2_homo_entrez_id[id]
                            
                        id2index[id] = index
                        
                        ne_type = tks[4]
                        '''if ne_type == 'SequenceVariant':
                            ne_type = 'GeneOrGeneProduct'''
                        for _id in re.split(re_id_spliter_str, id):
                            
                            if _id in entrez_id_2_homo_entrez_id:
                                _id = entrez_id_2_homo_entrez_id[_id]
                            
                            id2ne_type_dict[_id] = ne_type
                            id2index[_id] = index
                        
        if len(rel_pairs_set) != 0:
            pmid_2_rel_pairs_dict[pmid] = rel_pairs_set
            #print('len(rel_pairs_set)', pmid, len(rel_pairs_set))
    
    return pmid_2_rel_pairs_dict
    
def retrive_relation_pairs_dict(in_pubtator_file, 
                                to_binary=False,
                                has_novelty=False,
                                re_id_spliter_str=r',',
                                pmid_2_index_2_groupID_dict=None,
                                only_five_ne_pairs=False,
                                in_homo_id_tsv_file='',
                                only_load_has_id=False,
                                use_original_id_for_eval=False,
                                normalized_type_dict={}):
    
    pmid_2_rel_pairs_dict = {}
    pmid_2_id2ne_type_dict = {}
    
    rel_pairs_set = set()
    id2ne_type_dict = {}
    
    print(in_pubtator_file)
    
    entrez_id_2_homo_entrez_id = defaultdict()
    if in_homo_id_tsv_file != '':
        homo_id_2_entrez_id = {}
        with open(in_homo_id_tsv_file, 'r', encoding='utf8') as homo_reader:
            homo_reader.readline()
            for line in homo_reader:
                line = line.rstrip()
                tks = line.split('\t')
                entrez_id = tks[0]
                homo_id   = tks[1]
                if homo_id not in homo_id_2_entrez_id:
                    homo_id_2_entrez_id[homo_id] = entrez_id
                entrez_id_2_homo_entrez_id[entrez_id] = homo_id_2_entrez_id[homo_id]
    
    with open(in_pubtator_file, 'r', encoding='utf8') as pub_reader:
        
        pmid = ''
        id2index = {}
        
        for line in pub_reader:
            line = line.rstrip()
            
            if line == '':
                #print('len(rel_pairs_set)', pmid, len(rel_pairs_set))
                pmid_2_rel_pairs_dict[pmid] = rel_pairs_set
                pmid_2_id2ne_type_dict[pmid] = id2ne_type_dict
                rel_pairs_set = set()
                id2ne_type_dict = {}
                id2index = {}
            else:
                tks = line.split('|')
                if len(tks) > 1 and (tks[1] == 't' or tks[1] == 'a'):
                    #2234245	250	270	audiovisual toxicity	Disease	D014786|D006311
                    pmid = tks[0]
                else:
                    tks = line.split('\t')
                    
                    if len(tks) == 5 or len(tks) == 4:
                        rel_type = tks[1]
                        for id1 in re.split(re_id_spliter_str, tks[2]):
                            for id2 in re.split(re_id_spliter_str, tks[3]):
                                #id1      = re.sub('\s*\(.*?\)\s*$', '', tks[2])
                                #id2      = re.sub('\s*\(.*?\)\s*$', '', tks[3])
                                
                                if only_load_has_id and (len(tks) == 5) and (not tks[-1].startswith('HasID')):
                                    continue
                                
                                if '@' in id1:
                                    id1 = id1.split('@')[1]
                                if '@' in id2:
                                    id2 = id2.split('@')[1]
                                #print(line)                             
                                if id1 in entrez_id_2_homo_entrez_id:
                                    id1 = entrez_id_2_homo_entrez_id[id1]
                                if id2 in entrez_id_2_homo_entrez_id:
                                    id2 = entrez_id_2_homo_entrez_id[id2]
                                if (id1 not in id2ne_type_dict) or (id2 not in id2ne_type_dict):
                                    print(line)
                                    print(id2ne_type_dict.keys())
                                ne_type1 = id2ne_type_dict[id1]
                                ne_type2 = id2ne_type_dict[id2]
                                
                                if (pmid_2_index_2_groupID_dict != None) and (pmid in pmid_2_index_2_groupID_dict) and (id2index[id1] in pmid_2_index_2_groupID_dict[pmid]):
                                    id1 = pmid_2_index_2_groupID_dict[pmid][id2index[id1]][0]
                                if (pmid_2_index_2_groupID_dict != None) and (pmid in pmid_2_index_2_groupID_dict) and (id2index[id2] in pmid_2_index_2_groupID_dict[pmid]):
                                    id2 = pmid_2_index_2_groupID_dict[pmid][id2index[id2]][0]
                                                                
                                if has_novelty and len(tks) != 4:
                                    novelty  = tks[4]
                                else:
                                    novelty  = 'None'
                                
                                if to_binary:
                                    rel_type = 'Association'
                                    
                                if has_novelty:
                                    rel_type += '|' + novelty
                                #rel_pairs_set.add((id2, id1, rel_type))
                                if ne_type1 == 'DiseaseOrPhenotypicFeature' and ne_type2 == 'DiseaseOrPhenotypicFeature':
                                    continue
                                elif ne_type1 == 'GeneOrGeneProduct' and ne_type2 == 'SequenceVariant':
                                    continue
                                elif ne_type2 == 'GeneOrGeneProduct' and ne_type1 == 'SequenceVariant':
                                    continue
                                elif ne_type2 == 'GeneOrGeneProduct' and ne_type1 == 'OrganismTaxon':
                                    continue
                                elif ne_type2 == 'OrganismTaxon' and ne_type1 == 'GeneOrGeneProduct':
                                    continue
                                
                                if only_five_ne_pairs and (not _is_five_considered_ne_pairs(ne_type1, ne_type2)):
                                    continue
                                if use_original_id_for_eval:
                                    id2 = tks[-1].split('|')[2]
                                    id1 = tks[-1].split('|')[1]
                                if str(id2) > str(id1):
                                    rel_pairs_set.add((id1, id2, rel_type, ne_type1, ne_type2))
                                else:
                                    rel_pairs_set.add((id2, id1, rel_type, ne_type2, ne_type1))
                    else:
                        #id = re.sub('\s*\(.*?\)\s*$', '', tks[5])
                        id = tks[5]
                        index = tks[1] + '|' + tks[2]
                        
                        
                        if '@' in id:
                            id = id.split('@')[1]
                        
                        if id in entrez_id_2_homo_entrez_id:
                            id = entrez_id_2_homo_entrez_id[id]
                            
                        id2index[id] = index
                        
                        ne_type = tks[4]
                        if ne_type in normalized_type_dict:
                            ne_type = normalized_type_dict[ne_type]
                        '''if ne_type == 'SequenceVariant':
                            ne_type = 'GeneOrGeneProduct'''
                        for _id in re.split(re_id_spliter_str, id):
                            
                            if _id in entrez_id_2_homo_entrez_id:
                                _id = entrez_id_2_homo_entrez_id[_id]
                            
                            id2ne_type_dict[_id] = ne_type
                            id2index[_id] = index
                        
        if len(rel_pairs_set) != 0:
            pmid_2_rel_pairs_dict[pmid] = rel_pairs_set
            #print('len(rel_pairs_set)', pmid, len(rel_pairs_set))
    
    return pmid_2_rel_pairs_dict, pmid_2_id2ne_type_dict
     
def eval(in_gold_pubtator_file,
         in_pred_pubtator_file,
         out_error_cases_file,
         in_pmid_file='',
         to_binary=False,
         has_novelty=False,
         eval_tags=set(['Development']),
         eval_pmids=set(),
         re_id_spliter_str=r',',
         pmid_2_index_2_groupID_dict = None,
         only_five_ne_pairs=False,
         in_homo_id_tsv_file='',
         only_load_has_id=False,
         use_original_id_for_eval=False):
                    
    gold_relation_pairs_dict, id2type_dict = retrive_relation_pairs_dict(
            in_gold_pubtator_file,
            to_binary, 
            has_novelty, 
            re_id_spliter_str,
            pmid_2_index_2_groupID_dict,
            only_five_ne_pairs,
            in_homo_id_tsv_file,
            only_load_has_id,
            use_original_id_for_eval)
    
    pred_relation_pairs_dict, _ = retrive_relation_pairs_dict(
            in_pred_pubtator_file, 
            to_binary,
            has_novelty, 
            re_id_spliter_str,
            pmid_2_index_2_groupID_dict,
            only_five_ne_pairs,
            in_homo_id_tsv_file,
            only_load_has_id)
    
    if use_original_id_for_eval:
        original_id_pairs_dict = retrive_original_relation_pairs_dict(
            in_gold_pubtator_file,
            to_binary, 
            has_novelty, 
            re_id_spliter_str,
            pmid_2_index_2_groupID_dict,
            only_five_ne_pairs,
            in_homo_id_tsv_file,
            only_load_has_id,
            use_original_id_for_eval)
        
        for pmid, pred_relation_pairs in pred_relation_pairs_dict.items():
            new_pairs = set()
            pair_id_2_gold_id_dict = original_id_pairs_dict[pmid]
            for pred_relation_pair in pred_relation_pairs:
                # (id1, id2, rel_type, ne_type2, ne_type1)
                pred_id_pair = (pred_relation_pair[0], pred_relation_pair[1])
                if pred_id_pair in pair_id_2_gold_id_dict:
                    gold_id_pair = pair_id_2_gold_id_dict[pred_id_pair]
                    new_pairs.add((gold_id_pair[0], gold_id_pair[1], pred_relation_pair[2], pred_relation_pair[3], pred_relation_pair[4]))
                else:
                    new_pairs.add(pred_relation_pair)
            pred_relation_pairs_dict[pmid] = new_pairs
    
    if in_pmid_file != '':
        with open(in_pmid_file, 'r', encoding='utf8') as pmid_reader:
            for line in pmid_reader:
                tks = line.rstrip().split('\t')
                pmid = tks[0]
                if tks[1] in eval_tags:
                    eval_pmids.add(pmid)        
    
    tp_count = 0.
    fp_count = 0.
    fn_count = 0.
    
    typed_tp_count = defaultdict(float)
    typed_fp_count = defaultdict(float)
    typed_fn_count = defaultdict(float)
    
    
    rel_typed_tp_count = defaultdict(float)
    rel_typed_fp_count = defaultdict(float)
    rel_typed_fn_count = defaultdict(float)
    
    outputs = []
    
    has_eval_set = len(eval_pmids) > 0
    
    for pmid in gold_relation_pairs_dict.keys():
        if has_eval_set and pmid not in eval_pmids:
            continue
        if pmid in pred_relation_pairs_dict:
            pred_relation_pairs = pred_relation_pairs_dict[pmid]
            gold_relation_pairs = gold_relation_pairs_dict[pmid]       
            print('==========>>')
            print('len(pred_relation_pairs)', len(pred_relation_pairs))
            print('len(gold_relation_pairs)', len(gold_relation_pairs))
            for pair in pred_relation_pairs:
                print('pred:', pair[0], pair[1], pair[2])
            for pair in gold_relation_pairs:
                print('gold:', pair[0], pair[1], pair[2])
            for gold_pair in gold_relation_pairs:
                if gold_pair in pred_relation_pairs:
                    tp_count += 1.
                    ne1_type = gold_pair[3]
                    ne2_type = gold_pair[4]
                    rel_type = gold_pair[2]

                    if ne1_type <= ne2_type:
                        rel_typed_tp_count[ne1_type + '|' + ne2_type + '|' + rel_type] += 1.
                        typed_tp_count[(ne1_type, ne2_type)] += 1.
                        outputs.append(ne1_type + '|' + ne2_type + 'TP:\t' + pmid + '\t' + gold_pair[0] + '\t' + gold_pair[1] + '\t' + gold_pair[2])
                    else:
                        rel_typed_tp_count[ne2_type + '|' + ne1_type + '|' + rel_type] += 1.
                        typed_tp_count[(ne2_type, ne1_type)] += 1.
                        outputs.append(ne2_type + '|' + ne1_type + 'TP:\t' + pmid + '\t' + gold_pair[0] + '\t' + gold_pair[1] + '\t' + gold_pair[2])
                else:
                    fn_count += 1.
                    ne1_type = gold_pair[3]
                    ne2_type = gold_pair[4]
                    rel_type = gold_pair[2]
                    
                    if ne1_type <= ne2_type:
                        rel_typed_fn_count[ne1_type + '|' + ne2_type + '|' + rel_type] += 1.
                        typed_fn_count[(ne1_type, ne2_type)] += 1.
                        outputs.append(ne1_type + '|' + ne2_type + 'FN:\t' + pmid + '\t' + gold_pair[0] + '\t' + gold_pair[1] + '\t' + gold_pair[2])
                    else:
                        rel_typed_fn_count[ne2_type + '|' + ne1_type + '|' + rel_type] += 1.
                        typed_fn_count[(ne2_type, ne1_type)] += 1.
                        outputs.append(ne2_type + '|' + ne1_type + 'FN:\t' + pmid + '\t' + gold_pair[0] + '\t' + gold_pair[1] + '\t' + gold_pair[2])
            for pred_pair in pred_relation_pairs:
                if pred_pair not in gold_relation_pairs:
                    fp_count += 1.                    
                    ne1_type = pred_pair[3]
                    ne2_type = pred_pair[4]
                    rel_type = pred_pair[2]
                    
                    if ne1_type <= ne2_type:
                        rel_typed_fp_count[ne1_type + '|' + ne2_type + '|' + rel_type] += 1.
                        typed_fp_count[(ne1_type, ne2_type)] += 1.
                        outputs.append(ne1_type + '|' + ne2_type + 'FP:\t' + pmid + '\t' + pred_pair[0] + '\t' + pred_pair[1] + '\t' + pred_pair[2])
                    else:
                        rel_typed_fp_count[ne2_type + '|' + ne1_type + '|' + rel_type] += 1.
                        typed_fp_count[(ne2_type, ne1_type)] += 1.
                        outputs.append(ne2_type + '|' + ne1_type + 'FP:\t' + pmid + '\t' + pred_pair[0] + '\t' + pred_pair[1] + '\t' + pred_pair[2])
    for pmid in pred_relation_pairs_dict.keys():
        if has_eval_set and pmid not in eval_pmids:
            continue
        if pmid not in gold_relation_pairs_dict:
            pred_relation_pairs = pred_relation_pairs_dict[pmid]
            for pred_pair in pred_relation_pairs:
                fp_count += 1.
                ne1_type = pred_pair[3]
                ne2_type = pred_pair[4]
                rel_type = pred_pair[2]

                if ne1_type <= ne2_type:
                    rel_typed_fp_count[ne1_type + '|' + ne2_type + '|' + rel_type] += 1.
                    typed_fp_count[(ne1_type, ne2_type)] += 1.
                    outputs.append(ne1_type + '|' + ne2_type + 'FP:\t' + pmid + '\t' + pred_pair[0] + '\t' + pred_pair[1] + '\t' + pred_pair[2])
                else:
                    rel_typed_fp_count[ne2_type + '|' + ne1_type + '|' + rel_type] += 1.
                    typed_fp_count[(ne2_type, ne1_type)] += 1.
                    outputs.append(ne2_type + '|' + ne1_type + 'FP:\t' + pmid + '\t' + pred_pair[0] + '\t' + pred_pair[1] + '\t' + pred_pair[2])
    
    all_pair_types = set(typed_fn_count.keys())
    all_pair_types.update(list(typed_tp_count.keys()))
    all_pair_types.update(list(typed_fp_count.keys()))
    all_pair_types = list(all_pair_types)
    all_pair_types.sort()
    
    all_rel_types = set(rel_typed_fn_count.keys())
    all_rel_types.update(list(rel_typed_tp_count.keys()))
    all_rel_types.update(list(rel_typed_fp_count.keys()))
    all_rel_types = list(all_rel_types)
    all_rel_types.sort()
    
    prec = tp_count / (tp_count + fp_count) if (tp_count + fp_count) != 0 else 0.
    reca = tp_count / (tp_count + fn_count) if (tp_count + fn_count) != 0 else 0.
    fsco = (2*prec*reca) / (prec + reca) if (prec + reca) != 0 else 0.
    
    with open(out_error_cases_file, 'w', encoding='utf8') as score_writer:
        score_writer.write('Overall')
        score_writer.write('\t' + str(int(tp_count + fn_count)))
        score_writer.write('\t' + str(int(tp_count)))
        score_writer.write('\t' + str(int(fp_count)))
        score_writer.write('\t' + str(int(fn_count)))
        score_writer.write('\t' + str(prec))
        score_writer.write('\t' + str(reca))
        score_writer.write('\t' + str(fsco) + '\n')
        score_writer.write('\n')
        for rel_type in all_rel_types:
            tp = int(rel_typed_tp_count[rel_type])
            fp = int(rel_typed_fp_count[rel_type])
            fn = int(rel_typed_fn_count[rel_type])
            prec = float(tp) / (float(tp) + float(fp)) if (float(tp) + float(fp)) > 0 else 0.
            reca = float(tp) / (float(tp) + float(fn)) if (float(tp) + float(fn)) > 0 else 0.
            fsco = (2*prec*reca) / (prec + reca) if (prec + reca) > 0 else 0.
            score_writer.write(rel_type)
            score_writer.write('\t' + str(tp + fn))
            score_writer.write('\t' + str(tp))
            score_writer.write('\t' + str(fp))
            score_writer.write('\t' + str(fn))
            score_writer.write('\t' + str(prec))
            score_writer.write('\t' + str(reca))
            score_writer.write('\t' + str(fsco) + '\n')
        score_writer.write('\n')
        for pair_types in all_pair_types:
            tp = int(typed_tp_count[pair_types])
            fp = int(typed_fp_count[pair_types])
            fn = int(typed_fn_count[pair_types])
            prec = float(tp) / (float(tp) + float(fp)) if (float(tp) + float(fp)) > 0 else 0.
            reca = float(tp) / (float(tp) + float(fn)) if (float(tp) + float(fn)) > 0 else 0.
            fsco = (2*prec*reca) / (prec + reca) if (prec + reca) > 0 else 0.
            score_writer.write(pair_types[0] + '|' + pair_types[1])
            score_writer.write('\t' + str(tp + fn))
            score_writer.write('\t' + str(tp))
            score_writer.write('\t' + str(fp))
            score_writer.write('\t' + str(fn))
            score_writer.write('\t' + str(prec))
            score_writer.write('\t' + str(reca))
            score_writer.write('\t' + str(fsco) + '\n')
        score_writer.write('\n')
        for output in outputs:
            score_writer.write(output + '\n')
            
    return tp_count, fp_count, fn_count, prec, reca, fsco

           
def _load_pmid_2_ne_type_id_dict(
        in_file,
        in_dir,
        in_format,
        re_id_spliter_str,
        in_homo_id_tsv_file=''):
    
    normalized_type_dict = {'Disease': 'DiseaseOrPhenotypicFeature',
                               'Gene': 'GeneOrGeneProduct',
                               'Chemical': 'ChemicalEntity'} # for pubtator3 biored test set eval
    

    def _load_pmid_2_ne_type_id(
            in_pubtator_file,
            re_id_spliter_str,
            in_homo_id_tsv_file):
        
        pmid_2_ne_type_id_dict = defaultdict(set)
        
        entrez_id_2_homo_entrez_id = defaultdict()
        if in_homo_id_tsv_file != '':
            homo_id_2_entrez_id = {}
            with open(in_homo_id_tsv_file, 'r', encoding='utf8') as homo_reader:
                homo_reader.readline()
                for line in homo_reader:
                    line = line.rstrip()
                    tks = line.split('\t')
                    entrez_id = tks[0]
                    homo_id   = tks[1]
                    if homo_id not in homo_id_2_entrez_id:
                        homo_id_2_entrez_id[homo_id] = entrez_id
                    entrez_id_2_homo_entrez_id[entrez_id] = homo_id_2_entrez_id[homo_id]
        
        with open(in_pubtator_file, 'r', encoding='utf8') as reader:
            
            for line in reader:
                line = line.rstrip()
                tks = line.split('\t')
                
                # if is NE
                if len(tks) == 6:
                    pmid = tks[0]
                    ne_type = tks[4]
                    if pmid not in pmid_2_ne_type_id_dict:
                        pmid_2_ne_type_id_dict[pmid] = set()
                    for id in re.split(re_id_spliter_str, tks[-1]):
                        if id == '-':
                            continue
                        if id in entrez_id_2_homo_entrez_id:
                            id = entrez_id_2_homo_entrez_id[id]
                        pmid_2_ne_type_id_dict[pmid].add((ne_type, id))
          
        return pmid_2_ne_type_id_dict
    
    
    pmid_2_ne_type_id_dict = {}
    if in_format == 'pubtator':
        if in_file != '':
            pmid_2_ne_type_id_dict = _load_pmid_2_ne_type_id(
                    in_pubtator_file      = in_file,
                    re_id_spliter_str     = re_id_spliter_str,
                    in_homo_id_tsv_file   = in_homo_id_tsv_file)
        else:
            pmid_2_ne_type_id_dict = {}
            
            for _in_file in os.listdir(in_dir):
                _pmid_2_id2ne_type_dict = _load_pmid_2_ne_type_id(
                    in_pubtator_file         = in_dir + '/' + _in_file,
                    re_id_spliter_str        = re_id_spliter_str,
                    in_homo_id_tsv_file      = in_homo_id_tsv_file)
                pmid_2_ne_type_id_dict.update(_pmid_2_id2ne_type_dict)
    return pmid_2_ne_type_id_dict        

def run_simple_tsv_eval(
        in_gold_tsv_file,
        in_pred_tsv_file,
        out_bin_result_file,
        labels,
        label = 'label',
        biored_eval_filter = '',
        label_index = -1):
    
    #print('in_gold_tsv_file', in_gold_tsv_file)
    #print('in_pred_tsv_file', in_pred_tsv_file)
    testdf = pd.read_csv(in_gold_tsv_file, sep="\t", index_col=0)
    try:
        preddf = pd.read_csv(in_pred_tsv_file, sep="\t", header=None)
    except:
        return
    
    pred = [preddf.iloc[i].tolist() for i in preddf.index]
    pred_class = [np.argmax(v) for v in pred]
    
    
    test_labels = None
    test_is_in_same_sents = None
    
    try:
        test_labels           = testdf[label]
        index_list            = testdf["pmid"]
        id1_list              = testdf["id1"]
        id2_list              = testdf["id2"]        
        src_list              = testdf["ne_type1"]
        tgt_list              = testdf["ne_type2"]
    except:
        testdf = pd.read_csv(in_gold_tsv_file, sep="\t", header=None)
        test_labels           = testdf.iloc[:,label_index]
        index_list            = testdf.iloc[:,0]
        id1_list              = testdf.iloc[:,3]
        id2_list              = testdf.iloc[:,4]
        
        src_list              = testdf.iloc[:,1]
        tgt_list              = testdf.iloc[:,2]
    
    eval_pairs = set()
    
    if biored_eval_filter != '':
        for _biored_eval_filter in biored_eval_filter.split(';'):
            src_type = _biored_eval_filter.split('|')[0]
            tgt_type = _biored_eval_filter.split('|')[1]
            eval_pairs.add((src_type, tgt_type))
        
    outputs = []
    tp = 0.
    fp = 0.
    fn = 0.
    
    tp_by_pair_dict = defaultdict(int)
    fp_by_pair_dict = defaultdict(int)
    fn_by_pair_dict = defaultdict(int)
    all_pairs = set()
    
    for _pred, _gold, index, id1, id2, _src_type, _tgt_type in zip(pred_class, test_labels, index_list, id1_list, id2_list, src_list, tgt_list):
        pred_label = labels[_pred]
        sindex = str(index)
        
        pair = (_src_type, _tgt_type)
        
        if (biored_eval_filter != ""):
            if pair not in eval_pairs:
                continue
        all_pairs.add(pair)
        
        if not pred_label.startswith('None'):
            pred_label = 'Association'
        if not _gold.startswith('None'):
            _gold = 'Association'
        
        if pred_label == _gold:
            if not pred_label.startswith('None'):
                tp += 1
                tp_by_pair_dict[pair] += 1
                outputs.append('TP\t' + sindex + '\t' + str(id1) + '\t' + str(id2) + '\t' + pred_label)
        else:
            if not pred_label.startswith('None'):
                if not _gold.startswith('None'):
                    fp += 1
                    fn += 1
                    fp_by_pair_dict[pair] += 1
                    fn_by_pair_dict[pair] += 1
                    outputs.append('FP\t' + sindex + '\t' + str(id1) + '\t' + str(id2) + '\t' + pred_label)
                    outputs.append('FN\t' + sindex + '\t' + str(id1) + '\t' + str(id2) + '\t' + _gold)
                else:
                    fp += 1
                    fp_by_pair_dict[pair] += 1
                    outputs.append('FP\t' + sindex + '\t' + str(id1) + '\t' + str(id2) + '\t' + pred_label)
            else:
                if not _gold.startswith('None'):
                    fn += 1
                    fn_by_pair_dict[pair] += 1
                    outputs.append('FN\t' + sindex + '\t' + str(id1) + '\t' + str(id2) + '\t' + _gold)
            #print(index, pred_label)
            
    all_pairs = list(all_pairs)
    all_pairs.sort()
    
    with open(out_bin_result_file, 'w', encoding='utf8') as writer:
        
        prec = tp / (tp+fp) if (tp+fp) > 0 else 0.
        recall = tp/ (tp+fn) if (tp+fn) > 0 else 0.
        f = (2*prec*recall) / (prec+recall) if (prec+recall) > 0 else 0.
        writer.write("TP: \t" + str(int(tp)) + '\n')
        writer.write("FP: \t" + str(int(fp)) + '\n')
        writer.write("FN: \t" + str(int(fn)) + '\n\n')
        writer.write("Precision: \t" + str(prec) + '\n')
        writer.write("Recall: \t" + str(recall) + '\n')
        writer.write("F-score: \t" + str(f) + '\n\n')
        
        for pair in all_pairs:
            _tp = tp_by_pair_dict[pair]
            _fp = fp_by_pair_dict[pair]
            _fn = fn_by_pair_dict[pair]
            prec = _tp / (_tp+_fp) if (_tp+_fp) > 0 else 0.
            recall = _tp/ (_tp+_fn) if (_tp+_fn) > 0 else 0.
            f = (2*prec*recall) / (prec+recall) if (prec+recall) > 0 else 0.
            writer.write(str(pair) + '\n')
            writer.write("TP: \t" + str(int(_tp)) + '\n')
            writer.write("FP: \t" + str(int(_fp)) + '\n')
            writer.write("FN: \t" + str(int(_fn)) + '\n\n')
            writer.write("Precision: \t" + str(prec) + '\n')
            writer.write("Recall: \t" + str(recall) + '\n')
            writer.write("F-score: \t" + str(f) + '\n\n')
        
        for output in outputs:
            writer.write(output + '\n')
            
        
def run_tsv_eval(
        in_gold_pubtator_file,
        in_gold_tsv_file,
        in_pred_tsv_file,
        out_pred_pubtator_file,
        out_result_file,
        out_bin_result_file,
        labels,
        re_id_spliter_str,
        only_five_ne_pairs=False):
    
    dump_pred_2_pubtator_file(in_pubtator_file       = in_gold_pubtator_file, 
                              in_gold_tsv_file       = in_gold_tsv_file,
                              in_pred_tsv_file       = in_pred_tsv_file,
                              out_pubtator_pred_file = out_pred_pubtator_file,
                              labels                 = labels)
    
    eval(in_gold_pubtator_file, 
         out_pred_pubtator_file, 
         out_result_file,
         to_binary = False,
         re_id_spliter_str = re_id_spliter_str,
         only_five_ne_pairs = only_five_ne_pairs)
    
    eval(in_gold_pubtator_file, 
         out_pred_pubtator_file, 
         out_bin_result_file,
         to_binary = True,
         re_id_spliter_str = re_id_spliter_str,
         only_five_ne_pairs = only_five_ne_pairs)
    
def _eval(
        gold_relation_pairs_dict,
        pred_relation_pairs_dict,
        out_score_file,
        ignore_pred_pmids_not_in_gold=False,
        in_homo_id_tsv_file = ''):
    
    tp_count = 0.
    fp_count = 0.
    fn_count = 0.
    
    typed_tp_count = defaultdict(float)
    typed_fp_count = defaultdict(float)
    typed_fn_count = defaultdict(float)
    
    
    rel_typed_tp_count = defaultdict(float)
    rel_typed_fp_count = defaultdict(float)
    rel_typed_fn_count = defaultdict(float)
    
    outputs = []
    eval_pmids = set()
    
    for pmid in gold_relation_pairs_dict.keys():
        if pmid in pred_relation_pairs_dict:
            pred_relation_pairs = pred_relation_pairs_dict[pmid]
            gold_relation_pairs = gold_relation_pairs_dict[pmid]       
            print('==========>>', pmid)
            print('len(pred_relation_pairs)', len(pred_relation_pairs))
            print('len(gold_relation_pairs)', len(gold_relation_pairs))
            for pair in pred_relation_pairs:
                print('pred:', pair[0], pair[1], pair[2])
            for pair in gold_relation_pairs:
                print('gold:', pair[0], pair[1], pair[2])
            for gold_pair in gold_relation_pairs:
                if gold_pair in pred_relation_pairs:
                    tp_count += 1.
                    ne1_type = gold_pair[3]
                    ne2_type = gold_pair[4]
                    rel_type = gold_pair[2]

                    if ne1_type <= ne2_type:
                        rel_typed_tp_count[ne1_type + '|' + ne2_type + '|' + rel_type] += 1.
                        typed_tp_count[(ne1_type, ne2_type)] += 1.
                        outputs.append(ne1_type + '|' + ne2_type + 'TP:\t' + pmid + '\t' + gold_pair[0] + '\t' + gold_pair[1] + '\t' + gold_pair[2])
                    else:
                        rel_typed_tp_count[ne2_type + '|' + ne1_type + '|' + rel_type] += 1.
                        typed_tp_count[(ne2_type, ne1_type)] += 1.
                        outputs.append(ne2_type + '|' + ne1_type + 'TP:\t' + pmid + '\t' + gold_pair[0] + '\t' + gold_pair[1] + '\t' + gold_pair[2])
                else:
                    fn_count += 1.
                    ne1_type = gold_pair[3]
                    ne2_type = gold_pair[4]
                    rel_type = gold_pair[2]
                    
                    if ne1_type <= ne2_type:
                        rel_typed_fn_count[ne1_type + '|' + ne2_type + '|' + rel_type] += 1.
                        typed_fn_count[(ne1_type, ne2_type)] += 1.
                        outputs.append(ne1_type + '|' + ne2_type + 'FN:\t' + pmid + '\t' + gold_pair[0] + '\t' + gold_pair[1] + '\t' + gold_pair[2])
                    else:
                        rel_typed_fn_count[ne2_type + '|' + ne1_type + '|' + rel_type] += 1.
                        typed_fn_count[(ne2_type, ne1_type)] += 1.
                        outputs.append(ne2_type + '|' + ne1_type + 'FN:\t' + pmid + '\t' + gold_pair[0] + '\t' + gold_pair[1] + '\t' + gold_pair[2])
            for pred_pair in pred_relation_pairs:
                if pred_pair not in gold_relation_pairs:
                    fp_count += 1.                    
                    ne1_type = pred_pair[3]
                    ne2_type = pred_pair[4]
                    rel_type = pred_pair[2]
                    
                    if ne1_type <= ne2_type:
                        rel_typed_fp_count[ne1_type + '|' + ne2_type + '|' + rel_type] += 1.
                        typed_fp_count[(ne1_type, ne2_type)] += 1.
                        outputs.append(ne1_type + '|' + ne2_type + 'FP:\t' + pmid + '\t' + pred_pair[0] + '\t' + pred_pair[1] + '\t' + pred_pair[2])
                    else:
                        rel_typed_fp_count[ne2_type + '|' + ne1_type + '|' + rel_type] += 1.
                        typed_fp_count[(ne2_type, ne1_type)] += 1.
                        outputs.append(ne2_type + '|' + ne1_type + 'FP:\t' + pmid + '\t' + pred_pair[0] + '\t' + pred_pair[1] + '\t' + pred_pair[2])
                        
    if not ignore_pred_pmids_not_in_gold:
        for pmid in pred_relation_pairs_dict.keys():
            if pmid not in gold_relation_pairs_dict:
                pred_relation_pairs = pred_relation_pairs_dict[pmid]
                for pred_pair in pred_relation_pairs:
                    fp_count += 1.
                    ne1_type = pred_pair[3]
                    ne2_type = pred_pair[4]
                    rel_type = pred_pair[2]
    
                    if ne1_type <= ne2_type:
                        rel_typed_fp_count[ne1_type + '|' + ne2_type + '|' + rel_type] += 1.
                        typed_fp_count[(ne1_type, ne2_type)] += 1.
                        outputs.append(ne1_type + '|' + ne2_type + 'FP:\t' + pmid + '\t' + pred_pair[0] + '\t' + pred_pair[1] + '\t' + pred_pair[2])
                    else:
                        rel_typed_fp_count[ne2_type + '|' + ne1_type + '|' + rel_type] += 1.
                        typed_fp_count[(ne2_type, ne1_type)] += 1.
                        outputs.append(ne2_type + '|' + ne1_type + 'FP:\t' + pmid + '\t' + pred_pair[0] + '\t' + pred_pair[1] + '\t' + pred_pair[2])
    
    all_pair_types = set(typed_fn_count.keys())
    all_pair_types.update(list(typed_tp_count.keys()))
    all_pair_types.update(list(typed_fp_count.keys()))
    all_pair_types = list(all_pair_types)
    all_pair_types.sort()
    
    all_rel_types = set(rel_typed_fn_count.keys())
    all_rel_types.update(list(rel_typed_tp_count.keys()))
    all_rel_types.update(list(rel_typed_fp_count.keys()))
    all_rel_types = list(all_rel_types)
    all_rel_types.sort()
    
    prec = tp_count / (tp_count + fp_count) if (tp_count + fp_count) != 0 else 0.
    reca = tp_count / (tp_count + fn_count) if (tp_count + fn_count) != 0 else 0.
    fsco = (2*prec*reca) / (prec + reca) if (prec + reca) != 0 else 0.
    
    with open(out_score_file, 'w', encoding='utf8') as score_writer:
        score_writer.write('Overall')
        score_writer.write('\t' + str(int(tp_count + fn_count)))
        score_writer.write('\t' + str(int(tp_count)))
        score_writer.write('\t' + str(int(fp_count)))
        score_writer.write('\t' + str(int(fn_count)))
        score_writer.write('\t' + str(prec))
        score_writer.write('\t' + str(reca))
        score_writer.write('\t' + str(fsco) + '\n')
        score_writer.write('\n')
        for rel_type in all_rel_types:
            tp = int(rel_typed_tp_count[rel_type])
            fp = int(rel_typed_fp_count[rel_type])
            fn = int(rel_typed_fn_count[rel_type])
            prec = float(tp) / (float(tp) + float(fp)) if (float(tp) + float(fp)) > 0 else 0.
            reca = float(tp) / (float(tp) + float(fn)) if (float(tp) + float(fn)) > 0 else 0.
            fsco = (2*prec*reca) / (prec + reca) if (prec + reca) > 0 else 0.
            score_writer.write(rel_type)
            score_writer.write('\t' + str(tp + fn))
            score_writer.write('\t' + str(tp))
            score_writer.write('\t' + str(fp))
            score_writer.write('\t' + str(fn))
            score_writer.write('\t' + str(prec))
            score_writer.write('\t' + str(reca))
            score_writer.write('\t' + str(fsco) + '\n')
        score_writer.write('\n')
        for pair_types in all_pair_types:
            tp = int(typed_tp_count[pair_types])
            fp = int(typed_fp_count[pair_types])
            fn = int(typed_fn_count[pair_types])
            prec = float(tp) / (float(tp) + float(fp)) if (float(tp) + float(fp)) > 0 else 0.
            reca = float(tp) / (float(tp) + float(fn)) if (float(tp) + float(fn)) > 0 else 0.
            fsco = (2*prec*reca) / (prec + reca) if (prec + reca) > 0 else 0.
            score_writer.write(pair_types[0] + '|' + pair_types[1])
            score_writer.write('\t' + str(tp + fn))
            score_writer.write('\t' + str(tp))
            score_writer.write('\t' + str(fp))
            score_writer.write('\t' + str(fn))
            score_writer.write('\t' + str(prec))
            score_writer.write('\t' + str(reca))
            score_writer.write('\t' + str(fsco) + '\n')
        score_writer.write('\n')
        for output in outputs:
            score_writer.write(output + '\n')
            
    return tp_count, fp_count, fn_count, prec, reca, fsco
    
def _load_relation_pairs_dict(
        in_file,
        in_dir,
        in_format,
        to_binary,
        re_id_spliter_str,
        in_homo_id_tsv_file=''):
    
    relation_pairs_dict = {}
    
    normalized_type_dict = {'Disease': 'DiseaseOrPhenotypicFeature',
                               'Gene': 'GeneOrGeneProduct',
                               'Chemical': 'ChemicalEntity'} # for pubtator3 biored test set eval
    
    if in_format == 'pubtator':
        if in_file != '':
            relation_pairs_dict, _ = retrive_relation_pairs_dict(
                    in_pubtator_file      = in_file,
                    re_id_spliter_str     = re_id_spliter_str,
                    to_binary = to_binary,
                    in_homo_id_tsv_file   = in_homo_id_tsv_file,
                    normalized_type_dict  = normalized_type_dict)
        else:
            relation_pairs_dict = {}
            
            for _in_file in os.listdir(in_dir):
                _relation_pairs_dict, _ = retrive_relation_pairs_dict(
                    in_pubtator_file         = in_dir + '/' + _in_file,
                    re_id_spliter_str        = re_id_spliter_str,
                    to_binary                = to_binary,
                    in_homo_id_tsv_file      = in_homo_id_tsv_file,
                    normalized_type_dict  = normalized_type_dict)
                relation_pairs_dict.update(_relation_pairs_dict)
    return relation_pairs_dict

        
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
        
def get_tp_fp_fn_dict(
        in_gold_pubtator_file, 
        in_pred_pubtator_file):
    
    pmid_2_tp_dict = defaultdict(lambda: list())
    pmid_2_fp_dict = defaultdict(lambda: list())
    pmid_2_fn_dict = defaultdict(lambda: list())
    
    
    gold_relation_pairs_dict, id2type_dict = retrive_relation_pairs_dict(
            in_pubtator_file            = in_gold_pubtator_file, 
            to_binary                   = True,
            has_novelty                 = False,
            re_id_spliter_str           = r';',
            pmid_2_index_2_groupID_dict = None,
            only_five_ne_pairs          = True)
    pred_relation_pairs_dict, _ = retrive_relation_pairs_dict(
            in_pubtator_file            = in_pred_pubtator_file, 
            to_binary                   = True,
            has_novelty                 = False,
            re_id_spliter_str           = r';',
            pmid_2_index_2_groupID_dict = None,
            only_five_ne_pairs          = True)
    
    tp_count = 0.
    fp_count = 0.
    fn_count = 0.
    
    typed_tp_count = defaultdict(float)
    typed_fp_count = defaultdict(float)
    typed_fn_count = defaultdict(float)
    
    
    rel_typed_tp_count = defaultdict(float)
    rel_typed_fp_count = defaultdict(float)
    rel_typed_fn_count = defaultdict(float)
    
    outputs = []    
    
    for pmid in gold_relation_pairs_dict.keys():
        if pmid in pred_relation_pairs_dict:
            pred_relation_pairs = pred_relation_pairs_dict[pmid]
            gold_relation_pairs = gold_relation_pairs_dict[pmid]
            for gold_pair in gold_relation_pairs:
                if gold_pair in pred_relation_pairs:
                    tp_count += 1.
                    ne1_type = gold_pair[3]
                    ne2_type = gold_pair[4]
                    rel_type = gold_pair[2]

                    if ne1_type <= ne2_type:
                        pmid_2_tp_dict[pmid].append(ne1_type + '\t' + ne2_type + '\t' + gold_pair[0] + '\t' + gold_pair[1] + '\t' + gold_pair[2])
                    else:
                        pmid_2_tp_dict[pmid].append(ne2_type + '\t' + ne1_type + '\t' + gold_pair[1] + '\t' + gold_pair[0] + '\t' + gold_pair[2])
                else:
                    fn_count += 1.
                    ne1_type = gold_pair[3]
                    ne2_type = gold_pair[4]
                    rel_type = gold_pair[2]
                    
                    if ne1_type <= ne2_type:
                        pmid_2_fn_dict[pmid].append(ne1_type + '\t' + ne2_type + '\t' + gold_pair[0] + '\t' + gold_pair[1] + '\t' + gold_pair[2])
                    else:
                        pmid_2_fn_dict[pmid].append(ne2_type + '\t' + ne1_type + '\t' + gold_pair[1] + '\t' + gold_pair[0] + '\t' + gold_pair[2])
            for pred_pair in pred_relation_pairs:
                if pred_pair not in gold_relation_pairs:
                    fp_count += 1.                    
                    ne1_type = pred_pair[3]
                    ne2_type = pred_pair[4]
                    rel_type = pred_pair[2]
                    
                    if ne1_type <= ne2_type:
                        pmid_2_fp_dict[pmid].append(ne1_type + '\t' + ne2_type + '\t' + pred_pair[0] + '\t' + pred_pair[1] + '\t' + pred_pair[2])
                    else:
                        pmid_2_fp_dict[pmid].append(ne2_type + '\t' + ne1_type + '\t' + pred_pair[1] + '\t' + pred_pair[0] + '\t' + pred_pair[2])
    for pmid in pred_relation_pairs_dict.keys():
        if pmid not in gold_relation_pairs_dict:
            pred_relation_pairs = pred_relation_pairs_dict[pmid]
            for pred_pair in pred_relation_pairs:
                fp_count += 1.
                ne1_type = pred_pair[3]
                ne2_type = pred_pair[4]
                rel_type = pred_pair[2]

                if ne1_type <= ne2_type:
                    pmid_2_fp_dict[pmid].append(ne1_type + '\t' + ne2_type + '\t' + pred_pair[0] + '\t' + pred_pair[1] + '\t' + pred_pair[2])
                else:
                    pmid_2_fp_dict[pmid].append(ne2_type + '\t' + ne1_type + '\t' + pred_pair[1] + '\t' + pred_pair[0] + '\t' + pred_pair[2])

    return pmid_2_tp_dict, pmid_2_fp_dict, pmid_2_fn_dict

    
if __name__ == '__main__':
    
    options, args = parser.parse_args()
    
    exp_option    = options.exp_option
    
    labels =   ['None',
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
            
    if exp_option == 'biored_eval':
        # litcoin + only evaluate five different relation pairs
        in_gold_pubtator_file    = 'datasets/ncbi_relation/Test.PubTator'
        in_gold_tsv_file         = 'datasets/biorex/processed/test.tsv'
        in_pred_tsv_file         = options.in_pred_tsv_file
        out_pred_pubtator_file   = options.out_pred_pubtator_file
        out_result_file          = options.out_result_file
        out_bin_result_file      = options.out_bin_result_file
        
        run_tsv_eval(
            in_gold_pubtator_file  = in_gold_pubtator_file,
            in_gold_tsv_file       = in_gold_tsv_file,
            in_pred_tsv_file       = in_pred_tsv_file,
            out_pred_pubtator_file = out_pred_pubtator_file,
            out_result_file        = out_result_file,
            out_bin_result_file    = out_bin_result_file,
            labels                 = labels,
            re_id_spliter_str      = r'\;',
            only_five_ne_pairs     = True)
        
    elif exp_option == 'simple_tsv_eval':
        # litcoin + only evaluate five different relation pairs
        in_gold_tsv_file         = options.in_gold_tsv_file
        in_pred_tsv_file         = options.in_pred_tsv_file
        out_bin_result_file      = options.out_bin_result_file
        biored_eval_filter       = options.biored_eval_filter
        run_simple_tsv_eval(
            in_gold_tsv_file       = in_gold_tsv_file,
            in_pred_tsv_file       = in_pred_tsv_file,
            out_bin_result_file    = out_bin_result_file,
            labels                 = labels,
            biored_eval_filter     = biored_eval_filter)