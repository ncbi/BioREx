# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 15:41:27 2021

@author: laip2
"""


from annotation import AnnotationInfo
from document import PubtatorDocument, TextInstance

import utils
import random
import os
import re

def __load_abs_from_tsv(in_abs_tsv_file):
    
    all_abstracts_dict = {}
    
    with open(in_abs_tsv_file, 'r', encoding='utf8') as tsv_reader:
        
        for line in tsv_reader:
            line = line.rstrip()
            tks = line.split('\t')
            pmid              = tks[0]
            title_text        = re.sub(r'(\u2005|\u2002|\u00a0|\u1680|\u180e|\u2000|\u2001|\u2003|\u2004|\u2006|\u2007|\u2008|\u2009|\u200a|\u200b|\u202f|\u205f|\u3000|\ufeff)', ' ', tks[1])
            abstract_text     = re.sub(r'(\u2005|\u2002|\u00a0|\u1680|\u180e|\u2000|\u2001|\u2003|\u2004|\u2006|\u2007|\u2008|\u2009|\u200a|\u200b|\u202f|\u205f|\u3000|\ufeff)', ' ', tks[2])
            _abstract = TextInstance(title_text + ' ' + abstract_text)
            all_abstracts_dict[pmid] = PubtatorDocument(pmid)
            all_abstracts_dict[pmid].text_instances.append(_abstract)
    
    return all_abstracts_dict

def __append_ann_into_abstracts_dict(
        all_abstracts_dict, 
        in_ann_tsv_file,
        normalized_type_dict = None):
    
    with open(in_ann_tsv_file, 'r', encoding='utf8') as tsv_reader:
        
        for line in tsv_reader:
            line = line.rstrip()
            #11319232	T12	GENE	1244	1248	ACS1
            tks = line.split('\t')
            pmid        = tks[0]
            theme_id    = tks[1]
            ne_type     = tks[2]
            position    = int(tks[3])
            length      = int(tks[4]) - int(tks[3])
            text        = tks[5]
            if normalized_type_dict != None and (ne_type in normalized_type_dict):
                ne_type = normalized_type_dict[ne_type]
            if ne_type.startswith('GENE'):
                ne_type = 'GENE'
            ainfo       = AnnotationInfo(position,
                                         length,
                                         text,
                                         ne_type)
            ainfo.ids.add(theme_id)
            all_abstracts_dict[pmid].text_instances[0].annotations.append(ainfo)
            
def __append_rel_into_abstracts_dict(
        all_abstracts_dict, 
        in_rel_tsv_file):
    
    with open(in_rel_tsv_file, 'r', encoding='utf8') as tsv_reader:
        
        for line in tsv_reader:
            line = line.rstrip()
            #23568856	SUBSTRATE	Arg1:T7	Arg2:T21
            tks = line.split('\t')
            pmid        = tks[0]
            rel_type    = tks[1]
            arg1        = tks[2].split(':')[1]
            arg2        = tks[3].split(':')[1]
            
            rel_pair     = (arg1, arg2)
            if all_abstracts_dict[pmid].relation_pairs == None:
                all_abstracts_dict[pmid].relation_pairs = {}
            all_abstracts_dict[pmid].relation_pairs[rel_pair] = rel_type

def load_drugprot_into_document_dict(
        in_abs_tsv_file,
        in_ann_tsv_file,
        spacy_model,
        in_rel_tsv_file = '',
        normalized_type_dict = None):
    
    all_abstracts_dict = __load_abs_from_tsv(in_abs_tsv_file)
    __append_ann_into_abstracts_dict(all_abstracts_dict, in_ann_tsv_file, normalized_type_dict)
    if in_rel_tsv_file != '':
        __append_rel_into_abstracts_dict(all_abstracts_dict, in_rel_tsv_file)
        
    all_documents = list(all_abstracts_dict.values())
    utils.tokenize_documents_by_spacy(all_documents, spacy_model)
    
    return all_abstracts_dict
    
def load_drugprot_into_document_list(
        in_abs_tsv_file,
        in_ann_tsv_file,
        spacy_model,
        in_rel_tsv_file = '',
        normalized_type_dict = None):
    
    all_abstracts_dict = __load_abs_from_tsv(in_abs_tsv_file)
    __append_ann_into_abstracts_dict(all_abstracts_dict, in_ann_tsv_file, normalized_type_dict)
    if in_rel_tsv_file != '':
        __append_rel_into_abstracts_dict(all_abstracts_dict, in_rel_tsv_file)
    
    all_documents = list(all_abstracts_dict.values())
    utils.tokenize_documents_by_spacy(all_documents, spacy_model)
    
    return all_documents
