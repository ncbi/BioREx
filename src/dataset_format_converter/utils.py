# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 14:09:51 2021

@author: laip2
"""
import re
import spacy
import scispacy
import json
from collections import defaultdict
from document import PubtatorDocument, TextInstance
from annotation import AnnotationInfo
import sys
import logging
from sentence_spliter import regex_sentence_boundary_gen
        
logger = logging.getLogger(__name__)

def _spacy_split_sentence(text, nlp):
    offset = 0
    offsets = []
    doc = nlp(text)
    
    do_not_split = False
    start = 0
    end = 0
    for sent in doc.sents:
        if re.search(r'\b[a-z]\.$|[A-Z] ?\>$|[^a-z]del\.$| viz\.$', sent.text):
            if not do_not_split:
                start = offset
            end = offset + len(sent.text)
            offset = end
            for c in text[end:]:
                if c == ' ':
                    offset += 1
                else:
                    break
            do_not_split = True
        else:
            if do_not_split:                
                do_not_split = False
                end = offset + len(sent.text)
                offset = end
                for c in text[end:]:
                    if c == ' ':
                        offset += 1
                    else:
                        break
                offsets.append((start, end))
            else:
                start = offset
                end = offset + len(sent.text)
                offsets.append((start, end))
                
                offset = end
                for c in text[end:]:
                    if c == ' ':
                        offset += 1
                    else:
                        break
        
    if do_not_split:
        offsets.append((start, end))
    return offsets

def split_sentence(document, nlp = None):
    new_text_instances = []
    for text_instance in document.text_instances:
        
        #offsets = [o for o in _nltk_split_sentence(text_instance.text)]
        if nlp == None:
            offsets = [o for o in regex_sentence_boundary_gen(text_instance.text)]
        else:
            offsets = [o for o in _spacy_split_sentence(text_instance.text, nlp)]
            
        _tmp_text_instances = []
        for start, end in offsets:
            new_text_instance = TextInstance(text_instance.text[start:end])
            new_text_instance.offset = start
            new_text_instance.section = text_instance.section
            _tmp_text_instances.append(new_text_instance)
        for annotation in text_instance.annotations:
            is_entity_splited = True
            for _tmp_text_instance in _tmp_text_instances:
                if _tmp_text_instance.offset <= annotation.position and \
                    (annotation.position + annotation.length) - _tmp_text_instance.offset <= len(_tmp_text_instance.text):
                    annotation.position = annotation.position - _tmp_text_instance.offset
                    _tmp_text_instance.annotations.append(annotation)
                    is_entity_splited = False
                    break
            if is_entity_splited:
                print(annotation.position, annotation.length, annotation.text)
                print (' splited by Spacy\' sentence spliter is failed to be loaded into TextInstance\n')
                for _tmp_text_instance in _tmp_text_instances:
                    print (_tmp_text_instance.offset, len(_tmp_text_instance.text), _tmp_text_instance.text)
        new_text_instances.extend(_tmp_text_instances)
    
    document.text_instances = new_text_instances

def tokenize_document_by_spacy(document, nlp):
    for text_instance in document.text_instances: 
  
        doc = nlp(re.sub(r'\s+', ' ', text_instance.text))
                
        tokens = []
        for i, token in enumerate(doc):
            tokens.append(token.text)
            text_instance.pos_tags.append(token.pos_)
            text_instance.head.append(token.dep_)
            text_instance.head_indexes.append(token.head.i)
            text_instance.stems.append(token.lemma_)
            
        text_instance.tokenized_text = ' '.join(tokens)

def tokenize_documents_by_spacy(documents, spacy_model):
    
    nlp = spacy.load(spacy_model)
    
    for document in documents:
        split_sentence(document, nlp)
        tokenize_document_by_spacy(document, nlp)

def shift_neighbor_indices_and_add_end_tag(tagged_sent,
                                           ne_positions,
                                           ne_list, 
                                           neighbor_indices,
                                           has_end_tag):
                    
    new_tagged_sent = tagged_sent.split(' ')
    
    # if parsing sentence fail => len(neighbor_indices) == 0
    if len(neighbor_indices) > 0:
        # update indices by using ne_positions if indices > ne_positions then shift NE's length
        for _neighbor_indices in neighbor_indices:
            
            for i, _indice in enumerate(_neighbor_indices):
                
                if not has_end_tag:
                    _shift_num = 0
                else:
                    # we consider "end tag" as part of ne text
                    _shift_num = 1
                for j, shift_point_indice in enumerate(ne_positions):
                    if _indice > shift_point_indice:
                        _shift_num += len(ne_list[j].split(' '))
                _neighbor_indices[i] += _shift_num
            
    ne_positions.reverse()
    ne_list.reverse()
        
    for ne_position, ne_text in zip(ne_positions, ne_list):
        
        if len(neighbor_indices) > 0:
            ne_tag_neighbor_indices = neighbor_indices[ne_position]
        
        # add ne into neighbor and tagged sent
        for i, _ne_token in enumerate(ne_text.split(' ')):
            
            if len(neighbor_indices) > 0:
                # ne text point to ne tag
                neighbor_indices.insert(ne_position + i, [ne_position])
                neighbor_indices[ne_position + i] += ne_tag_neighbor_indices
            
            # insert ne text
            new_tagged_sent.insert(ne_position + 1 + i, _ne_token)
        
        
        if has_end_tag:
            # ne text point to ne tag
            end_tag_index = ne_position + len(ne_text.split(' ')) + 1
            if len(neighbor_indices) > 0:
                neighbor_indices.insert(end_tag_index, [ne_position])
                neighbor_indices[end_tag_index] += ne_tag_neighbor_indices
            new_tagged_sent.insert(end_tag_index, new_tagged_sent[ne_position].replace('@', '@/'))
                        
    return ' '.join(new_tagged_sent)
        
def convert_iob2_to_tagged_sent(
        tokens, 
        labels, 
        in_neighbors_list,
        token_offset,
        to_mask_src_and_tgt = False,
        has_end_tag = False):
    
        
    num_orig_tokens = len(tokens)
    
    previous_label = 'O'
    
    orig_token_index_2_new_token_index_mapping = []
    
    current_idx = -1
    
    tagged_sent = ''
    ne_type = ''
    ne_text = ''
    ne_list = []
    # convert IOB2 to bert format
    # NEs are replaced by tags
    for i, (token, label) in enumerate(zip(tokens, labels)):
        if label == 'O':
            if previous_label != 'O':
                tagged_sent += '$ ' + token
                #print('1 ne_list.append(ne_text)', ne_text)
                ne_list.append(ne_text)
                ne_text = ''
            else:
                tagged_sent += ' ' + token    
            current_idx += 1
                
        elif label.startswith('B-'):
            if previous_label != 'O':
                tagged_sent += '$ @' + label[2:]
                #print('2 ne_list.append(ne_text)', ne_text)
                ne_list.append(ne_text)
                ne_text = token
                ne_type = label[2:]
            else:
                tagged_sent += ' @' + label[2:]
                ne_text = token
                ne_type = label[2:]
            current_idx += 1
                
        elif label.startswith('I-'):
            ne_text += ' ' + token
        #print('=================>')
        #print(i, token, label)
        #print(tagged_sent)
        previous_label = label
        orig_token_index_2_new_token_index_mapping.append(current_idx)
    if ne_text != '':
        ne_list.append(ne_text)
        ne_text = ''
    tagged_sent = tagged_sent.strip()
    if previous_label != 'O':
        tagged_sent += '$'
            
    #    
    
    # update neighbor index
    previous_idx = 0
    _new_neighbors = [] # 
    
        
    _tokens = tagged_sent.split(' ')
    
    ne_positions = []
    for i in range(len(_tokens)):
        token = _tokens[i]
        if token.startswith('@') and token.endswith('$'):
            ne_positions.append(i)
            
    new_in_neighbors_list = []
    if len(in_neighbors_list) != 0:
        # update in_neighbors_list to new_in_neighbors_list by orig_token_index_2_new_token_index_mapping
        for i in range(num_orig_tokens):
            if previous_idx == orig_token_index_2_new_token_index_mapping[i]:
                for neighbor_idx in in_neighbors_list[i]:
                    _new_neighbors.append(orig_token_index_2_new_token_index_mapping[neighbor_idx])
            else:
                new_in_neighbors_list.append(list(set(_new_neighbors)))
                _new_neighbors = []
                for neighbor_idx in in_neighbors_list[i]:
                    _new_neighbors.append(orig_token_index_2_new_token_index_mapping[neighbor_idx])
            previous_idx = orig_token_index_2_new_token_index_mapping[i]
        new_in_neighbors_list.append(list(set(_new_neighbors)))
    #
    
    # insert ne text and update neighbor index again
    if to_mask_src_and_tgt == False:
        tagged_sent = shift_neighbor_indices_and_add_end_tag(
                               tagged_sent,
                               ne_positions,
                               ne_list,
                               new_in_neighbors_list,
                               has_end_tag)
    #

    # add token_offset to neighbor index
    new_in_neighbors_list = ['|'.join([str(i + token_offset) for i in set(neighbors)]) for neighbors in new_in_neighbors_list]
    
    
    return tagged_sent.strip(),\
           ' '.join(new_in_neighbors_list),\
           token_offset + len(new_in_neighbors_list)
    
def enumerate_all_id_pairs_by_specified(document,
                                        src_tgt_pairs,
                                        only_pair_in_same_sent):
    all_pairs = set()
    
    if only_pair_in_same_sent:
        for text_instance in document.text_instances:
                    
            all_id_infos_list = list()
            _all_id_infos_set = set()
            
            text_instance.annotations = sorted(text_instance.annotations, key=lambda x: x.position, reverse=False)
            for annotation in text_instance.annotations:
                for id in annotation.ids:
                    if (id, annotation.ne_type) not in _all_id_infos_set:
                        all_id_infos_list.append((id, annotation.ne_type))
                        _all_id_infos_set.add((id, annotation.ne_type))
            
            #print('====>len(all_id_infos_list)', len(all_id_infos_list))
            for i in range(0, len(all_id_infos_list) - 1):
                id1_info = all_id_infos_list[i]
                for j in range(i + 1, len(all_id_infos_list)):
                    id2_info = all_id_infos_list[j]
                    #print(id1_info[0], id2_info[1], id1_info[1], id2_info[1])
                    for src_ne_type, tgt_ne_type in src_tgt_pairs:
                        if id1_info[1] == src_ne_type and id2_info[1] == tgt_ne_type:
                            all_pairs.add((id1_info[0], id2_info[0], id1_info[1], id2_info[1]))
                            break
                            #print('OK')
                        elif id2_info[1] == src_ne_type and id1_info[1] == tgt_ne_type:
                            all_pairs.add((id2_info[0], id1_info[0], id2_info[1], id1_info[1]))
                            break
                        #print('OK')
    else:    
        all_id_infos_list = list()
        _all_id_infos_set = set()
        
        for text_instance in document.text_instances:
            text_instance.annotations = sorted(text_instance.annotations, key=lambda x: x.position, reverse=False)
            for annotation in text_instance.annotations:
                for id in annotation.ids:
                    if (id, annotation.ne_type) not in _all_id_infos_set:
                        all_id_infos_list.append((id, annotation.ne_type))
                        _all_id_infos_set.add((id, annotation.ne_type))
        
        #print('====>len(all_id_infos_list)', len(all_id_infos_list))
        for i in range(0, len(all_id_infos_list) - 1):
            id1_info = all_id_infos_list[i]
            for j in range(i + 1, len(all_id_infos_list)):
                id2_info = all_id_infos_list[j]
                #print(id1_info[0], id2_info[1], id1_info[1], id2_info[1])
                for src_ne_type, tgt_ne_type in src_tgt_pairs:
                    if id1_info[1] == src_ne_type and id2_info[1] == tgt_ne_type:
                        all_pairs.add((id1_info[0], id2_info[0], id1_info[1], id2_info[1]))
                        break
                        #print('OK')
                    elif id2_info[1] == src_ne_type and id1_info[1] == tgt_ne_type:
                        all_pairs.add((id2_info[0], id1_info[0], id2_info[1], id1_info[1]))
                        break
                        #print('OK')
    return all_pairs
    
def enumerate_all_id_pairs(document,
                           src_ne_type,
                           tgt_ne_type,
                           only_pair_in_same_sent):
    all_pairs = set()
    
    if only_pair_in_same_sent:
        for text_instance in document.text_instances:
                    
            all_id_infos_list = list()
            _all_id_infos_set = set()
            
            text_instance.annotations = sorted(text_instance.annotations, key=lambda x: x.position, reverse=False)
            for annotation in text_instance.annotations:
                for id in annotation.ids:
                    if (id, annotation.ne_type) not in _all_id_infos_set:
                        all_id_infos_list.append((id, annotation.ne_type))
                        _all_id_infos_set.add((id, annotation.ne_type))
            
            #print('====>len(all_id_infos_list)', len(all_id_infos_list))
            for i in range(0, len(all_id_infos_list) - 1):
                id1_info = all_id_infos_list[i]
                for j in range(i + 1, len(all_id_infos_list)):
                    id2_info = all_id_infos_list[j]
                    #print(id1_info[0], id2_info[1], id1_info[1], id2_info[1])
                    if src_ne_type == 'Any' and tgt_ne_type == 'Any':
                        # comparing NE types (id_info[1])
                        if id1_info[1] <= id2_info[1]:
                            all_pairs.add((id1_info[0], id2_info[0], id1_info[1], id2_info[1]))
                        else:
                            all_pairs.add((id2_info[0], id1_info[0], id2_info[1], id1_info[1]))
                    elif id1_info[1] == src_ne_type and id2_info[1] == tgt_ne_type:
                        all_pairs.add((id1_info[0], id2_info[0], id1_info[1], id2_info[1]))
                        #print('OK')
                    elif id2_info[1] == src_ne_type and id1_info[1] == tgt_ne_type:
                        all_pairs.add((id2_info[0], id1_info[0], id2_info[1], id1_info[1]))
                        #print('OK')
    else:    
        all_id_infos_list = list()
        _all_id_infos_set = set()
        
        for text_instance in document.text_instances:
            text_instance.annotations = sorted(text_instance.annotations, key=lambda x: x.position, reverse=False)
            for annotation in text_instance.annotations:
                for id in annotation.ids:
                    if (id, annotation.ne_type) not in _all_id_infos_set:
                        all_id_infos_list.append((id, annotation.ne_type))
                        _all_id_infos_set.add((id, annotation.ne_type))
        
        #print('====>len(all_id_infos_list)', len(all_id_infos_list))
        for i in range(0, len(all_id_infos_list) - 1):
            id1_info = all_id_infos_list[i]
            for j in range(i + 1, len(all_id_infos_list)):
                id2_info = all_id_infos_list[j]
                #print(id1_info[0], id2_info[1], id1_info[1], id2_info[1])
                if src_ne_type == 'Any' and tgt_ne_type == 'Any':
                    if id1_info[1] <= id2_info[1]:
                        all_pairs.add((id1_info[0], id2_info[0], id1_info[1], id2_info[1]))
                    else:
                        all_pairs.add((id2_info[0], id1_info[0], id2_info[1], id1_info[1]))
                elif id1_info[1] == src_ne_type and id2_info[1] == tgt_ne_type:
                    all_pairs.add((id1_info[0], id2_info[0], id1_info[1], id2_info[1]))
                    #print('OK')
                elif id2_info[1] == src_ne_type and id1_info[1] == tgt_ne_type:
                    all_pairs.add((id2_info[0], id1_info[0], id2_info[1], id1_info[1]))
                    #print('OK')
    return all_pairs
        
def convert_text_instance_2_iob2(text_instance, id1, id2, do_mask_other_nes):
    tokens = []
    labels = []
    
    for token in text_instance.tokenized_text.split(' '):
        tokens.append(token)
        labels.append('O')
        
    annotation_indexes_wo_count_space = []
    for annotation in text_instance.annotations:
        start = len(text_instance.text[:annotation.position].replace(' ', ''))
        end = start + len(annotation.text.replace(' ', ''))
        annotation_indexes_wo_count_space.append((start, end))
    
    for (start, end), annotation in zip(annotation_indexes_wo_count_space, text_instance.annotations):
        offset = 0
        for i, token in enumerate(tokens):
            if offset == start:
                if id1 in annotation.ids:
                    labels[i] = "B-" + annotation.ne_type + 'Src'
                elif id2 in annotation.ids:
                    labels[i] = "B-" + annotation.ne_type + 'Tgt'
                elif do_mask_other_nes:
                    labels[i] = "B-" + annotation.ne_type
            elif start < offset and offset < end:
                if id1 in annotation.ids:
                    labels[i] = "I-" + annotation.ne_type + 'Src'
                elif id2 in annotation.ids:
                    labels[i] = "I-" + annotation.ne_type + 'Tgt'
                elif do_mask_other_nes:
                    labels[i] = "I-" + annotation.ne_type
            elif offset < start and start < offset + len(token): #ex: renin-@angiotensin$
                if id1 in annotation.ids:
                    labels[i] = "B-" + annotation.ne_type + 'Src'
                elif id2 in annotation.ids:
                    labels[i] = "B-" + annotation.ne_type + 'Tgt'
                elif do_mask_other_nes:
                    labels[i] = "B-" + annotation.ne_type                
                    
            offset += len(token)
        
    return tokens, labels

__treated_as_disease_set = set(
        list(['patient', 'patients', 'family', 'families', 'child', 'children', 'woman', 'man', 'men', 'women']))
__treated_as_variant_set = set(
        list(['mutation', 'mutations', 'mutant', 'mutants', 'polymorphism', 'polymorphizm', 'variant', 'variants', 'men', 'women']))

def treated_as_disease(token):
    
    return token.lower() in __treated_as_disease_set

def treated_as_variant(token):
    
    return token.lower() in __treated_as_variant_set


def convert_text_instance_2_iob2_with_corresponding_gene_id(
        text_instance, 
        id1,
        id2, 
        do_mask_other_nes,
        unique_disease_id = '',
        gene_id_2_variant_ids = {}):
    tokens = []
    labels = []
    
    for token in text_instance.tokenized_text.split(' '):
        tokens.append(token)
        labels.append('O')
        
    annotation_indexes_wo_count_space = []
    for annotation in text_instance.annotations:
        start = len(text_instance.text[:annotation.position].replace(' ', ''))
        end = start + len(annotation.text.replace(' ', ''))
        annotation_indexes_wo_count_space.append((start, end))
    
    for (start, end), annotation in zip(annotation_indexes_wo_count_space, text_instance.annotations):
        offset = 0
        
        for i, token in enumerate(tokens):
            if offset == start:
                # only variant has corresponding_gene_id, if corresponding_gene_id is gene id then considers the variant as gene
                if id1 == annotation.corresponding_gene_id:
                    labels[i] = 'B-' + annotation.ne_type + 'Src'
                elif id2 == annotation.corresponding_gene_id:
                    labels[i] = 'B-' + annotation.ne_type + 'Tgt'
                elif (id1 in annotation.ids):
                    labels[i] = 'B-' + annotation.ne_type + 'Src'
                elif (id2 in annotation.ids) or (id2 == annotation.corresponding_gene_id):
                    labels[i] = 'B-' + annotation.ne_type + 'Tgt'
                elif do_mask_other_nes:
                    labels[i] = 'B-' + annotation.ne_type
            elif start < offset and offset < end:
                if id1 == annotation.corresponding_gene_id:
                    labels[i] = 'I-' + annotation.ne_type + 'Src'
                elif id2 == annotation.corresponding_gene_id:
                    labels[i] = 'I-' + annotation.ne_type + 'Tgt'
                elif (id1 in annotation.ids):
                    labels[i] = 'I-' + annotation.ne_type + 'Src'
                elif (id2 in annotation.ids):
                    labels[i] = 'I-' + annotation.ne_type + 'Tgt'
                elif do_mask_other_nes:
                    labels[i] = 'I-' + annotation.ne_type
            elif offset < start and start < offset + len(token): #ex: renin-@angiotensin$
                
                if id1 == annotation.corresponding_gene_id:
                    labels[i] = 'B-' + annotation.ne_type + 'Src'
                elif id2 == annotation.corresponding_gene_id:
                    labels[i] = 'B-' + annotation.ne_type + 'Tgt'
                elif (id1 in annotation.ids):
                    labels[i] = 'B-' + annotation.ne_type + 'Src'
                elif (id2 in annotation.ids):
                    labels[i] = 'B-' + annotation.ne_type + 'Tgt'
                elif do_mask_other_nes:
                    labels[i] = 'B-' + annotation.ne_type
            elif unique_disease_id != '' and treated_as_disease(token):
                labels[i] = 'B-DiseaseOrPhenotypicFeatureSrc'
                    
            offset += len(token)
        
    return tokens, labels

def convert_text_instance_2_iob2_for_dv(
        text_instance, 
        id1,
        id2, 
        do_mask_other_nes,
        gene_id_2_variant_ids = {}):
    tokens = []
    labels = []
    
    for token in text_instance.tokenized_text.split(' '):
        tokens.append(token)
        labels.append('O')
    
    variant_ids = set()
    has_gene_id = False
    annotation_indexes_wo_count_space = []
    for annotation in text_instance.annotations:
        start = len(text_instance.text[:annotation.position].replace(' ', ''))
        end = start + len(annotation.text.replace(' ', ''))
        annotation_indexes_wo_count_space.append((start, end))
        if (id1 in annotation.ids) and annotation.ne_type == 'GeneOrGeneProduct':
            has_gene_id = True
            if id1 in gene_id_2_variant_ids:
                variant_ids = gene_id_2_variant_ids[id1]
        elif (id2 in annotation.ids) and annotation.ne_type == 'GeneOrGeneProduct':
            has_gene_id = True
            if id2 in gene_id_2_variant_ids:
                variant_ids = gene_id_2_variant_ids[id2]
            
    
    for (start, end), annotation in zip(annotation_indexes_wo_count_space, text_instance.annotations):
        offset = 0
        
        for i, token in enumerate(tokens):
            if offset == start:
                # only variant has corresponding_gene_id, if corresponding_gene_id is gene id then considers the variant as gene
                if (id1 in annotation.ids):
                    labels[i] = 'B-' + annotation.ne_type + 'Src'
                elif (id2 in annotation.ids) or (id2 == annotation.corresponding_gene_id):
                    labels[i] = 'B-' + annotation.ne_type + 'Tgt'
                elif do_mask_other_nes:
                    labels[i] = 'B-' + annotation.ne_type
            elif start < offset and offset < end:
                if (id1 in annotation.ids):
                    labels[i] = 'I-' + annotation.ne_type + 'Src'
                elif (id2 in annotation.ids):
                    labels[i] = 'I-' + annotation.ne_type + 'Tgt'
                elif do_mask_other_nes:
                    labels[i] = 'I-' + annotation.ne_type
            elif offset < start and start < offset + len(token): #ex: renin-@angiotensin$
                
                if (id1 in annotation.ids):
                    labels[i] = 'B-' + annotation.ne_type + 'Src'
                elif (id2 in annotation.ids):
                    labels[i] = 'B-' + annotation.ne_type + 'Tgt'
                elif do_mask_other_nes:
                    labels[i] = 'B-' + annotation.ne_type
            elif has_gene_id and treated_as_variant(token) and (id2 in variant_ids):
                labels[i] = 'B-SequenceVariantTgt'
                    
            offset += len(token)
        
    return tokens, labels

def convert_text_instance_2_iob2_for_dvg(
        text_instance, 
        disease_id, 
        gene_id, 
        variant_id,
        disease_tag,
        gene_tag,
        variant_tag,
        do_mask_other_nes):
    tokens = []
    labels = []
    
    for token in text_instance.tokenized_text.split(' '):
        tokens.append(token)
        labels.append('O')
        
    annotation_indexes_wo_count_space = []
    for annotation in text_instance.annotations:
        start = len(text_instance.text[:annotation.position].replace(' ', ''))
        end = start + len(annotation.text.replace(' ', ''))
        annotation_indexes_wo_count_space.append((start, end))
    
    for (start, end), annotation in zip(annotation_indexes_wo_count_space, text_instance.annotations):
        offset = 0
        for i, token in enumerate(tokens):
            if offset == start:
                if gene_id in annotation.ids:
                    labels[i] = "B-" + gene_tag + 'Tgt'
                elif variant_id in annotation.ids:
                    labels[i] = "B-" + variant_tag + 'Tgt'
                    tokens[i] = 'variant'
                elif disease_id in annotation.ids:
                    labels[i] = "B-" + disease_tag + 'Src'
                elif do_mask_other_nes:
                    labels[i] = "B-" + annotation.ne_type
            elif start < offset and offset < end:
                if gene_id in annotation.ids:
                    labels[i] = "I-" + gene_tag + 'Tgt'
                elif variant_id in annotation.ids:
                    labels[i] = "I-" + variant_tag + 'Tgt'
                    tokens[i] = 'variant'
                elif disease_id in annotation.ids:
                    labels[i] = "I-" + disease_tag + 'Src'
                elif do_mask_other_nes:
                    labels[i] = "I-" + annotation.ne_type
            elif offset < start and start < offset + len(token): #ex: renin-@angiotensin$
                if gene_id in annotation.ids:
                    labels[i] = "B-" + gene_tag + 'Tgt'
                elif variant_id in annotation.ids:
                    labels[i] = "B-" + variant_tag + 'Tgt'
                    tokens[i] = 'variant'
                elif disease_id in annotation.ids:
                    labels[i] = "I-" + disease_tag + 'Src'
                elif do_mask_other_nes:
                    labels[i] = "B-" + annotation.ne_type                
                    
            offset += len(token)
        
    return tokens, labels

def get_in_neighbors_list(text_instance):
    in_neighbors_list = []
    in_neighbors_head_list = []
    for current_idx, (head, head_idx) in enumerate(zip(
                                         text_instance.head,
                                         text_instance.head_indexes)):
        neighbors = []
        neighbors_head = []
        
        neighbors.append(head_idx)
        neighbors_head.append(head)
        
        in_neighbors_list.append(neighbors)
        in_neighbors_head_list.append(neighbors_head)
        
    return in_neighbors_list, in_neighbors_head_list

def get_ne_id_2_ne_text_dict(document):
    ne_id_2_ne_text_dict = {}
    for text_instance in document.text_instances:
        for ann in text_instance.annotations:
            for id in ann.ids:
                ne_id_2_ne_text_dict[id] = ann.text
    return ne_id_2_ne_text_dict
            
def enumerate_all_dgv_pairs(
        document,
        disease_tag = 'DiseaseOrPhenotypicFeature',
        gene_tag = 'GeneOrGeneProduct',
        variant_tag = 'SequenceVariant'):
    
    all_dgv_pairs = set()
    
    gene_ids = set()
    disease_ids = set()
    var_ids = set()
    
    for text_instance in document.text_instances:
        for ann in text_instance.annotations:
            if ann.orig_ne_type == variant_tag:    
                for id in ann.ids:
                    var_ids.add(id)
            elif ann.ne_type == gene_tag:    
                for id in ann.ids:
                    gene_ids.add(id)
            elif ann.ne_type == disease_tag:    
                for id in ann.ids:
                    disease_ids.add(id)

    for var_id, gene_id in document.variant_gene_pairs:
        for disease_id in disease_ids:
            if (var_id in var_ids) and (gene_id in gene_ids):
                all_dgv_pairs.add((disease_id, gene_id, var_id))
                
    return all_dgv_pairs

def dump_documents_2_bert_gt_format(
    all_documents, 
    out_bert_file, 
    src_ne_type,
    tgt_ne_type,
    src_tgt_pairs = set(),
    is_test_set = False, 
    do_mask_other_nes = False,
    only_pair_in_same_sent = False,
    to_mask_src_and_tgt = False,
    to_insert_src_and_tgt_at_left = False,
    has_novelty = False,
    has_end_tag = False,
    task_tag = None,
    neg_label = 'None',
    pos_label = '',
    has_dgv = False,
    use_corresponding_gene_id = False,
    has_ne_type = False,
    only_co_occurrence_sent = False):
    
    num_seq_lens = []
    
    #print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
    _index = 0
    with open(out_bert_file, 'w', encoding='utf8') as bert_writer:
        
        if is_test_set:
            if not has_novelty:
                bert_writer.write('pmid\tid1\tid2\tis_in_same_sent\tmin_sents_window\tsentence\tin_neighbors\tlabel\n')
            else:
                bert_writer.write('pmid\tid1\tid2\tis_in_same_sent\tmin_sents_window\tsentence\tin_neighbors\tlabel\tnovelty\n')
        
        number_unique_YES_instances = 0
        
        #print('================>')
        #print(len(all_documents))
        
        for document in all_documents:
            pmid = document.id
            if len(src_tgt_pairs) == 0:
                all_pairs = enumerate_all_id_pairs(document,
                                                   src_ne_type,
                                                   tgt_ne_type,
                                                   only_pair_in_same_sent)
            else:
                all_pairs = enumerate_all_id_pairs_by_specified(document,
                                                   src_tgt_pairs,
                                                   only_pair_in_same_sent)
                
            all_dgv_pairs = set()
            if has_dgv:
                all_dgv_pairs = enumerate_all_dgv_pairs(document)
                
            
            gene_id_2_variant_ids = {}
            for v_id, g_id in document.variant_gene_pairs:
                if g_id not in gene_id_2_variant_ids:
                    gene_id_2_variant_ids[g_id] = set()
                gene_id_2_variant_ids[g_id].add(v_id)
            
            ne_id_2_ne_text_dict = get_ne_id_2_ne_text_dict(document)
            
            unique_YES_instances = set()
            
            unique_disease_id = ''
            disease_ids = set()
            variant_ids = set()
            for text_instance in document.text_instances:
                for annotation in text_instance.annotations:
                    if annotation.ne_type == 'DiseaseOrPhenotypicFeature':
                        for id in annotation.ids:
                            disease_ids.add(id)
                    elif annotation.orig_ne_type == 'SequenceVariant':
                        for id in annotation.ids:
                            variant_ids.add(id)
            if len(disease_ids) == 1:
                for id in disease_ids:    
                    unique_disease_id = id

                        
            #print('===============>document.relation_pairs', document.relation_pairs)
            #print('===============>all_pairs', all_pairs)
            
            # for pairs have two entities
            for relation_pair in all_pairs:
        
                if not has_novelty:
                    relation_label = neg_label
                else:
                    relation_label = neg_label + '|None' # rel_type|novelty novelty => ['None', 'No', 'Novel']
                ne_text_pair = ''
                
                #print('=================>relation_pair', relation_pair)
                if not document.relation_pairs:
                    #print('=================>no relation_pair', document.id)
                    document.relation_pairs = {}
                
                if (relation_pair[0], relation_pair[1]) in document.relation_pairs:
                    relation_label = document.relation_pairs[(relation_pair[0], relation_pair[1])]
                    if pos_label != '' and (not has_novelty):
                        relation_label = pos_label
                elif (relation_pair[1], relation_pair[0]) in document.relation_pairs:
                    relation_label = document.relation_pairs[(relation_pair[1], relation_pair[0])]
                    if pos_label != '' and (not has_novelty):
                        relation_label = pos_label
                id1 = relation_pair[0]
                id2 = relation_pair[1]    
                id1type = relation_pair[2]
                id2type = relation_pair[3]
                ne_text_pair = ne_id_2_ne_text_dict[id1] + ' and ' + ne_id_2_ne_text_dict[id2]
                
                tagged_ne_text_pair = '@' + id1type + 'Src$ ' + ne_id_2_ne_text_dict[id1] + ' @/' + id1type + 'Src$ and ' +\
                                      '@' + id2type + 'Tgt$ ' + ne_id_2_ne_text_dict[id2] + ' @/' + id2type + 'Tgt$'
                
                tagged_sents = []
                all_sents_in_neighbors = []
                #all_sents_out_neighbors = []
                
                is_in_same_sent = False
                
                src_sent_ids = []
                tgt_sent_ids = []
                                
                token_offset = 0
                
                pair_type = ''
                
                if id1type == 'GeneOrGeneProduct' and id2type == 'DiseaseOrPhenotypicFeature':
                    pair_type = 'dg'
                elif id2type == 'GeneOrGeneProduct' and id1type == 'DiseaseOrPhenotypicFeature':
                    pair_type = 'dg'
                elif (id1type == 'SequenceVariant' or (id1 in variant_ids)) and id2type == 'DiseaseOrPhenotypicFeature':
                    pair_type = 'dv'
                elif (id2type == 'SequenceVariant' or (id2 in variant_ids)) and id1type == 'DiseaseOrPhenotypicFeature':
                    pair_type = 'dv'
                
                #print('ggggggggggggggggg')
                for sent_id, text_instance in enumerate(document.text_instances):
                    
                    if use_corresponding_gene_id and (pair_type == 'dg'):
                        if (id1 == '6331' and id2 == 'D001919') or (id2 == '6331' and id1 == 'D001919'):
                            for ann in text_instance.annotations:
                                print(ann.text, ann.ne_type)
                        tokens, labels = convert_text_instance_2_iob2_with_corresponding_gene_id(
                                text_instance, 
                                id1, 
                                id2, 
                                do_mask_other_nes,
                                unique_disease_id,
                                gene_id_2_variant_ids)
                        if (id1 == '6331' and id2 == 'D001919') or (id2 == '6331' and id1 == 'D001919'):
                            print(' '.join(labels))
                    elif use_corresponding_gene_id and (pair_type == 'dv'):
                        tokens, labels = convert_text_instance_2_iob2_for_dv(
                                text_instance, 
                                id1, 
                                id2, 
                                do_mask_other_nes,
                                gene_id_2_variant_ids)
                    else:
                        tokens, labels = convert_text_instance_2_iob2(text_instance, id1, id2, do_mask_other_nes)
                    #print(' '.join(tokens))
                    
                    in_neighbors_list, _ = get_in_neighbors_list(text_instance)
                    #out_neighbors_list, _ = get_out_neighbors_list(text_instance)
                    
                    # raise if neighbor is wrong
                    if len(tokens) != len(in_neighbors_list):
                        print('==================>')
                        print('len(tokens)', len(tokens))
                        print('len(in_neighbors_list)', len(in_neighbors_list))
                        print('tokens', tokens)
                        print(document.id, sent_id)
                        '''
                        for _sent_id, _text_instance in enumerate(document.text_instances):
                            
                            print(_sent_id, _text_instance.tokenized_text)
                            
                        for current_idx, (head, head_idx) in enumerate(zip(
                                         text_instance.head,
                                         text_instance.head_indexes)):
                            print(tokens[current_idx], head_idx, head, in_neighbors_list[current_idx])
                        print('==================>')
                        for i in range(len(tokens)):
                            print(tokens[i], in_neighbors_list[i])'''
                        #raise('GG')
                        in_neighbors_list = []
                    #
                    
                    # check if Source and Target are in the same sentence
                    if not is_in_same_sent:
                        is_Src_in = False
                        is_Tgt_in = False
                        for _label in labels:
                            if 'Src' in _label:
                                is_Src_in = True
                                src_sent_ids.append(sent_id)
                                break
                        for _label in labels:
                            if 'Tgt' in _label:
                                is_Tgt_in = True
                                tgt_sent_ids.append(sent_id)
                                break
                        if is_Src_in and is_Tgt_in:
                            is_in_same_sent = True
                    #
                        
                    
                    tagged_sent, in_neighbors_str, token_offset =\
                        convert_iob2_to_tagged_sent(
                            tokens,
                            labels,
                            in_neighbors_list,
                            #out_neighbors_list,
                            token_offset,
                            to_mask_src_and_tgt,
                            has_end_tag)
                      
                    if only_co_occurrence_sent:
                        if is_in_same_sent:
                            tagged_sents.append(tagged_sent)
                            all_sents_in_neighbors.append(in_neighbors_str)
                    else:
                        tagged_sents.append(tagged_sent)
                        all_sents_in_neighbors.append(in_neighbors_str)
                    #all_sents_out_neighbors.append(out_neighbors_str)
                
                
                min_sents_window = 100
                for src_sent_id in src_sent_ids:
                    for tgt_sent_id in tgt_sent_ids:
                        _min_sents_window = abs(src_sent_id - tgt_sent_id)
                        if _min_sents_window < min_sents_window:
                            min_sents_window = _min_sents_window
                            
                num_seq_lens.append(float(len(tagged_sent.split(' '))))
                
                #print('================>id1', id1)
                #print('================>all_sents_in_neighbors', all_sents_in_neighbors)
                
                if pair_type == 'dg' or pair_type == 'dv':
                    #out_sent = ' '.join(tagged_sents)
                    if to_mask_src_and_tgt:
                        out_sent = 'What is ' + task_tag + ' ? [SEP] ' + ' '.join(tagged_sents)
                    else:
                        out_sent = 'What is ' + task_tag + ' between ' + tagged_ne_text_pair + ' ? [SEP] ' + ' '.join(tagged_sents)
                elif task_tag != None:
                    if to_mask_src_and_tgt:
                        out_sent = 'What is ' + task_tag + ' ? [SEP] ' + ' '.join(tagged_sents)
                    else:
                        out_sent = 'What is ' + task_tag + ' between ' + tagged_ne_text_pair + ' ? [SEP] ' + ' '.join(tagged_sents)
                elif to_insert_src_and_tgt_at_left:
                    if to_mask_src_and_tgt:
                        out_sent = ' '.join(tagged_sents)
                    else:
                        out_sent = ne_text_pair + ' [SEP] ' + ' '.join(tagged_sents)
                else:
                    out_sent = ' '.join(tagged_sents)
                
                if id1 == '-1' or id2 == '-1':
                    continue
                if ' '.join(tagged_sents) == '':
                    continue
                if has_ne_type:
                    instance = document.id + '\t' +\
                               id1type + '\t' +\
                               id2type + '\t' +\
                               id1 + '\t' +\
                               id2 + '\t' +\
                               str(is_in_same_sent) + '\t' +\
                               str(min_sents_window) + '\t' +\
                               out_sent + '\t' +\
                               ' '.join(all_sents_in_neighbors)
                           #' '.join(all_sents_in_neighbors) + '\t' +\
                           #' '.join(all_sents_out_neighbors)
                else:
                    instance = document.id + '\t' +\
                               id1 + '\t' +\
                               id2 + '\t' +\
                               str(is_in_same_sent) + '\t' +\
                               str(min_sents_window) + '\t' +\
                               out_sent + '\t' +\
                               ' '.join(all_sents_in_neighbors)
                    
                if relation_label != neg_label:
                    unique_YES_instances.add(instance)
                
                if is_test_set or (id1 != '-' and id2 != '-'):
                    if has_novelty:
                        relation_label = relation_label.replace('|', '\t')
                    bert_writer.write(instance + '\t' + 
                                      relation_label + '\n')
                
            for dgv_pair in all_dgv_pairs:
        
                if not has_novelty:
                    relation_label = neg_label
                else:
                    relation_label = neg_label + '|None' # rel_type|novelty novelty => ['None', 'No', 'Novel']
                ne_text_pair = ''
                
                #print('=================>relation_pair', relation_pair)
                if not document.nary_relations:
                    #print('=================>no relation_pair', document.id)
                    document.nary_relations = {}
                
                if dgv_pair in document.nary_relations:
                    relation_label = document.nary_relations[dgv_pair]
                    
                disease_id = dgv_pair[0]
                gene_id = dgv_pair[1]
                variant_id = dgv_pair[2]
                
                disease_tag = 'DiseaseOrPhenotypicFeature'
                gene_tag = 'GeneOrGeneProduct'
                variant_tag = 'SequenceVariant'
                
                '''
                tagged_ne_text_pair = '@' + gene_tag + 'Src$ ' + ne_id_2_ne_text_dict[gene_id] + ' @/' + gene_tag + 'Src$ , ' +\
                                      '@' + variant_tag + 'Src$ ' + ne_id_2_ne_text_dict[variant_id] + ' @/' + variant_tag + 'Src$ , and ' +\
                                      '@' + disease_tag + 'Tgt$ ' + ne_id_2_ne_text_dict[disease_id] + ' @/' + disease_tag + 'Tgt$'
                '''
                
                '''tagged_ne_text_pair = '@' + gene_tag + 'Src$ ' + ne_id_2_ne_text_dict[gene_id] + ' @/' + gene_tag + 'Src$ , ' +\
                                      '@' + gene_tag + 'Src$ ' + ne_id_2_ne_text_dict[variant_id] + ' @/' + gene_tag + 'Src$ , and ' +\
                                      '@' + disease_tag + 'Tgt$ ' + ne_id_2_ne_text_dict[disease_id] + ' @/' + disease_tag + 'Tgt$'
                '''
                tagged_ne_text_pair = '@' + disease_tag + 'Src$ ' + ne_id_2_ne_text_dict[disease_id] + ' @/' + disease_tag + 'Src$ , ' +\
                                      '@' + gene_tag + 'Tgt$ ' + ne_id_2_ne_text_dict[gene_id] + ' @/' + gene_tag + 'Tgt$ , and ' +\
                                      '@' + variant_tag + 'Tgt$ ' + ne_id_2_ne_text_dict[variant_id] + ' @/' + variant_tag + 'Tgt$'
                tagged_sents = []
                all_sents_in_neighbors = []
                #all_sents_out_neighbors = []
                
                is_in_same_sent = False
                
                src_sent_ids = []
                tgt_sent_ids = []
                                
                token_offset = 0
                #print('ggggggggggggggggg')
                for sent_id, text_instance in enumerate(document.text_instances):
                    
                    tokens, labels = convert_text_instance_2_iob2_for_dvg(
                                                                            text_instance, 
                                                                            disease_id, 
                                                                            gene_id, 
                                                                            variant_id,
                                                                            disease_tag,
                                                                            gene_tag,
                                                                            variant_tag,
                                                                            do_mask_other_nes)
                    #print(' '.join(tokens))
                    
                    in_neighbors_list, _ = get_in_neighbors_list(text_instance)
                    #out_neighbors_list, _ = get_out_neighbors_list(text_instance)
                    
                    # raise if neighbor is wrong
                    if len(tokens) != len(in_neighbors_list):
                        print('==================>')
                        print('len(tokens)', len(tokens))
                        print('len(in_neighbors_list)', len(in_neighbors_list))
                        print('tokens', tokens)
                        print(document.id, sent_id)
                        for _sent_id, _text_instance in enumerate(document.text_instances):
                            
                            print(_sent_id, _text_instance.tokenized_text)
                            
                        for current_idx, (head, head_idx) in enumerate(zip(
                                         text_instance.head,
                                         text_instance.head_indexes)):
                            print(tokens[current_idx], head_idx, head, in_neighbors_list[current_idx])
                        print('==================>')
                        for i in range(len(tokens)):
                            print(tokens[i], in_neighbors_list[i])
                        raise('GG')
                    #
                    
                    # check if Source and Target are in the same sentence
                    if not is_in_same_sent:
                        is_Src_in = False
                        is_Tgt_in = False
                        for _label in labels:
                            if 'Src' in _label:
                                is_Src_in = True
                                src_sent_ids.append(sent_id)
                                break
                        for _label in labels:
                            if 'Tgt' in _label:
                                is_Tgt_in = True
                                tgt_sent_ids.append(sent_id)
                                break
                        if is_Src_in and is_Tgt_in:
                            is_in_same_sent = True
                    #
                        
                    
                    tagged_sent, in_neighbors_str, token_offset =\
                        convert_iob2_to_tagged_sent(
                            tokens,
                            labels,
                            in_neighbors_list,
                            #out_neighbors_list,
                            token_offset,
                            True,
                            False)
                        
                    tagged_sents.append(tagged_sent)
                    all_sents_in_neighbors.append(in_neighbors_str)
                    #all_sents_out_neighbors.append(out_neighbors_str)
                
                
                min_sents_window = 100
                for src_sent_id in src_sent_ids:
                    for tgt_sent_id in tgt_sent_ids:
                        _min_sents_window = abs(src_sent_id - tgt_sent_id)
                        if _min_sents_window < min_sents_window:
                            min_sents_window = _min_sents_window
                            
                num_seq_lens.append(float(len(tagged_sent.split(' '))))
                
                #print('================>id1', id1)
                #print('================>all_sents_in_neighbors', all_sents_in_neighbors)
                
                '''if task_tag != None:
                    out_sent = 'What is ' + task_tag + ' between ' + tagged_ne_text_pair + ' ? [SEP] ' + ' '.join(tagged_sents)
                elif to_insert_src_and_tgt_at_left:
                    out_sent = ne_text_pair + ' [SEP] ' + ' '.join(tagged_sents)
                else:
                    out_sent = ' '.join(tagged_sents)
                '''
                    
                out_sent = ' '.join(tagged_sents)
                
                if gene_id == '-1' or disease_id == '-1' or variant_id == '-1':
                    continue
                if has_ne_type:
                    instance = document.id + '\t' +\
                               gene_tag + '\t' +\
                               disease_tag + '\t' +\
                               gene_id + '@' + variant_id + '\t' +\
                               disease_id + '\t' +\
                               str(is_in_same_sent) + '\t' +\
                               str(min_sents_window) + '\t' +\
                               out_sent + '\t' +\
                               ' '.join(all_sents_in_neighbors)
                               #' '.join(all_sents_in_neighbors) + '\t' +\
                               #' '.join(all_sents_out_neighbors)
                else:
                    instance = document.id + '\t' +\
                               gene_id + '@' + variant_id + '\t' +\
                               disease_id + '\t' +\
                               str(is_in_same_sent) + '\t' +\
                               str(min_sents_window) + '\t' +\
                               out_sent + '\t' +\
                               ' '.join(all_sents_in_neighbors)
                               #' '.join(all_sents_in_neighbors) + '\t' +\
                               #' '.join(all_sents_out_neighbors)
                if relation_label != neg_label:
                    unique_YES_instances.add(instance)
                
                if is_test_set or (id1 != '-' and id2 != '-'):
                    if has_novelty:
                        relation_label = relation_label.replace('|', '\t')
                    bert_writer.write(instance + '\t' + 
                                      relation_label + '\n')
                
            number_unique_YES_instances += len(unique_YES_instances)
                    
            bert_writer.flush()
            
        print('number_unique_YES_instances', number_unique_YES_instances)
    #raise('GG')
    if len(num_seq_lens) == 0:
        return 0
    
    return sum(num_seq_lens) / len(num_seq_lens)


def dump_documents_2_bert_format(
    all_documents, 
    out_bert_file, 
    src_ne_type,
    tgt_ne_type,
    src_tgt_pairs = set(),
    is_test_set = False, 
    do_mask_other_nes = False,
    only_pair_in_same_sent = False,
    to_mask_src_and_tgt = False,
    to_insert_src_and_tgt_at_left = False,
    has_novelty = False,
    has_end_tag = False,
    task_tag = None,
    neg_label = 'None',
    pos_label = '',
    has_dgv = False,
    use_corresponding_gene_id = False,
    has_ne_type = False,
    only_co_occurrence_sent = False):
    
    num_seq_lens = []
    
    #print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
    _index = 0
    
    out_str = ''
    
    if is_test_set:
        if not has_novelty:
            out_str += 'pmid\tid1\tid2\tis_in_same_sent\tmin_sents_window\tsentence\tin_neighbors\tlabel\n'
        else:
            out_str += 'pmid\tid1\tid2\tis_in_same_sent\tmin_sents_window\tsentence\tin_neighbors\tlabel\tnovelty\n'
    
    number_unique_YES_instances = 0
    
    #print('================>')
    #print(len(all_documents))
    
    for document in all_documents:
        pmid = document.id
        if len(src_tgt_pairs) == 0:
            all_pairs = enumerate_all_id_pairs(document,
                                               src_ne_type,
                                               tgt_ne_type,
                                               only_pair_in_same_sent)
        else:
            all_pairs = enumerate_all_id_pairs_by_specified(document,
                                               src_tgt_pairs,
                                               only_pair_in_same_sent)
            
        all_dgv_pairs = set()
        if has_dgv:
            all_dgv_pairs = enumerate_all_dgv_pairs(document)
            
        
        gene_id_2_variant_ids = {}
        for v_id, g_id in document.variant_gene_pairs:
            if g_id not in gene_id_2_variant_ids:
                gene_id_2_variant_ids[g_id] = set()
            gene_id_2_variant_ids[g_id].add(v_id)
        
        ne_id_2_ne_text_dict = get_ne_id_2_ne_text_dict(document)
        
        unique_YES_instances = set()
        
        unique_disease_id = ''
        disease_ids = set()
        variant_ids = set()
        for text_instance in document.text_instances:
            for annotation in text_instance.annotations:
                if annotation.ne_type == 'DiseaseOrPhenotypicFeature':
                    for id in annotation.ids:
                        disease_ids.add(id)
                elif annotation.orig_ne_type == 'SequenceVariant':
                    for id in annotation.ids:
                        variant_ids.add(id)
        if len(disease_ids) == 1:
            for id in disease_ids:    
                unique_disease_id = id

                    
        #print('===============>document.relation_pairs', document.relation_pairs)
        #print('===============>all_pairs', all_pairs)
        
        # for pairs have two entities
        for relation_pair in all_pairs:
    
            if not has_novelty:
                relation_label = neg_label
            else:
                relation_label = neg_label + '|None' # rel_type|novelty novelty => ['None', 'No', 'Novel']
            ne_text_pair = ''
            
            #print('=================>relation_pair', relation_pair)
            if not document.relation_pairs:
                #print('=================>no relation_pair', document.id)
                document.relation_pairs = {}
            
            if (relation_pair[0], relation_pair[1]) in document.relation_pairs:
                relation_label = document.relation_pairs[(relation_pair[0], relation_pair[1])]
                if pos_label != '' and (not has_novelty):
                    relation_label = pos_label
            elif (relation_pair[1], relation_pair[0]) in document.relation_pairs:
                relation_label = document.relation_pairs[(relation_pair[1], relation_pair[0])]
                if pos_label != '' and (not has_novelty):
                    relation_label = pos_label
            id1 = relation_pair[0]
            id2 = relation_pair[1]    
            id1type = relation_pair[2]
            id2type = relation_pair[3]
            ne_text_pair = ne_id_2_ne_text_dict[id1] + ' and ' + ne_id_2_ne_text_dict[id2]
            
            tagged_ne_text_pair = '@' + id1type + 'Src$ ' + ne_id_2_ne_text_dict[id1] + ' @/' + id1type + 'Src$ and ' +\
                                  '@' + id2type + 'Tgt$ ' + ne_id_2_ne_text_dict[id2] + ' @/' + id2type + 'Tgt$'
            
            tagged_sents = []
            all_sents_in_neighbors = []
            #all_sents_out_neighbors = []
            
            is_in_same_sent = False
            
            src_sent_ids = []
            tgt_sent_ids = []
                            
            token_offset = 0
            
            pair_type = ''
            
            if id1type == 'GeneOrGeneProduct' and id2type == 'DiseaseOrPhenotypicFeature':
                pair_type = 'dg'
            elif id2type == 'GeneOrGeneProduct' and id1type == 'DiseaseOrPhenotypicFeature':
                pair_type = 'dg'
            elif (id1type == 'SequenceVariant' or (id1 in variant_ids)) and id2type == 'DiseaseOrPhenotypicFeature':
                pair_type = 'dv'
            elif (id2type == 'SequenceVariant' or (id2 in variant_ids)) and id1type == 'DiseaseOrPhenotypicFeature':
                pair_type = 'dv'
            
            #print('ggggggggggggggggg')
            for sent_id, text_instance in enumerate(document.text_instances):
                
                tagged_sent = text_instance.text
                is_Src_in = False
                is_Tgt_in = False
                text_instance.annotations.sort(key=lambda x: x.position, reverse=True)
                for ann in text_instance.annotations:
                    if id1 in ann.ids:
                        if to_mask_src_and_tgt:
                            tagged_sent = tagged_sent[0:ann.position] + '@' + id1type + 'Src$' + tagged_sent[ann.position + ann.length:]
                        else:
                            tagged_sent = tagged_sent[0:ann.position + ann.length] + ' @' + id1type + 'Src/$' + tagged_sent[ann.position + ann.length:]
                            tagged_sent = tagged_sent[0:ann.position] + '@' + id1type + 'Src$ ' + tagged_sent[ann.position:]
                        is_Src_in = True
                        src_sent_ids.append(sent_id)
                    elif id2 in ann.ids:
                        if to_mask_src_and_tgt:
                            tagged_sent = tagged_sent[0:ann.position] + '@' + id1type + 'Tgt$' + tagged_sent[ann.position + ann.length:]
                        else:
                            tagged_sent = tagged_sent[0:ann.position + ann.length] + ' @' + id2type + 'Tgt/$' + tagged_sent[ann.position + ann.length:]
                            tagged_sent = tagged_sent[0:ann.position] + '@' + id2type + 'Tgt$ ' + tagged_sent[ann.position:]
                        is_Tgt_in = True
                        tgt_sent_ids.append(sent_id)
                
                if is_Src_in and is_Tgt_in:
                    is_in_same_sent = True
                #
                if only_co_occurrence_sent:
                    if is_in_same_sent:
                        tagged_sents.append(tagged_sent)
                else:
                    tagged_sents.append(tagged_sent)
                #all_sents_out_neighbors.append(out_neighbors_str)
            
            min_sents_window = 100
            for src_sent_id in src_sent_ids:
                for tgt_sent_id in tgt_sent_ids:
                    _min_sents_window = abs(src_sent_id - tgt_sent_id)
                    if _min_sents_window < min_sents_window:
                        min_sents_window = _min_sents_window
                        
            num_seq_lens.append(float(len(tagged_sent.split(' '))))
            
            #print('================>id1', id1)
            #print('================>all_sents_in_neighbors', all_sents_in_neighbors)
            
            if pair_type == 'dg' or pair_type == 'dv':
                #out_sent = ' '.join(tagged_sents)
                if to_mask_src_and_tgt:
                    out_sent = 'What is ' + task_tag + ' ? [SEP] ' + ' '.join(tagged_sents)
                else:
                    out_sent = 'What is ' + task_tag + ' between ' + tagged_ne_text_pair + ' ? [SEP] ' + ' '.join(tagged_sents)
            elif task_tag != None:
                if to_mask_src_and_tgt:
                    out_sent = 'What is ' + task_tag + ' ? [SEP] ' + ' '.join(tagged_sents)
                else:
                    out_sent = 'What is ' + task_tag + ' between ' + tagged_ne_text_pair + ' ? [SEP] ' + ' '.join(tagged_sents)
            elif to_insert_src_and_tgt_at_left:
                if to_mask_src_and_tgt:
                    out_sent = ' '.join(tagged_sents)
                else:
                    out_sent = ne_text_pair + ' [SEP] ' + ' '.join(tagged_sents)
            else:
                out_sent = ' '.join(tagged_sents)
            
            if id1 == '-1' or id2 == '-1':
                continue
            if ' '.join(tagged_sents) == '':
                continue
            if has_ne_type:
                instance = document.id + '\t' +\
                           id1type + '\t' +\
                           id2type + '\t' +\
                           id1 + '\t' +\
                           id2 + '\t' +\
                           str(is_in_same_sent) + '\t' +\
                           str(min_sents_window) + '\t' +\
                           out_sent + '\t' +\
                           ' '.join(all_sents_in_neighbors)
                       #' '.join(all_sents_in_neighbors) + '\t' +\
                       #' '.join(all_sents_out_neighbors)
            else:
                instance = document.id + '\t' +\
                           id1 + '\t' +\
                           id2 + '\t' +\
                           str(is_in_same_sent) + '\t' +\
                           str(min_sents_window) + '\t' +\
                           out_sent + '\t' +\
                           ' '.join(all_sents_in_neighbors)
                
            if relation_label != neg_label:
                unique_YES_instances.add(instance)
            
            if is_test_set or (id1 != '-' and id2 != '-'):
                if has_novelty:
                    relation_label = relation_label.replace('|', '\t')
                out_str += instance + '\t' + relation_label + '\n'
            
        for dgv_pair in all_dgv_pairs:
    
            if not has_novelty:
                relation_label = neg_label
            else:
                relation_label = neg_label + '|None' # rel_type|novelty novelty => ['None', 'No', 'Novel']
            ne_text_pair = ''
            
            #print('=================>relation_pair', relation_pair)
            if not document.nary_relations:
                #print('=================>no relation_pair', document.id)
                document.nary_relations = {}
            
            if dgv_pair in document.nary_relations:
                relation_label = document.nary_relations[dgv_pair]
                
            disease_id = dgv_pair[0]
            gene_id = dgv_pair[1]
            variant_id = dgv_pair[2]
            
            disease_tag = 'DiseaseOrPhenotypicFeature'
            gene_tag = 'GeneOrGeneProduct'
            variant_tag = 'SequenceVariant'
            
            '''
            tagged_ne_text_pair = '@' + gene_tag + 'Src$ ' + ne_id_2_ne_text_dict[gene_id] + ' @/' + gene_tag + 'Src$ , ' +\
                                  '@' + variant_tag + 'Src$ ' + ne_id_2_ne_text_dict[variant_id] + ' @/' + variant_tag + 'Src$ , and ' +\
                                  '@' + disease_tag + 'Tgt$ ' + ne_id_2_ne_text_dict[disease_id] + ' @/' + disease_tag + 'Tgt$'
            '''
            
            '''tagged_ne_text_pair = '@' + gene_tag + 'Src$ ' + ne_id_2_ne_text_dict[gene_id] + ' @/' + gene_tag + 'Src$ , ' +\
                                  '@' + gene_tag + 'Src$ ' + ne_id_2_ne_text_dict[variant_id] + ' @/' + gene_tag + 'Src$ , and ' +\
                                  '@' + disease_tag + 'Tgt$ ' + ne_id_2_ne_text_dict[disease_id] + ' @/' + disease_tag + 'Tgt$'
            '''
            tagged_ne_text_pair = '@' + disease_tag + 'Src$ ' + ne_id_2_ne_text_dict[disease_id] + ' @/' + disease_tag + 'Src$ , ' +\
                                  '@' + gene_tag + 'Tgt$ ' + ne_id_2_ne_text_dict[gene_id] + ' @/' + gene_tag + 'Tgt$ , and ' +\
                                  '@' + variant_tag + 'Tgt$ ' + ne_id_2_ne_text_dict[variant_id] + ' @/' + variant_tag + 'Tgt$'
            tagged_sents = []
            all_sents_in_neighbors = []
            #all_sents_out_neighbors = []
            
            is_in_same_sent = False
            
            src_sent_ids = []
            tgt_sent_ids = []
                            
            token_offset = 0
            #print('ggggggggggggggggg')
            for sent_id, text_instance in enumerate(document.text_instances):
                
                tokens, labels = convert_text_instance_2_iob2_for_dvg(
                                                                        text_instance, 
                                                                        disease_id, 
                                                                        gene_id, 
                                                                        variant_id,
                                                                        disease_tag,
                                                                        gene_tag,
                                                                        variant_tag,
                                                                        do_mask_other_nes)
                #print(' '.join(tokens))
                
                in_neighbors_list, _ = get_in_neighbors_list(text_instance)
                #out_neighbors_list, _ = get_out_neighbors_list(text_instance)
                
                # raise if neighbor is wrong
                if len(tokens) != len(in_neighbors_list):
                    print('==================>')
                    print('len(tokens)', len(tokens))
                    print('len(in_neighbors_list)', len(in_neighbors_list))
                    print('tokens', tokens)
                    print(document.id, sent_id)
                    for _sent_id, _text_instance in enumerate(document.text_instances):
                        
                        print(_sent_id, _text_instance.tokenized_text)
                        
                    for current_idx, (head, head_idx) in enumerate(zip(
                                     text_instance.head,
                                     text_instance.head_indexes)):
                        print(tokens[current_idx], head_idx, head, in_neighbors_list[current_idx])
                    print('==================>')
                    for i in range(len(tokens)):
                        print(tokens[i], in_neighbors_list[i])
                    raise('GG')
                #
                
                # check if Source and Target are in the same sentence
                if not is_in_same_sent:
                    is_Src_in = False
                    is_Tgt_in = False
                    for _label in labels:
                        if 'Src' in _label:
                            is_Src_in = True
                            src_sent_ids.append(sent_id)
                            break
                    for _label in labels:
                        if 'Tgt' in _label:
                            is_Tgt_in = True
                            tgt_sent_ids.append(sent_id)
                            break
                    if is_Src_in and is_Tgt_in:
                        is_in_same_sent = True
                #
                    
                
                tagged_sent, in_neighbors_str, token_offset =\
                    convert_iob2_to_tagged_sent(
                        tokens,
                        labels,
                        in_neighbors_list,
                        #out_neighbors_list,
                        token_offset,
                        True,
                        False)
                    
                tagged_sents.append(tagged_sent)
                all_sents_in_neighbors.append(in_neighbors_str)
                #all_sents_out_neighbors.append(out_neighbors_str)
            
            
            min_sents_window = 100
            for src_sent_id in src_sent_ids:
                for tgt_sent_id in tgt_sent_ids:
                    _min_sents_window = abs(src_sent_id - tgt_sent_id)
                    if _min_sents_window < min_sents_window:
                        min_sents_window = _min_sents_window
                        
            num_seq_lens.append(float(len(tagged_sent.split(' '))))
            
            #print('================>id1', id1)
            #print('================>all_sents_in_neighbors', all_sents_in_neighbors)
            
            '''if task_tag != None:
                out_sent = 'What is ' + task_tag + ' between ' + tagged_ne_text_pair + ' ? [SEP] ' + ' '.join(tagged_sents)
            elif to_insert_src_and_tgt_at_left:
                out_sent = ne_text_pair + ' [SEP] ' + ' '.join(tagged_sents)
            else:
                out_sent = ' '.join(tagged_sents)
            '''
                
            out_sent = ' '.join(tagged_sents)
            
            if gene_id == '-1' or disease_id == '-1' or variant_id == '-1':
                continue
            if has_ne_type:
                instance = document.id + '\t' +\
                           gene_tag + '\t' +\
                           disease_tag + '\t' +\
                           gene_id + '@' + variant_id + '\t' +\
                           disease_id + '\t' +\
                           str(is_in_same_sent) + '\t' +\
                           str(min_sents_window) + '\t' +\
                           out_sent + '\t' +\
                           ' '.join(all_sents_in_neighbors)
                           #' '.join(all_sents_in_neighbors) + '\t' +\
                           #' '.join(all_sents_out_neighbors)
            else:
                instance = document.id + '\t' +\
                           gene_id + '@' + variant_id + '\t' +\
                           disease_id + '\t' +\
                           str(is_in_same_sent) + '\t' +\
                           str(min_sents_window) + '\t' +\
                           out_sent + '\t' +\
                           ' '.join(all_sents_in_neighbors)
                           #' '.join(all_sents_in_neighbors) + '\t' +\
                           #' '.join(all_sents_out_neighbors)
            if relation_label != neg_label:
                unique_YES_instances.add(instance)
            
            if is_test_set or (id1 != '-' and id2 != '-'):
                if has_novelty:
                    relation_label = relation_label.replace('|', '\t')
                out_str += instance + '\t' + relation_label + '\n'
            
        number_unique_YES_instances += len(unique_YES_instances)
                
    print('number_unique_YES_instances', number_unique_YES_instances)
    #raise('GG')
    
    with open(out_bert_file, 'w', encoding='utf8') as bert_writer:
        bert_writer.write(out_str)
        
    if len(num_seq_lens) == 0:
        return 0
    return sum(num_seq_lens) / len(num_seq_lens)

def get_rel_types_distribution(all_documents):
    
    rel_count_dict = defaultdict(float)
    
    num_all_rel_pairs = 0.
    for document in all_documents:
        if not document.relation_pairs:
            #print('=================>no relation_pair', document.id)
            document.relation_pairs = {}
                
        for relation_pair, rel_type in document.relation_pairs.items():
            rel_count_dict[rel_type] += 1.
            num_all_rel_pairs += 1.

    rel_ratio_dict = defaultdict(float)
    for rel_type in rel_count_dict.keys():
        rel_ratio_dict[rel_type] = rel_count_dict[rel_type] / num_all_rel_pairs
    
    return rel_ratio_dict, rel_count_dict
    

def dump_documents_2_bert_gt_format_by_sent_level(
    all_documents, 
    out_bert_file, 
    src_ne_type,
    tgt_ne_type,
    src_tgt_pairs = set(),
    is_test_set = False, 
    do_mask_other_nes = False,
    to_mask_src_and_tgt = False,
    to_insert_src_and_tgt_at_left = False,
    has_end_tag = False,
    task_tag = None,
    neg_label = 'None',
    add_ne_type = True,
    pos_label = ''):
    
    num_seq_lens = []
    
    #print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXX')

    print('=========>out_bert_file', out_bert_file)

    #logger.info('=========>out_bert_file' + out_bert_file)
    
    #debug_writer = open('123_' + out_bert_file.split('\\')[-1].split('/')[-1], 'w', encoding='utf8')

    #logger.info('=========>out_bert_file' + '123_' + out_bert_file.split('\\')[-1].split('/')[-1])

    with open(out_bert_file, 'w', encoding='utf8') as bert_writer:
        print('=============>out_bert_file', out_bert_file, 'opened')
        if is_test_set:
            bert_writer.write('pmid\tid1\tid2\tsentence\tlabel\n')
            #debug_writer.write('pmid\tid1\tid2\tsentence\tlabel\n')
            #debug_writer.flush()

        number_unique_YES_instances = 0
        
        #print('================>')
        #print(len(all_documents))
        
        for document in all_documents:
            
            if len(src_tgt_pairs) == 0:
                all_pairs = enumerate_all_id_pairs(document,
                                                   src_ne_type,
                                                   tgt_ne_type,
                                                   True)
            else:
                all_pairs = enumerate_all_id_pairs_by_specified(document,
                                                   src_tgt_pairs,
                                                   True)
            
            ne_id_2_ne_text_dict = get_ne_id_2_ne_text_dict(document)
            
            unique_YES_instances = set()
            #print('===============>document.relation_pairs', document.relation_pairs)
            #print('===============>all_pairs', all_pairs)
            
            for relation_pair in all_pairs:
        
                relation_label = neg_label
                
                ne_text_pair = ''
                
                #print('=================>relation_pair', relation_pair)
                if not document.relation_pairs:
                    #print('=================>no relation_pair', document.id)
                    document.relation_pairs = {}
                
                if (relation_pair[0], relation_pair[1]) in document.relation_pairs:
                    relation_label = document.relation_pairs[(relation_pair[0], relation_pair[1])]
                    if pos_label != '':
                        relation_label = pos_label
                elif (relation_pair[1], relation_pair[0]) in document.relation_pairs:
                    relation_label = document.relation_pairs[(relation_pair[1], relation_pair[0])]
                    if pos_label != '':
                        relation_label = pos_label
                
                id1 = relation_pair[0]
                id2 = relation_pair[1]
                id1type = relation_pair[2]
                id2type = relation_pair[3]    
                ne_text_pair = ne_id_2_ne_text_dict[id1] + ' and ' + ne_id_2_ne_text_dict[id2]                
                tagged_ne_text_pair = '@' + id1type + 'Src$ ' + ne_id_2_ne_text_dict[id1] + ' @/' + id1type + 'Src$ and ' +\
                                      '@' + id2type + 'Tgt$ ' + ne_id_2_ne_text_dict[id2] + ' @/' + id2type + 'Tgt$'
                
                #all_sents_out_neighbors = []
                
                
                #print('ggggggggggggggggg')
                for sent_id, text_instance in enumerate(document.text_instances):
                    
                    try:
                        tokens, labels = convert_text_instance_2_iob2(text_instance, id1, id2, do_mask_other_nes)
                        #print(' '.join(tokens))
                    
                        _has_src = False
                        _has_tgt = False
                        for label in labels:
                            if label.startswith('B-') and label.endswith('Src'):
                                _has_src = True
                            elif label.startswith('B-') and label.endswith('Tgt'):
                                _has_tgt = True
                        if (not _has_src) or (not _has_tgt):
                            continue
                        in_neighbors_list, _ = get_in_neighbors_list(text_instance)
                        #out_neighbors_list, _ = get_out_neighbors_list(text_instance)
                    
                        if len(tokens) != len(in_neighbors_list):
                            print('==================>')
                            print('len(tokens)', len(tokens))
                            print('len(in_neighbors_list)', len(in_neighbors_list))
                            print('tokens', tokens)
                            print(document.id, sent_id)
                            for _sent_id, _text_instance in enumerate(document.text_instances):
                             
                                print(_sent_id, _text_instance.tokenized_text)
                            
                            for current_idx, (head, head_idx) in enumerate(zip(
                                         text_instance.head,
                                         text_instance.head_indexes)):
                                print(tokens[current_idx], head_idx, head, in_neighbors_list[current_idx])
                            print('==================>')
                            for i in range(len(tokens)):
                                print(tokens[i], in_neighbors_list[i])
                            #raise('GG')
                            continue
                                                         
                        tagged_sent, in_neighbors_str, token_offset =\
                            convert_iob2_to_tagged_sent(
                                tokens,
                                labels,
                                in_neighbors_list,
                                #out_neighbors_list,
                                0,
                                to_mask_src_and_tgt,
                                has_end_tag)                            
                    
                        if task_tag != None:
                            if to_mask_src_and_tgt:
                                out_sent = 'What is ' + task_tag + ' ? [SEP] ' + tagged_sent
                            else:
                                out_sent = 'What is ' + task_tag + ' between ' + tagged_ne_text_pair + ' ? [SEP] ' + tagged_sent
                        elif to_insert_src_and_tgt_at_left:
                            if to_mask_src_and_tgt:
                                out_sent = tagged_sent
                            else:
                                out_sent = ne_text_pair + ' [SEP] ' + tagged_sent
                        else:
                            out_sent = tagged_sent
                        
                        if add_ne_type:
                            instance = document.id + '\t' +\
                                   id1type + '\t' +\
                                   id2type + '\t' +\
                                   id1 + '\t' +\
                                   id2 + '\t' +\
                                   'True\t' +\
                                   '0\t' +\
                                   out_sent + '\t' +\
                                   in_neighbors_str
                        else:
                            instance = document.id + '\t' +\
                                   id1 + '\t' +\
                                   id2 + '\t' +\
                                   'True\t' +\
                                   '0\t' +\
                                   out_sent + '\t' +\
                                   in_neighbors_str
                        if relation_label != neg_label:
                            unique_YES_instances.add(instance)
                    
                        if is_test_set or (id1 != '-' and id2 != '-'):
                            bert_writer.write(instance + '\t' + 
                                          relation_label + '\n')
                            #debug_writer.write(instance + '\t' +
                            #                  relation_label + '\n')
                    except:
                        continue
                
            number_unique_YES_instances += len(unique_YES_instances)
                    
            bert_writer.flush()
            #debug_writer.flush()
            
        print('number_unique_YES_instances', number_unique_YES_instances)
    #raise('GG')
    #debug_writer.close()
    return 0

def dump_documents_2_bert_format_by_sent_level(
    all_documents, 
    out_bert_file, 
    src_ne_type,
    tgt_ne_type,
    is_test_set = False, 
    do_mask_other_nes = False,
    to_mask_src_and_tgt = False,
    to_insert_src_and_tgt_at_left = False,
    has_end_tag = False,
    task_tag = None,
    neg_label = 'None'):
    
    num_seq_lens = []
    
    #print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXX')

    print('=========>out_bert_file', out_bert_file)

    #logger.info('=========>out_bert_file' + out_bert_file)
    
    #debug_writer = open('123_' + out_bert_file.split('\\')[-1].split('/')[-1], 'w', encoding='utf8')

    #logger.info('=========>out_bert_file' + '123_' + out_bert_file.split('\\')[-1].split('/')[-1])
    out_str = ''
    
    print('=============>out_bert_file', out_bert_file, 'opened')
    if is_test_set:
        out_str += 'pmid\tid1\tid2\tsentence\tlabel\n'
        #debug_writer.write('pmid\tid1\tid2\tsentence\tlabel\n')
        #debug_writer.flush()

    number_unique_YES_instances = 0
    
    #print('================>')
    #print(len(all_documents))
    
    for document in all_documents:
        
        all_pairs = enumerate_all_id_pairs(document,
                                           src_ne_type,
                                           tgt_ne_type,
                                           True)
        
        ne_id_2_ne_text_dict = get_ne_id_2_ne_text_dict(document)
        
        unique_YES_instances = set()
        #print('===============>document.relation_pairs', document.relation_pairs)
        #print('===============>all_pairs', all_pairs)
        
        for relation_pair in all_pairs:
    
            relation_label = neg_label
            
            ne_text_pair = ''
            
            #print('=================>relation_pair', relation_pair)
            if not document.relation_pairs:
                #print('=================>no relation_pair', document.id)
                document.relation_pairs = {}
            
            if (relation_pair[0], relation_pair[1]) in document.relation_pairs:
                relation_label = document.relation_pairs[(relation_pair[0], relation_pair[1])]
            elif (relation_pair[1], relation_pair[0]) in document.relation_pairs:
                relation_label = document.relation_pairs[(relation_pair[1], relation_pair[0])]
            
            id1 = relation_pair[0]
            id2 = relation_pair[1]
            id1type = relation_pair[2]
            id2type = relation_pair[3]    
            ne_text_pair = ne_id_2_ne_text_dict[id1] + ' and ' + ne_id_2_ne_text_dict[id2]                
            tagged_ne_text_pair = '@' + id1type + 'Src$ ' + ne_id_2_ne_text_dict[id1] + ' @/' + id1type + 'Src$ and ' +\
                                  '@' + id2type + 'Tgt$ ' + ne_id_2_ne_text_dict[id2] + ' @/' + id2type + 'Tgt$'
            
            #all_sents_out_neighbors = []
            
            
            #print('ggggggggggggggggg')
            for sent_id, text_instance in enumerate(document.text_instances):
                
                try:
                    tokens, labels = convert_text_instance_2_iob2(text_instance, id1, id2, do_mask_other_nes)
                    #print(' '.join(tokens))
                
                    if 'B-' + src_ne_type + 'Src' not in labels:
                        continue
                
                    in_neighbors_list, _ = get_in_neighbors_list(text_instance)
                    #out_neighbors_list, _ = get_out_neighbors_list(text_instance)
                
                    if len(tokens) != len(in_neighbors_list):
                        print('==================>')
                        print('len(tokens)', len(tokens))
                        print('len(in_neighbors_list)', len(in_neighbors_list))
                        print('tokens', tokens)
                        print(document.id, sent_id)
                        for _sent_id, _text_instance in enumerate(document.text_instances):
                         
                            print(_sent_id, _text_instance.tokenized_text)
                        
                        for current_idx, (head, head_idx) in enumerate(zip(
                                     text_instance.head,
                                     text_instance.head_indexes)):
                            print(tokens[current_idx], head_idx, head, in_neighbors_list[current_idx])
                        print('==================>')
                        for i in range(len(tokens)):
                            print(tokens[i], in_neighbors_list[i])
                        #raise('GG')
                        continue
                                                     
                    tagged_sent, in_neighbors_str, token_offset =\
                        convert_iob2_to_tagged_sent(
                            tokens,
                            labels,
                            in_neighbors_list,
                            #out_neighbors_list,
                            0,
                            to_mask_src_and_tgt,
                            has_end_tag)                            
                
                    if task_tag != None:
                        out_sent = 'What is ' + task_tag + ' between ' + tagged_ne_text_pair + ' ? [SEP] ' + tagged_sent
                    elif to_insert_src_and_tgt_at_left:
                        out_sent = ne_text_pair + ' [SEP] ' + tagged_sent
                    else:
                        out_sent = tagged_sent
                    
                    instance = document.id + '\t' +\
                           id1 + '\t' +\
                           id2 + '\t' +\
                           out_sent
                           
                    if relation_label != neg_label:
                        unique_YES_instances.add(instance)
                
                    if is_test_set or (id1 != '-' and id2 != '-'):
                        out_str += instance + '\t' + relation_label + '\n'
                        #debug_writer.write(instance + '\t' +
                        #                  relation_label + '\n')
                except:
                    continue
            
        number_unique_YES_instances += len(unique_YES_instances)
                
        #debug_writer.flush()
        
    print('number_unique_YES_instances', number_unique_YES_instances)
    with open(out_bert_file, 'w', encoding='utf8') as bert_writer:
        bert_writer.write(out_str)
    #raise('GG')
    #debug_writer.close()
    return 0

