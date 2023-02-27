# -*- coding: utf-8 -*-

from collections import defaultdict
from pathlib import Path
import optparse
import random

parser = optparse.OptionParser()

parser.add_option('--exp_option',
                  action="store",
                  dest="exp_option",         
                  help="sampling tasks", default="")

parser.add_option('--in_sampling_data_dir',
                  action="store",
                  dest="in_sampling_data_dir",         
                  help="input sampling data dir", default="")

parser.add_option('--out_dataset_dir',
                  action="store",
                  dest="out_dataset_dir",         
                  help="output processed folder path", default="")

parser.add_option('--out_sampling_data_dir',
                  action="store",
                  dest="out_sampling_data_dir",         
                  help="output sampling meta folder path", default="")

def load_pair_dict(in_pair_file, top_n = 100):

    pairs_dict = defaultdict(int)
    top_n_pairs_dict = defaultdict(str)

    with open(in_pair_file, 'r', encoding='utf8') as reader:
        _top_n = 0
        for line in reader:

            line = line.rstrip()
            tks = line.split('\t')
            id1 = tks[0]
            id2 = tks[1]
            count = int(tks[3])
            pairs_dict[(id1, id2)] = count
            if _top_n < top_n:
                top_n_pairs_dict[(id1, id2)] = tks[4]
                _top_n += 1

    return pairs_dict, top_n_pairs_dict

def generate_pos_neg_list_file(top_n,
                               in_sampling_data_dir,
                               out_sampling_data_dir):

    pairs = [('ChemicalEntity', 'ChemicalEntity'), 
             ('ChemicalEntity', 'DiseaseOrPhenotypicFeature'),
             ('ChemicalEntity', 'GeneOrGeneProduct'),
             ('DiseaseOrPhenotypicFeature',  'GeneOrGeneProduct'),
             ('GeneOrGeneProduct',     'GeneOrGeneProduct')]
    
    for pair in pairs:

        in_positive_pair_file = in_sampling_data_dir + '/out_' + pair[0] + '-' + pair[1] + '.pmi.by_pmid.txt'
        in_co_occur_pair_file = in_sampling_data_dir + '/' + pair[0] + '-' + pair[1] + '_co_occur.txt'
        out_list_file = out_sampling_data_dir + '/' + pair[0] + '-' + pair[1] + '_list.txt'
        
        positive_pairs_dict, top_n_pairs_dict = load_pair_dict(in_positive_pair_file, top_n)

        writer = open(out_list_file, 'w', encoding='utf8')

        for top_n_pair, str_pmids in top_n_pairs_dict.items():
            writer.write(top_n_pair[0] + '\t' + top_n_pair[1] + '\tP\t' + str(len(str_pmids.split('|'))) + '\t' + str_pmids + '\n')
        num_not_in_pos = 0
        with open(in_co_occur_pair_file, 'r', encoding='utf8') as reader:
            _top_n = 0
            for line in reader:
                line = line.rstrip()
                tks = line.split('\t')
                id1 = tks[0]
                id2 = tks[1]
                count = int(tks[2])
                pmids = tks[3]
                id_pair = (id1, id2)

                if id_pair not in positive_pairs_dict:
                    if _top_n < top_n:
                        writer.write(id1 + '\t' + id2 + '\tN\t' + str(len(pmids.split('|'))) + '\t' + pmids + '\n')
                        _top_n += 1
                    num_not_in_pos += 1
        num_pos = len(positive_pairs_dict)
        print(pair[0] + '\t' + 
              pair[1] + '\t' + 
              str(num_not_in_pos + num_pos) + '\t' +
              str(num_not_in_pos) + '\t' +
              str(num_pos) + '\t' +
              str(float(num_pos)/float(num_not_in_pos + num_pos)))

        writer.close()

def load_positive_and_negative_pairs(
            positive_pairs,
            negative_pairs,
            in_pos_and_neg_tsv_file):
    
    with open(in_pos_and_neg_tsv_file, 'r', encoding='utf8') as reader:
        
        for line in reader:
            
            line = line.rstrip()
            tks = line.split('\t')
            id1 = tks[0]
            id2 = tks[1]
            pair = (id1, id2)
            label = tks[2]
            pmids = set(tks[4].split('|'))
            if label == 'P':
                positive_pairs[pair] = pmids
            else:
                negative_pairs[pair] = pmids

def run_1M_distant_supervision_sampling(
        top_n,
        out_dataset_dir,
        in_sampling_data_dir,
        out_sampling_data_dir):

    Path(out_dataset_dir).mkdir(parents=True, exist_ok=True)    
    Path(out_sampling_data_dir).mkdir(parents=True, exist_ok=True)  
    out_train_tsv_file = out_dataset_dir + '/train.tsv'
    
    pairs = [('ChemicalEntity', 'ChemicalEntity'), 
             ('ChemicalEntity', 'DiseaseOrPhenotypicFeature'),
             ('ChemicalEntity', 'GeneOrGeneProduct'),
             ('DiseaseOrPhenotypicFeature',  'GeneOrGeneProduct'),
             ('GeneOrGeneProduct',     'GeneOrGeneProduct')]
    
    generate_pos_neg_list_file(top_n = top_n,
                               in_sampling_data_dir  = in_sampling_data_dir,
                               out_sampling_data_dir = out_sampling_data_dir)
    
    positive_pairs = {}
    negative_pairs = {}
    for pair in pairs:
        
        in_pos_and_neg_tsv_file = out_sampling_data_dir + '/' + pair[0] + '-' + pair[1] + '_list.txt'
                
        load_positive_and_negative_pairs(
                positive_pairs,
                negative_pairs,
                in_pos_and_neg_tsv_file)
    
    train_writer = open(out_train_tsv_file, 'w', encoding='utf8')
    
    # Putting all processed .tsv into one folder can simplify below codes
    for i in range(1,11):
        
        for j in range(i * 100-99, i*100 + 1):

            in_processed_tsv_file = '../pubtator_rel' + str(i) + '/datasets/all_pubtator/1M_new/processed/' + str(j) + '.tsv' 
            
            with open(in_processed_tsv_file, 'r', encoding='utf8') as reader:
                
                for line in reader:
                    line = line.rstrip()
                    tks = line.split('\t')
                    pmid = tks[0]
                    id1  = tks[3]
                    id2  = tks[4]
                    pair = (id1, id2)
                    if (pair in positive_pairs) and (pmid in positive_pairs[pair]):
                        tks[-1] = 'Association\n'
                        line = '\t'.join(tks)
                        train_writer.write(line)
                        del positive_pairs[pair]
                    elif (pair in negative_pairs) and (pmid in negative_pairs[pair]):
                        tks[-1] = 'None\n'
                        line = '\t'.join(tks)
                        train_writer.write(line)
                        del negative_pairs[pair]
    train_writer.close()
    
def run_8_datasets_sampling(
        in_train_tsv_file,
        out_train80_tsv_file,
        out_test20_tsv_file):
    
    document_ids = set()
    with open(in_train_tsv_file, 'r', encoding='utf8') as reader:
        for line in reader:
            doc_id = line.split('\t')[0]
            document_ids.add(doc_id)
            
    document_ids = list(document_ids)
    document_ids.sort()
    random.shuffle(document_ids)
    test_ids = set(document_ids[:int(len(document_ids)*0.2)])
    
    train_writer = open(out_train80_tsv_file, 'w', encoding='utf8')
    test_writer = open(out_test20_tsv_file, 'w', encoding='utf8')
    
    with open(in_train_tsv_file, 'r', encoding='utf8') as tsv_reader:
        for line in tsv_reader:
            tks = line.split('\t')
            doc_id = tks[0]
            if doc_id not in test_ids:
                train_writer.write(line)
            else:
                test_writer.write(line)
    
    train_writer.close()
    test_writer.close()
    
def run_10_cv_sampling(
        in_train_tsv_file,
        out_cv_dir):
    
    document_ids = set()
    with open(in_train_tsv_file, 'r', encoding='utf8') as reader:
        for line in reader:
            doc_id = line.split('\t')[0]
            document_ids.add(doc_id)
            
    document_ids = list(document_ids)
    document_ids.sort()
    random.shuffle(document_ids)
    test_ids = set(document_ids[:int(len(document_ids)*0.2)])
    
    total = len(document_ids)
    size = int(total / 10)
    
    for i in range(10):
        
        
        if i < 9:
            train_ids = set(document_ids[0:i*size] + document_ids[(i+1)*size:])
            test_ids = set(document_ids[(i)*size:(i+1)*size])
        else:
            train_ids = set(document_ids[0:i*size])
            test_ids = set(document_ids[(i)*size:])
        
        
        print(total, len(train_ids), len(test_ids))
        print(train_ids.intersection(test_ids))
        
        train_writer = open(out_cv_dir + 'train' + str(i) + '.tsv', 'w', encoding='utf8')
        test_writer = open(out_cv_dir + 'test' + str(i) + '.tsv', 'w', encoding='utf8')
        
        with open(in_train_tsv_file, 'r', encoding='utf8') as tsv_reader:
            for line in tsv_reader:
                tks = line.split('\t')
                doc_id = tks[0]
                if doc_id not in test_ids:
                    train_writer.write(line)
                else:
                    test_writer.write(line)
        
        train_writer.close()
        test_writer.close()
        

def run_10_cv_sampling_no_doc_id(
        in_train_tsv_file,
        out_cv_dir):
    
    all_lines = []
    with open(in_train_tsv_file, 'r', encoding='utf8') as reader:
        for line in reader:
            all_lines.append(line)
            
    random.shuffle(all_lines)
    
    total = len(all_lines)
    size = int(total / 10)
    
    for i in range(10):        
        
        if i < 9:
            train_lines = set(all_lines[0:i*size] + all_lines[(i+1)*size:])
            test_lines = set(all_lines[(i)*size:(i+1)*size])
        else:
            train_lines = set(all_lines[0:i*size])
            test_lines = set(all_lines[(i)*size:])
        
        
        print(total, len(train_lines), len(test_lines))
        
        train_writer = open(out_cv_dir + 'train' + str(i) + '.tsv', 'w', encoding='utf8')
        test_writer = open(out_cv_dir + 'test' + str(i) + '.tsv', 'w', encoding='utf8')
        
        for train_line in train_lines:
            train_writer.write(train_line)
            
        for test_line in test_lines:
            test_writer.write(test_line)
        
        train_writer.close()
        test_writer.close()
        
if __name__ == '__main__':
    
    
    options, args = parser.parse_args()
    exp_option = options.exp_option
    
    if exp_option == 'divide_2_train_and_test':        
        run_8_datasets_sampling(
            in_train_tsv_file    = 'datasets/disgenet/processed/train.tsv',
            out_train80_tsv_file = 'datasets/disgenet/processed/train80.tsv',
            out_test20_tsv_file  = 'datasets/disgenet/processed/test20.tsv')
        run_8_datasets_sampling(
            in_train_tsv_file    = 'datasets/emu_bc/processed/train.tsv',
            out_train80_tsv_file = 'datasets/emu_bc/processed/train80.tsv',
            out_test20_tsv_file  = 'datasets/emu_bc/processed/test20.tsv')
        run_8_datasets_sampling(
            in_train_tsv_file    = 'datasets/emu_pc/processed/train.tsv',
            out_train80_tsv_file = 'datasets/emu_pc/processed/train80.tsv',
            out_test20_tsv_file  = 'datasets/emu_pc/processed/test20.tsv')
        run_8_datasets_sampling(
            in_train_tsv_file    = 'datasets/pharmgkb/processed/train.tsv',
            out_train80_tsv_file = 'datasets/pharmgkb/processed/train80.tsv',
            out_test20_tsv_file  = 'datasets/pharmgkb/processed/test20.tsv')
        run_8_datasets_sampling(
            in_train_tsv_file    = 'datasets/aimed/processed/train.tsv',
            out_train80_tsv_file = 'datasets/aimed/processed/train80.tsv',
            out_test20_tsv_file  = 'datasets/aimed/processed/test20.tsv')
        run_8_datasets_sampling(
            in_train_tsv_file    = 'datasets/hprd50/processed/train.tsv',
            out_train80_tsv_file = 'datasets/hprd50/processed/train80.tsv',
            out_test20_tsv_file  = 'datasets/hprd50/processed/test20.tsv')
                
    elif exp_option == 'aimed_10cv':
        
        run_10_cv_sampling_no_doc_id(
            in_train_tsv_file    = 'datasets/aimed/processed/train.tsv',
            out_cv_dir           = 'datasets/aimed/')
                
    elif exp_option == 'hprd_10cv':
        
        run_10_cv_sampling_no_doc_id(
            in_train_tsv_file    = 'datasets/hprd50/processed/train.tsv',
            out_cv_dir           = 'datasets/hprd50/')