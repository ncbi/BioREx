#!/bin/bash

python src/dataset_format_converter/convert_pubtator_2_tsv.py \
    --exp_option biored

python src/dataset_format_converter/convert_pubtator_2_tsv.py \
    --exp_option aimed

python src/dataset_format_converter/convert_pubtator_2_tsv.py \
    --exp_option cdr
   
python src/dataset_format_converter/convert_pubtator_2_tsv.py \
    --exp_option ddi
 
python src/dataset_format_converter/convert_pubtator_2_tsv.py \
    --exp_option disgenet

python src/dataset_format_converter/convert_pubtator_2_tsv.py \
    --exp_option drugprot
    
python src/dataset_format_converter/convert_pubtator_2_tsv.py \
    --exp_option emu_bc
    
python src/dataset_format_converter/convert_pubtator_2_tsv.py \
    --exp_option emu_pc
    
python src/dataset_format_converter/convert_pubtator_2_tsv.py \
    --exp_option hprd50
    
python src/dataset_format_converter/convert_pubtator_2_tsv.py \
    --exp_option pharmgkb
    
python src/utils/sampling/run_sample_subset_from_tsv.py \
    --exp_option divide_2_train_and_test


norm_rel_aimed=""
norm_rel_cdr="ChemicalSrc|DiseaseTgt|*|Positive_Correlation"
norm_rel_ddi="drugSrc|drugTgt|*|Association"
norm_rel_disgenet_and_emu=""
norm_rel_drugprot="CHEMICALSrc|GENETgt|ACTIVATOR|Positive_Correlation;CHEMICALSrc|GENETgt|AGONIST|Positive_Correlation;CHEMICALSrc|GENETgt|AGONIST-ACTIVATOR|Positive_Correlation;CHEMICALSrc|GENETgt|AGONIST-INHIBITOR|Negative_Correlation;CHEMICALSrc|GENETgt|ANTAGONIST|Negative_Correlation;CHEMICALSrc|GENETgt|DIRECT-REGULATOR|Association;CHEMICALSrc|GENETgt|INDIRECT-DOWNREGULATOR|Negative_Correlation;CHEMICALSrc|GENETgt|INDIRECT-UPREGULATOR|Positive_Correlation;CHEMICALSrc|GENETgt|INHIBITOR|Negative_Correlation;CHEMICALSrc|GENETgt|PART-OF|Association;CHEMICALSrc|GENETgt|PRODUCT-OF|Association;CHEMICALSrc|GENETgt|SUBSTRATE|Association;CHEMICALSrc|GENETgt|SUBSTRATE_PRODUCT-OF|Association"
norm_rel_hprd=""
norm_rel_pharmgkb=""

norm_pair_aimed="proteinSrc|proteinTgt|GeneOrGeneProductSrc|GeneOrGeneProductTgt"
norm_pair_cdr=""
norm_pair_ddi=""
norm_pair_disgenet_and_emu="DiseaseSrc|GeneTgt|DiseaseOrPhenotypicFeatureSrc|GeneOrGeneProductTgt"
norm_pair_drugprot=""
norm_pair_hprd="GeneSrc|GeneTgt|GeneOrGeneProductSrc|GeneOrGeneProductTgt"
norm_pair_pharmgkb="ChemicalSrc|GeneTgt|ChemicalEntitySrc|GeneOrGeneProductTgt"

normalize_pair_2_rel_type="${norm_rel_aimed};${norm_rel_cdr};${norm_rel_ddi};${norm_rel_disgenet_and_emu};${norm_rel_drugprot};${norm_rel_hprd};${norm_rel_pharmgkb}"
normalize_pair_2_pair="${norm_pair_aimed};${norm_pair_cdr};${norm_pair_ddi};${norm_pair_disgenet_and_emu};${norm_pair_drugprot};${norm_pair_hprd};${norm_pair_pharmgkb}"
normalize_tag_2_tag="[DISGENET]|[Litcoin];[EMU_BC]|[Litcoin];[EMU_PC]|[Litcoin];[PHARMGKB]|[Litcoin]"

in_biored_train="datasets/ncbi_relation/processed/train.tsv"
in_biored_test="datasets/ncbi_relation/processed/test.tsv"
in_aimed_train="datasets/aimed/processed/train80.tsv"
in_aimed_test="datasets/aimed/processed/test20.tsv"
in_cdr_train="datasets/cdr/processed/train.tsv;datasets/cdr/processed/dev.tsv"
in_cdr_test="datasets/cdr/processed/test.tsv"
in_ddi_train="datasets/ddi/processed/train.tsv"
in_ddi_test="datasets/ddi/processed/test.tsv"
in_disgenet_train="datasets/disgenet/processed/train80.tsv|None-DISGENET"
in_disgenet_test="datasets/disgenet/processed/test20.tsv"
in_drugprot_train="datasets/drugprot/drugprot-gs-training-development/processed/train.tsv"
in_drugprot_test="datasets/drugprot/drugprot-gs-training-development/processed/dev.tsv"
in_emu_train="datasets/emu_pc/processed/train80.tsv|None-EMU_PC;datasets/emu_bc/processed/train80.tsv|None-EMU_BC"
in_emu_test="datasets/emu_pc/processed/test20.tsv;datasets/emu_bc/processed/test20.tsv"
in_hprd50_train="datasets/hprd50/processed/train80.tsv"
in_hprd50_test="datasets/hprd50/processed/test20.tsv"
in_pharmgkb_train="datasets/pharmgkb/processed/train80.tsv|None-PHARMGKB"
in_pharmgkb_test="datasets/pharmgkb/processed/test20.tsv"

out_biorex_train="datasets/biorex/processed/train.tsv"
out_biorex_test="datasets/biorex/processed/test.tsv"

out_aimed_test_for_biorex="datasets/aimed/processed/test_for_biorex.tsv"
out_cdr_test_for_biorex="datasets/cdr/processed/test_for_biorex.tsv"
out_ddi_test_for_biorex="datasets/ddi/processed/test_for_biorex.tsv"
out_disgenet_test_for_biorex="datasets/disgenet/processed/test_for_biorex.tsv"
out_drugprot_test_for_biorex="datasets/drugprot/processed/dev_for_biorex.tsv"
out_emu_test_for_biorex="datasets/emu/processed/test_for_biorex.tsv"
out_hprd50_test_for_biorex="datasets/hprd50/processed/test_for_biorex.tsv"
out_pharmgkb_test_for_biorex="datasets/pharmgkb/processed/test_for_biorex.tsv"


# train: biorex
# test: biored
python src/dataset_format_converter/convert_pubtator_2_tsv.py \
  --exp_option combine_sets \
  --in_main_train_files "${in_biored_train}" \
  --in_main_test_files "${in_biored_test}" \
  --in_other_train_files "${in_aimed_train};${in_cdr_train};${in_ddi_train};${in_disgenet_train};${in_drugprot_train};${in_emu_train};${in_hprd50_train};${in_pharmgkb_train}" \
  --in_other_test_files "" \
  --out_train_tsv_file "${out_biorex_train}" \
  --out_test_tsv_file "${out_biorex_test}" \
  --normalize_pair_2_rel_type "${normalize_pair_2_rel_type}" \
  --normalize_pair_2_pair "${normalize_pair_2_pair}" \
  --normalize_tag_2_tag "${normalize_tag_2_tag}"
  
python src/dataset_format_converter/convert_pubtator_2_tsv.py \
  --exp_option combine_sets \
  --in_main_test_files "${in_aimed_test}" \
  --out_test_tsv_file "${out_aimed_test_for_biorex}" \
  --normalize_pair_2_rel_type "${normalize_pair_2_rel_type}" \
  --normalize_pair_2_pair "${normalize_pair_2_pair}" \
  --normalize_tag_2_tag "${normalize_tag_2_tag}"

python src/dataset_format_converter/convert_pubtator_2_tsv.py \
  --exp_option combine_sets \
  --in_main_test_files "${in_cdr_test}" \
  --out_test_tsv_file "${out_cdr_test_for_biorex}" \
  --normalize_pair_2_rel_type "${normalize_pair_2_rel_type}" \
  --normalize_pair_2_pair "${normalize_pair_2_pair}" \
  --normalize_tag_2_tag "${normalize_tag_2_tag}"
  
python src/dataset_format_converter/convert_pubtator_2_tsv.py \
  --exp_option combine_sets \
  --in_main_test_files "${in_ddi_test}" \
  --out_test_tsv_file "${out_ddi_test_for_biorex}" \
  --normalize_pair_2_rel_type "${normalize_pair_2_rel_type}" \
  --normalize_pair_2_pair "${normalize_pair_2_pair}" \
  --normalize_tag_2_tag "${normalize_tag_2_tag}"
  
python src/dataset_format_converter/convert_pubtator_2_tsv.py \
  --exp_option combine_sets \
  --in_main_test_files "${in_disgenet_test}" \
  --out_test_tsv_file "${out_disgenet_test_for_biorex}" \
  --normalize_pair_2_rel_type "${normalize_pair_2_rel_type}" \
  --normalize_pair_2_pair "${normalize_pair_2_pair}" \
  --normalize_tag_2_tag "${normalize_tag_2_tag}"
  
python src/dataset_format_converter/convert_pubtator_2_tsv.py \
  --exp_option combine_sets \
  --in_main_test_files "${in_drugprot_test}" \
  --out_test_tsv_file "${out_drugprot_test_for_biorex}" \
  --normalize_pair_2_rel_type "${normalize_pair_2_rel_type}" \
  --normalize_pair_2_pair "${normalize_pair_2_pair}" \
  --normalize_tag_2_tag "${normalize_tag_2_tag}"
  
python src/dataset_format_converter/convert_pubtator_2_tsv.py \
  --exp_option combine_sets \
  --in_main_test_files "${in_emu_test}" \
  --out_test_tsv_file "${out_emu_test_for_biorex}" \
  --normalize_pair_2_rel_type "${normalize_pair_2_rel_type}" \
  --normalize_pair_2_pair "${normalize_pair_2_pair}" \
  --normalize_tag_2_tag "${normalize_tag_2_tag}"
    
python src/dataset_format_converter/convert_pubtator_2_tsv.py \
  --exp_option combine_sets \
  --in_main_test_files "${in_hprd50_test}" \
  --out_test_tsv_file "${out_hprd50_test_for_biorex}" \
  --normalize_pair_2_rel_type "${normalize_pair_2_rel_type}" \
  --normalize_pair_2_pair "${normalize_pair_2_pair}" \
  --normalize_tag_2_tag "${normalize_tag_2_tag}"
  
python src/dataset_format_converter/convert_pubtator_2_tsv.py \
  --exp_option combine_sets \
  --in_main_test_files "${in_pharmgkb_test}" \
  --out_test_tsv_file "${out_pharmgkb_test_for_biorex}" \
  --normalize_pair_2_rel_type "${normalize_pair_2_rel_type}" \
  --normalize_pair_2_pair "${normalize_pair_2_pair}" \
  --normalize_tag_2_tag "${normalize_tag_2_tag}"