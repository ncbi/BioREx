#!/bin/bash

cuda_visible_devices=0

task_name="biorex"

in_biorex_train="datasets/biorex/processed/train.tsv"
in_biorex_test="datasets/biorex/processed/test.tsv"

in_train_tsv_file="${in_biorex_train}"
in_test_tsv_file="${in_biorex_test}"

biored_eval_filter="ChemicalEntity|ChemicalEntity;ChemicalEntity|DiseaseOrPhenotypicFeature;ChemicalEntity|GeneOrGeneProduct;DiseaseOrPhenotypicFeature|GeneOrGeneProduct;GeneOrGeneProduct|GeneOrGeneProduct"

pre_train_model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"

entity_num=2

cuda_visible_devices=$cuda_visible_devices python src/run_ncbi_rel_exp.py \
  --task_name $task_name \
  --train_file "${in_train_tsv_file}" \
  --dev_file "${in_train_tsv_file}" \
  --test_file "${in_test_tsv_file}" \
  --use_balanced_neg false \
  --to_add_tag_as_special_token true \
  --model_name_or_path "${pre_train_model}" \
  --output_dir "biorex_model" \
  --num_train_epochs 10 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 32 \
  --do_train \
  --do_predict \
  --logging_steps 10 \
  --evaluation_strategy steps \
  --save_steps 10 \
  --overwrite_output_dir \
  --max_seq_length 512
  
cp "biorex_model/test_results.tsv" "out_biorex_results.tsv"

python src/utils/run_pubtator_eval.py --exp_option 'biored_eval' \
  --in_gold_tsv_file "${in_test_tsv_file}" \
  --in_pred_tsv_file "out_biorex_results.tsv" \
  --out_bin_result_file "out_biorex_bin_results.txt" \
  --out_result_file "out_biorex_results.txt" \
  --out_pred_pubtator_file "out_biorex_bin_results.pubtator"
  
