#!/bin/bash
 
in_pubtator_file="input.pubtator"
out_tsv_file="out_processed.tsv"
out_pubtator_file="predict.pubtator"

pre_train_model="pretrained_model_biolinkbert"

echo 'Converting the dataset into BioREx input format'

python src/dataset_format_converter/convert_pubtator_2_tsv.py \
    --exp_option biored_pred \
    --in_pubtator_file ${in_pubtator_file} \
    --out_tsv_file ${out_tsv_file}

cuda_visible_devices=$1

echo 'Generating RE predictions'
cuda_visible_devices=$cuda_visible_devices python src/run_ncbi_rel_exp.py \
  --task_name "biorex" \
  --test_file "${out_tsv_file}" \
  --use_balanced_neg false \
  --to_add_tag_as_special_token true \
  --model_name_or_path "${pre_train_model}" \
  --output_dir "biorex_model" \
  --num_train_epochs 10 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 32 \
  --do_predict \
  --logging_steps 10 \
  --evaluation_strategy steps \
  --save_steps 10 \
  --overwrite_output_dir \
  --max_seq_length 512
  
cp "biorex_model/test_results.tsv" "out_biorex_results.tsv"

python src/utils/run_pubtator_eval.py --exp_option 'to_pubtator' \
  --in_test_pubtator_file "${in_pubtator_file}" \
  --in_test_tsv_file "${out_tsv_file}" \
  --in_pred_tsv_file "out_biorex_results.tsv" \
  --out_pred_pubtator_file "${out_pubtator_file}"
  
