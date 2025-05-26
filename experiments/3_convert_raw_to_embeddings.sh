#!/bin/bash
source .venv/bin/activate
export PYTHONPATH=.
exp_name="wikitext103_single_clause_0.1_50K_fixed"

# BERT
model_sizes=("base" "large")
for model_size in "${model_sizes[@]}"; do
    python scripts/convert_raw_to_bert.py "data/${exp_name}/train.spl" "data/${exp_name}/train.bert${model_size}" "${model_size}"
    python scripts/convert_raw_to_bert.py "data/${exp_name}/dev.spl" "data/${exp_name}/dev.bert${model_size}" "${model_size}"
    python scripts/convert_raw_to_bert.py "data/${exp_name}/test.spl" "data/${exp_name}/test.bert${model_size}" "${model_size}"
done

# GPT-2
model_sizes=("small" "medium")
for model_size in "${model_sizes[@]}"; do
    python scripts/convert_raw_to_gpt2.py "data/${exp_name}/train.spl" "data/${exp_name}/train.gpt2${model_size}" "${model_size}"
    python scripts/convert_raw_to_gpt2.py "data/${exp_name}/dev.spl" "data/${exp_name}/dev.gpt2${model_size}" "${model_size}"
    python scripts/convert_raw_to_gpt2.py "data/${exp_name}/test.spl" "data/${exp_name}/test.gpt2${model_size}" "${model_size}"
done
