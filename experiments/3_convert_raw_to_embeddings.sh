#!/bin/bash
source .venv/bin/activate
export PYTHONPATH=.
exp_name="wikitext103_single_clause_0.1_50K"

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

# marvin
test_sets=(
    "prep_good"
    "double_prep_good"
    "prep_obj_good"
    "simple_svo_good"
)
# BERT for marvin sets
bert_model_sizes=("base" "large")
for model_size in "${bert_model_sizes[@]}"; do
    for test_set in "${test_sets[@]}"; do
        python scripts/convert_raw_to_bert.py "data/marvin/${test_set}.spl" "data/marvin/${test_set}.bert${model_size}" "${model_size}"
    done
done

# GPT-2 for marvin sets
gpt2_model_sizes=("small" "medium")
for model_size in "${gpt2_model_sizes[@]}"; do
    for test_set in "${test_sets[@]}"; do
        python scripts/convert_raw_to_gpt2.py "data/marvin/${test_set}.spl" "data/marvin/${test_set}.gpt2${model_size}" "${model_size}"
    done
done
