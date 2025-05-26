#!/bin/bash
source .venv/bin/activate
export PYTHONPATH=.
exp_name="wikitext103_single_clause_0.1_50K_fixed"

target_test_sets=(
    "test"
    "prep_fixed_good"
    "double_prep_fixed_good"
    "prep_obj_good"
    "simple_svo_fixed_good"
)

for target_test_set in "${target_test_sets[@]}"; do
    python scripts/aggregate_results.py \
        --experiment_prefix "${exp_name}_cumulative" \
        --models "gpt2small" "gpt2medium" "bertbase" "bertlarge" \
        --include_descendants \
        --target_test_set $target_test_set
done
