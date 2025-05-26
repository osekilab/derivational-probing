#!/bin/bash
source .venv/bin/activate
export PYTHONPATH=.
exp_name="wikitext103_single_clause_0.1_50K"

python scripts/convert_conll_to_raw.py "data/${exp_name}/train.conllx" > "data/${exp_name}/train.spl"
python scripts/convert_conll_to_raw.py "data/${exp_name}/dev.conllx" > "data/${exp_name}/dev.spl"
python scripts/convert_conll_to_raw.py "data/${exp_name}/test.conllx" > "data/${exp_name}/test.spl"

# marvin
test_sets=(
    "prep_good"
    "double_prep_good"
    "prep_obj_good"
    "simple_svo_good"
)

for test_set in "${test_sets[@]}"; do
    python scripts/convert_conll_to_raw.py "data/marvin/${test_set}.conllx" > "data/marvin/${test_set}.spl"
done
