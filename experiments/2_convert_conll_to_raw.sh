#!/bin/bash
source .venv/bin/activate
export PYTHONPATH=.
exp_name="wikitext103_single_clause_0.1_50K_fixed"

python scripts/convert_conll_to_raw.py "data/${exp_name}/train.conllx" > "data/${exp_name}/train.spl"
python scripts/convert_conll_to_raw.py "data/${exp_name}/dev.conllx" > "data/${exp_name}/dev.spl"
python scripts/convert_conll_to_raw.py "data/${exp_name}/test.conllx" > "data/${exp_name}/test.spl"
