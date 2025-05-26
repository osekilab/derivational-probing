#!/bin/bash
source .venv/bin/activate
export PYTHONPATH=.
exp_name="wikitext103_single_clause_0.1_50K_fixed"
python scripts/filter_data.py \
    --config_path config/data/${exp_name}.json
