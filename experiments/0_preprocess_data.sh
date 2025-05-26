#!/bin/bash
source .venv/bin/activate
export PYTHONPATH=.

# preprocess wikitext103
python scripts/parse_wikitext_103.py \
    --output_dir "data/wikitext103"

python scripts/post_edit_wiki_conll.py \
    --input_path "data/wikitext103/all.conllx" \
    --output_path "data/wikitext103/all_post_edit.conllx"


# marvin to conllx
python scripts/marvin_to_conllx.py \
    --input_files "data/marvin/prep_fixed_good.txt" "data/marvin/double_prep_fixed_good.txt" "data/marvin/prep_obj_good.txt" "data/marvin/simple_svo_fixed_good.txt" \
    --output_dir "data/marvin"
