
source .venv/bin/activate
export PYTHONPATH=.

for seed in $(seq 1 5); do
    python structural_probes/run_experiment.py \
        config/cumulative_bertlarge/16.yaml \
        --train-probe \
        --seed $seed \
        --corpus_root data \
        --corpus_train_path wikitext103_single_clause_0.1_50K_fixed/train.conllx \
        --corpus_dev_path wikitext103_single_clause_0.1_50K_fixed/dev.conllx \
        --corpus_test_path wikitext103_single_clause_0.1_50K_fixed/test.conllx \
        --embeddings_root data \
        --embeddings_train_path wikitext103_single_clause_0.1_50K_fixed/train.bertlarge \
        --embeddings_dev_path wikitext103_single_clause_0.1_50K_fixed/dev.bertlarge \
        --embeddings_test_path wikitext103_single_clause_0.1_50K_fixed/test.bertlarge \
        --output_dir results/wikitext103_single_clause_0.1_50K_fixed_cumulative_bertlarge
done

for seed in $(seq 1 5); do
    for extra_test_name in prep_fixed_good double_prep_fixed_good prep_obj_good simple_svo_fixed_good; do
        python structural_probes/run_experiment.py \
            config/cumulative_bertlarge/16.yaml \
            --seed $seed \
            --corpus_root data \
            --corpus_train_path marvin/$extra_test_name.conllx \
            --corpus_dev_path marvin/$extra_test_name.conllx \
            --corpus_test_path marvin/$extra_test_name.conllx \
            --embeddings_root data \
            --embeddings_train_path marvin/$extra_test_name.bertlarge \
            --embeddings_dev_path marvin/$extra_test_name.bertlarge \
            --embeddings_test_path marvin/$extra_test_name.bertlarge \
            --output_dir results/wikitext103_single_clause_0.1_50K_fixed_cumulative_bertlarge \
            --extra_test_output_name $extra_test_name
    done
done
