
source .venv/bin/activate
export PYTHONPATH=.

for seed in $(seq 1 5); do
    python structural_probes/run_experiment.py \
        config/cumulative_gpt2medium/17.yaml \
        --train-probe \
        --seed $seed \
        --corpus_root data \
        --corpus_train_path wikitext103_single_clause_0.1_50K/train.conllx \
        --corpus_dev_path wikitext103_single_clause_0.1_50K/dev.conllx \
        --corpus_test_path wikitext103_single_clause_0.1_50K/test.conllx \
        --embeddings_root data \
        --embeddings_train_path wikitext103_single_clause_0.1_50K/train.gpt2medium \
        --embeddings_dev_path wikitext103_single_clause_0.1_50K/dev.gpt2medium \
        --embeddings_test_path wikitext103_single_clause_0.1_50K/test.gpt2medium \
        --output_dir results/wikitext103_single_clause_0.1_50K_cumulative_gpt2medium
done

for seed in $(seq 1 5); do
    for extra_test_name in prep_good double_prep_good prep_obj_good simple_svo_good; do
        python structural_probes/run_experiment.py \
            config/cumulative_gpt2medium/17.yaml \
            --seed $seed \
            --corpus_root data \
            --corpus_train_path marvin/$extra_test_name.conllx \
            --corpus_dev_path marvin/$extra_test_name.conllx \
            --corpus_test_path marvin/$extra_test_name.conllx \
            --embeddings_root data \
            --embeddings_train_path marvin/$extra_test_name.gpt2medium \
            --embeddings_dev_path marvin/$extra_test_name.gpt2medium \
            --embeddings_test_path marvin/$extra_test_name.gpt2medium \
            --output_dir results/wikitext103_single_clause_0.1_50K_cumulative_gpt2medium \
            --extra_test_output_name $extra_test_name
    done
done
