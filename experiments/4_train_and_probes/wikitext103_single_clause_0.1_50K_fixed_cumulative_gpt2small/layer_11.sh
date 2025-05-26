
source .venv/bin/activate
export PYTHONPATH=.

for seed in $(seq 1 5); do
    python structural_probes/run_experiment.py \
        config/cumulative_gpt2small/11.yaml \
        --train-probe \
        --seed $seed \
        --corpus_root data \
        --corpus_train_path wikitext103_single_clause_0.1_50K/train.conllx \
        --corpus_dev_path wikitext103_single_clause_0.1_50K/dev.conllx \
        --corpus_test_path wikitext103_single_clause_0.1_50K/test.conllx \
        --embeddings_root data \
        --embeddings_train_path wikitext103_single_clause_0.1_50K/train.gpt2small \
        --embeddings_dev_path wikitext103_single_clause_0.1_50K/dev.gpt2small \
        --embeddings_test_path wikitext103_single_clause_0.1_50K/test.gpt2small \
        --output_dir results/wikitext103_single_clause_0.1_50K_cumulative_gpt2small
done

for seed in $(seq 1 5); do
    for extra_test_name in prep_good double_prep_good prep_obj_good simple_svo_good; do
        python structural_probes/run_experiment.py \
            config/cumulative_gpt2small/11.yaml \
            --seed $seed \
            --corpus_root data \
            --corpus_train_path marvin/$extra_test_name.conllx \
            --corpus_dev_path marvin/$extra_test_name.conllx \
            --corpus_test_path marvin/$extra_test_name.conllx \
            --embeddings_root data \
            --embeddings_train_path marvin/$extra_test_name.gpt2small \
            --embeddings_dev_path marvin/$extra_test_name.gpt2small \
            --embeddings_test_path marvin/$extra_test_name.gpt2small \
            --output_dir results/wikitext103_single_clause_0.1_50K_cumulative_gpt2small \
            --extra_test_output_name $extra_test_name
    done
done
