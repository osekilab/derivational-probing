import torch
from transformers import BertTokenizer, BertForMaskedLM, AutoModelForMaskedLM
from structural_probes import DATA_DIR, PROJECT_DIR
import pickle
import json
from tqdm import tqdm


def load_bert_model_and_tokenizer(model_size="base"):
    """
    Loads BERT model and tokenizer.

    Args:
        model_size (str): Model size to use ('base' or 'large').

    Returns:
        tokenizer: BERT tokenizer.
        model: BERT model.
        device: Device to use (CPU or GPU).
    """
    if model_size == "base":
        model_name = "bert-base-cased"
    elif model_size == "large":
        model_name = "bert-large-cased"
    else:
        raise ValueError("BERT model must be base or large")

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.eval()

    # Device configuration (CPU/GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return tokenizer, model, device


def calculate_sentence_probability(sentence, tokenizer, model, device):
    """
    Calculate sentence probability.

    Args:
        sentence (str): Sentence.
        tokenizer: BERT tokenizer.
        model: BERT model.
        device: Device to use.

    Returns:
        float: Sentence probability.
    """
    tokens = tokenizer.tokenize(sentence)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor([token_ids]).to(device)

    total_prob = 1.0

    for i in range(1, len(tokens) - 1):  # BOS and EOS are excluded
        masked_input_ids = input_ids.clone()
        masked_input_ids[0, i] = tokenizer.mask_token_id

        with torch.no_grad():
            outputs = model(masked_input_ids)
            logits = outputs.logits

        # Get the probability of the correct token
        correct_token_id = input_ids[0, i]
        token_prob = torch.softmax(logits[0, i], dim=-1)[correct_token_id].item()
        total_prob *= token_prob

    return total_prob


def process_sentences(pos_sent_file, neg_sent_file, tokenizer, model, device):
    """
    Process sentences in a file and aggregate results.

    Args:
        pos_sent_file (str): Path to file containing positive sentences.
        neg_sent_file (str): Path to file containing negative sentences.
        tokenizer: BERT tokenizer.
        model: BERT model.
        device: Device to use.

    Returns:
        tuple: (correct_sents, incorrect_sents) tuple.
    """
    correct_sents = []
    incorrect_sents = []

    # Open file and process line by line
    with open(pos_sent_file, "r") as f_pos, open(neg_sent_file, "r") as f_neg:
        for pos_sent, neg_sent in tqdm(zip(f_pos, f_neg)):
            pos_sent = pos_sent.strip()
            neg_sent = neg_sent.strip()
            pos_sent = "[CLS] " + pos_sent + " [SEP]"
            neg_sent = "[CLS] " + neg_sent + " [SEP]"

            pos_prob = calculate_sentence_probability(
                pos_sent, tokenizer, model, device
            )
            neg_prob = calculate_sentence_probability(
                neg_sent, tokenizer, model, device
            )

            if pos_prob > neg_prob:
                correct_sents.append(pos_sent)
            else:
                incorrect_sents.append(pos_sent)

    return correct_sents, incorrect_sents


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def flatten_model_outputs(model_outputs):
    flat_model_outputs = []
    for obs in model_outputs:
        flat_model_outputs.extend(obs)
    return flat_model_outputs


def main():
    model_sizes = ["base", "large"]
    data_prefixes = [
        # "pmb_minimum_filtering_cumulative",
        # "wikitext103_single_clause_0.1",
        "wikitext103_single_clause_0.1_50K_fixed_cumulative"
    ]
    target_test_sets = [
        "simple_svo_fixed",
        "prep_fixed",
        "double_prep_fixed",
        "prep_obj",
    ]

    data_dir = DATA_DIR / "marvin"
    results_dir = PROJECT_DIR / "results"

    for model_size in model_sizes:
        tokenizer, model, device = load_bert_model_and_tokenizer(model_size=model_size)

        for data_prefix in data_prefixes:
            for target_test_set in target_test_sets:
                print(
                    f"data_prefix: {data_prefix}, model_size: {model_size}, target_test_set: {target_test_set}"
                )
                # Specify file path
                pos_sent_file = data_dir / f"{target_test_set}_good.txt"
                neg_sent_file = data_dir / f"{target_test_set}_bad.txt"

                correct_sents, incorrect_sents = process_sentences(
                    pos_sent_file, neg_sent_file, tokenizer, model, device
                )
                print(
                    f"data_prefix: {data_prefix}, model_size: {model_size}, target_test_set: {target_test_set}"
                )
                print(f"correct_sents: {len(correct_sents)}")
                print(f"incorrect_sents: {len(incorrect_sents)}")
                if len(correct_sents) == 0 or len(incorrect_sents) == 0:
                    print(f"No correct or incorrect sentences for {target_test_set}")
                    raise ValueError(
                        f"No correct or incorrect sentences for {target_test_set}"
                    )

                with open(
                    results_dir
                    / f"{data_prefix}_bert{model_size}"
                    / f"{target_test_set}_correct_whole_prob.txt",
                    "w",
                ) as f:
                    for sent in correct_sents:
                        f.write(sent + "\n")
                with open(
                    results_dir
                    / f"{data_prefix}_bert{model_size}"
                    / f"{target_test_set}_incorrect_whole_prob.txt",
                    "w",
                ) as f:
                    for sent in incorrect_sents:
                        f.write(sent + "\n")

                # filter
                result_path = (
                    results_dir
                    / f"{data_prefix}_bert{model_size}"
                    / f"{target_test_set}_good_results_list.pkl"
                )
                correct_path = (
                    results_dir
                    / f"{data_prefix}_bert{model_size}"
                    / f"{target_test_set}_correct_whole_prob.txt"
                )
                incorrect_path = (
                    results_dir
                    / f"{data_prefix}_bert{model_size}"
                    / f"{target_test_set}_incorrect_whole_prob.txt"
                )

                with open(correct_path, "r") as f:
                    correct_sents = f.readlines()
                with open(incorrect_path, "r") as f:
                    incorrect_sents = f.readlines()

                with open(result_path, "rb") as f:
                    results_list = pickle.load(f)

                correct_sents = [
                    sent.replace("[CLS] ", "").replace(" . [SEP]", "").strip()
                    for sent in correct_sents
                ]
                incorrect_sents = [
                    sent.replace("[CLS] ", "").replace(" . [SEP]", "").strip()
                    for sent in incorrect_sents
                ]
                print(f"{len(correct_sents)=}", f"{len(incorrect_sents)=}")
                print(f"{len(results_list[0])=}")
                correct_results_list = []
                incorrect_results_list = []
                for results in results_list:
                    current_correct_results = []
                    current_incorrect_results = []
                    for result in results:
                        if result["sentence"] in correct_sents:
                            current_correct_results.append(result)
                        elif result["sentence"] in incorrect_sents:
                            current_incorrect_results.append(result)
                    correct_results_list.append(current_correct_results)
                    incorrect_results_list.append(current_incorrect_results)
                print(
                    f"{len(correct_results_list[0])=}",
                    f"{len(incorrect_results_list[0])=}",
                )
                print("----------------------")
                with open(
                    results_dir
                    / f"{data_prefix}_bert{model_size}"
                    / f"{target_test_set}_correct_whole_prob_results_list.pkl",
                    "wb",
                ) as f:
                    pickle.dump(correct_results_list, f)
                with open(
                    results_dir
                    / f"{data_prefix}_bert{model_size}"
                    / f"{target_test_set}_incorrect_whole_prob_results_list.pkl",
                    "wb",
                ) as f:
                    pickle.dump(incorrect_results_list, f)

                # filter predictions/observations
                n_layers = 12 if model_size == "base" else 24
                n_seeds = 5
                for layer in range(n_layers):
                    for seed in range(1, n_seeds + 1):
                        predictions_path = (
                            results_dir
                            / f"{data_prefix}_bert{model_size}"
                            / f"layer{layer}"
                            / f"seed{seed}"
                            / f"{target_test_set}_good.predictions"
                        )
                        observations_path = (
                            results_dir
                            / f"{data_prefix}_bert{model_size}"
                            / f"layer{layer}"
                            / f"seed{seed}"
                            / f"{target_test_set}_good.observations"
                        )
                        predictions = read_json(predictions_path)
                        observations = read_json(observations_path)
                        correct_predictions = []
                        correct_observations = []
                        incorrect_predictions = []
                        incorrect_observations = []
                        for batch_pred, batch_obs in zip(predictions, observations):
                            batch_correct_predictions = []
                            batch_correct_observations = []
                            batch_incorrect_predictions = []
                            batch_incorrect_observations = []
                            for pred, obs in zip(batch_pred, batch_obs):
                                sentence = " ".join(obs[1][:-1])
                                if sentence in correct_sents:
                                    batch_correct_predictions.append(pred)
                                    batch_correct_observations.append(obs)
                                elif sentence in incorrect_sents:
                                    batch_incorrect_predictions.append(pred)
                                    batch_incorrect_observations.append(obs)
                            correct_predictions.append(batch_correct_predictions)
                            correct_observations.append(batch_correct_observations)
                            incorrect_predictions.append(batch_incorrect_predictions)
                            incorrect_observations.append(batch_incorrect_observations)

                        with open(
                            results_dir
                            / f"{data_prefix}_bert{model_size}"
                            / f"layer{layer}"
                            / f"seed{seed}"
                            / f"{target_test_set}_correct.predictions",
                            "w",
                        ) as f:
                            json.dump(correct_predictions, f)
                        with open(
                            results_dir
                            / f"{data_prefix}_bert{model_size}"
                            / f"layer{layer}"
                            / f"seed{seed}"
                            / f"{target_test_set}_correct.observations",
                            "w",
                        ) as f:
                            json.dump(correct_observations, f)
                        with open(
                            results_dir
                            / f"{data_prefix}_bert{model_size}"
                            / f"layer{layer}"
                            / f"seed{seed}"
                            / f"{target_test_set}_incorrect.predictions",
                            "w",
                        ) as f:
                            json.dump(incorrect_predictions, f)
                        with open(
                            results_dir
                            / f"{data_prefix}_bert{model_size}"
                            / f"layer{layer}"
                            / f"seed{seed}"
                            / f"{target_test_set}_incorrect.observations",
                            "w",
                        ) as f:
                            json.dump(incorrect_observations, f)
                print(
                    f"{sum([len(batch) for batch in correct_predictions])=}",
                    f"{sum([len(batch) for batch in correct_observations])=}",
                    f"{sum([len(batch) for batch in incorrect_predictions])=}",
                    f"{sum([len(batch) for batch in incorrect_observations])=}",
                )


if __name__ == "__main__":
    main()
