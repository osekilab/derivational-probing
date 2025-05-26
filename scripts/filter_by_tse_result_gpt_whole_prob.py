import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from structural_probes import DATA_DIR, PROJECT_DIR
import pickle
import json


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_model_and_tokenizer(model_size="small"):
    """
    Loads GPT-2 model and tokenizer.

    Args:
        model_size (str): Model size to use ('small' or 'medium').

    Returns:
        tokenizer: GPT-2 tokenizer.
        model: GPT-2 model.
        device: Device to use (CPU or GPU).
    """
    if model_size == "small":
        model_name = "gpt2"
    elif model_size == "medium":
        model_name = "gpt2-medium"
    else:
        raise ValueError(f"Invalid GPT-2 model: {model_size}")

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    # Device configuration (CPU/GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return tokenizer, model, device


def find_critical_span(pos_tokens, neg_tokens):
    """
    Find the start and end indices of the critical part.

    Args:
        pos_tokens (list): Token list of pos_sent.
        neg_tokens (list): Token list of neg_sent.

    Returns:
        tuple: (start_idx, end_idx_pos, end_idx_neg) of tuple.
    """
    len_pos = len(pos_tokens)
    len_neg = len(neg_tokens)
    max_len = max(len_pos, len_neg)

    start_idx = None
    for i in range(max_len):
        pos_token = pos_tokens[i] if i < len_pos else None
        neg_token = neg_tokens[i] if i < len_neg else None
        if pos_token != neg_token:
            start_idx = i
            break

    if start_idx is None:
        # Completely identical token sequence
        return None

    # Determine the end index of the critical part in pos_tokens
    end_idx_pos = start_idx + 1
    while end_idx_pos < len_pos and not pos_tokens[end_idx_pos].startswith("Ġ"):
        end_idx_pos += 1

    # Determine the end index of the critical part in neg_tokens
    end_idx_neg = start_idx + 1
    while end_idx_neg < len_neg and not neg_tokens[end_idx_neg].startswith("Ġ"):
        end_idx_neg += 1

    return start_idx, end_idx_pos, end_idx_neg


def compute_critical_token_probabilities(pos_sent, neg_sent, tokenizer, model, device):
    """
    Compute the probabilities of the critical part of pos_sent and neg_sent.

    Args:
        pos_sent (str): Positive sentence.
        neg_sent (str): Negative sentence.
        tokenizer: Tokenizer.
        model: GPT-2 model.
        device: Device to use.

    Returns:
        tuple: (pos_prob, neg_prob, is_correct) of tuple.
    """
    pos_tokens = tokenizer.tokenize(pos_sent)
    neg_tokens = tokenizer.tokenize(neg_sent)

    # Convert tokens to IDs
    pos_token_ids = tokenizer.convert_tokens_to_ids(pos_tokens)
    neg_token_ids = tokenizer.convert_tokens_to_ids(neg_tokens)

    # Compute probabilities for pos_token and neg_token

    with torch.no_grad():
        pos_outputs = model(
            input_ids=torch.tensor([pos_token_ids]).to(device),
            labels=torch.tensor([pos_token_ids]).to(device),
        )
        neg_outputs = model(
            input_ids=torch.tensor([neg_token_ids]).to(device),
            labels=torch.tensor([neg_token_ids]).to(device),
        )

    pos_loss = pos_outputs.loss
    neg_loss = neg_outputs.loss

    pos_prob = torch.exp(-pos_loss)
    neg_prob = torch.exp(-neg_loss)

    is_correct = pos_prob > neg_prob
    return pos_prob, neg_prob, is_correct


def process_sentences(pos_sent_file, neg_sent_file, tokenizer, model, device):
    """
    Process sentences in the file and aggregate results.

    Args:
        pos_sent_file (str): Path to file containing positive sentences.
        neg_sent_file (str): Path to file containing negative sentences.
        tokenizer: Tokenizer.
        model: GPT-2 model.
        device: Device to use.

    Returns:
        tuple: (correct_sents, incorrect_sents) of tuple.
    """
    correct_sents = []
    incorrect_sents = []

    # Open the file and process one line at a time
    with open(pos_sent_file, "r") as f_pos, open(neg_sent_file, "r") as f_neg:
        for pos_sent, neg_sent in zip(f_pos, f_neg):
            pos_sent = pos_sent.strip()
            neg_sent = neg_sent.strip()

            pos_sent = "<|endoftext|> " + pos_sent
            neg_sent = "<|endoftext|> " + neg_sent
            pos_prob, neg_prob, is_correct = compute_critical_token_probabilities(
                pos_sent, neg_sent, tokenizer, model, device
            )
            if pos_prob is None or neg_prob is None:
                continue
            if is_correct:
                correct_sents.append(pos_sent)
            else:
                incorrect_sents.append(pos_sent)
    return correct_sents, incorrect_sents


def main():
    model_sizes = ["small", "medium"]
    data_prefixes = ["wikitext103_single_clause_0.1_50K_cumulative"]
    target_test_sets = [
        "prep_fixed",
        "double_prep_fixed",
        "prep_obj",
        "simple_svo_fixed",
    ]

    data_dir = DATA_DIR / "marvin"
    results_dir = PROJECT_DIR / "results"

    for model_size in model_sizes:
        tokenizer, model, device = load_model_and_tokenizer(model_size=model_size)

        for data_prefix in data_prefixes:
            for target_test_set in target_test_sets:
                # Specify file paths
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
                    / f"{data_prefix}_gpt2{model_size}"
                    / f"{target_test_set}_correct_whole_prob.txt",
                    "w",
                ) as f:
                    for sent in correct_sents:
                        f.write(sent + "\n")
                with open(
                    results_dir
                    / f"{data_prefix}_gpt2{model_size}"
                    / f"{target_test_set}_incorrect_whole_prob.txt",
                    "w",
                ) as f:
                    for sent in incorrect_sents:
                        f.write(sent + "\n")

                # filter
                result_path = (
                    results_dir
                    / f"{data_prefix}_gpt2{model_size}"
                    / f"{target_test_set}_good_results_list.pkl"
                )
                correct_path = (
                    results_dir
                    / f"{data_prefix}_gpt2{model_size}"
                    / f"{target_test_set}_correct_whole_prob.txt"
                )
                incorrect_path = (
                    results_dir
                    / f"{data_prefix}_gpt2{model_size}"
                    / f"{target_test_set}_incorrect_whole_prob.txt"
                )

                with open(correct_path, "r") as f:
                    correct_sents = f.readlines()
                with open(incorrect_path, "r") as f:
                    incorrect_sents = f.readlines()

                with open(result_path, "rb") as f:
                    results_list = pickle.load(f)

                correct_sents = [
                    sent.replace("<|endoftext|>", "").replace(" .\n", "").strip()
                    for sent in correct_sents
                ]
                incorrect_sents = [
                    sent.replace("<|endoftext|>", "").replace(" .\n", "").strip()
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
                    / f"{data_prefix}_gpt2{model_size}"
                    / f"{target_test_set}_correct_whole_prob_results_list.pkl",
                    "wb",
                ) as f:
                    pickle.dump(correct_results_list, f)
                with open(
                    results_dir
                    / f"{data_prefix}_gpt2{model_size}"
                    / f"{target_test_set}_incorrect_whole_prob_results_list.pkl",
                    "wb",
                ) as f:
                    pickle.dump(incorrect_results_list, f)


if __name__ == "__main__":
    main()
