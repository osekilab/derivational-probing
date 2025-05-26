import json
import sys
from structural_probes.task import ParseDistanceTask
from collections import namedtuple, defaultdict
from scipy.stats import spearmanr, pearsonr
import numpy as np
import concurrent.futures
from collections import defaultdict
from pathlib import Path
import pickle
import numpy as np
from tqdm import tqdm
import os
import argparse

PROJECT_DIR = Path(__file__).resolve().parent.parent
MAX_WORKERS = os.cpu_count()


field_names = [
    "index",
    "sentence",
    "lemma_sentence",
    "upos_sentence",
    "xpos_sentence",
    "morph",
    "head_indices",
    "governance_relations",
    "secondary_relations",
    "extra_info",
    "embeddings",
]
OBSERVATION_CLASS = namedtuple("Observation", field_names)


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def extract_mean_loss_for_pairs(pair_results, pair_indices):
    def __is_same_pair(pair1, pair2):
        return (pair1[0] == pair2[0] and pair1[1] == pair2[1]) or (
            pair1[0] == pair2[1] and pair1[1] == pair2[0]
        )

    extracted_elems = [
        elem
        for elem in pair_results
        if any(
            __is_same_pair(elem["pair_indices"], pair_index)
            for pair_index in pair_indices
        )
    ]
    return (
        np.mean([elem["loss"] for elem in extracted_elems])
        if len(extracted_elems) > 0
        else None
    )


def get_descendants(index, head_indices):
    descendants = []
    for idx, head in enumerate(head_indices):
        if head == index:
            descendants.append(idx)
            descendants.extend(get_descendants(idx, head_indices))
    return descendants


def get_structure_indices(
    root_index, head_indices, governance_relations, tokens, include_descendants
):
    # Find primary dependencies (children of the root)
    primary_dep_indices = [
        idx for idx, head_index in enumerate(head_indices) if head_index == root_index
    ]
    # Map primary dependencies to their relations
    relation_count = defaultdict(int)
    primary_dep_to_indices = {}
    for primary_dep_index in primary_dep_indices:
        relation = governance_relations[primary_dep_index]
        relation_count[relation] += 1
        if relation_count[relation] > 1:
            relation = f"{relation}_{relation_count[relation]}"

        # Depending on the flag, include all descendants or only direct children
        if include_descendants:
            indices = get_descendants(primary_dep_index, head_indices) + [
                primary_dep_index
            ]
        else:
            # Include only the primary dependency and its direct children
            indices = [primary_dep_index] + [
                idx
                for idx, head in enumerate(head_indices)
                if head == primary_dep_index
            ]
        primary_dep_to_indices[relation] = indices

    # Root structure includes root and its direct children
    root_structure_indices = [root_index] + primary_dep_indices
    root_to_indices = {"root": root_structure_indices}

    # Combine structures
    structure_to_indices = {**primary_dep_to_indices, **root_to_indices}
    structure_to_tokens = {
        k: [tokens[idx] for idx in v] for k, v in structure_to_indices.items()
    }

    return structure_to_indices, structure_to_tokens


def prims_matrix_to_edges(matrix, words, poses):
    """
    Constructs a minimum spanning tree from the pairwise weights in matrix; returns the edges.
    """
    pairs_to_distances = {}
    uf = UnionFind(len(matrix))
    for i_index, line in enumerate(matrix):
        for j_index, dist in enumerate(line):
            pairs_to_distances[(i_index, j_index)] = dist
    edges = []
    for (i_index, j_index), distance in sorted(
        pairs_to_distances.items(), key=lambda x: x[1]
    ):
        if uf.find(i_index) != uf.find(j_index):
            uf.union(i_index, j_index)
            edges.append((i_index, j_index))
    return edges


class UnionFind:
    """
    Naive UnionFind implementation for Prim's MST algorithm
    """

    def __init__(self, n):
        self.parents = list(range(n))

    def union(self, i, j):
        if self.find(i) != self.find(j):
            i_parent = self.find(i)
            self.parents[i_parent] = j

    def find(self, i):
        i_parent = i
        while True:
            if i_parent != self.parents[i_parent]:
                i_parent = self.parents[i_parent]
            else:
                break
        return i_parent


def compute_uuas(gold_edges, pred_edges):
    """
    Computes the UUAS given gold and predicted edges.
    """
    gold_edge_set = set([frozenset(edge) for edge in gold_edges])
    pred_edge_set = set([frozenset(edge) for edge in pred_edges])
    if len(gold_edge_set) == 0:
        return None  # Cannot compute UUAS without gold edges
    correct = len(gold_edge_set & pred_edge_set)
    total = len(gold_edge_set)
    uuas = correct / total
    return uuas


def compute_metrics(prediction_path, observation_path, include_descendants=True):
    # Load predictions and observations
    predictions = read_json(prediction_path)
    observations = read_json(observation_path)
    # Flatten nested lists
    observations = [obs for sublist in observations for obs in sublist]
    predictions = [pred for sublist in predictions for pred in sublist]

    all_observation_results = []

    for i in range(len(observations)):
        observation = OBSERVATION_CLASS(*observations[i], embeddings=None)
        labels = ParseDistanceTask.labels(observation).numpy()[
            :-1, :-1
        ]  # Ignoring period
        prediction = np.array(predictions[i])[:-1, :-1]  # Ignoring period
        tokens = observation.sentence[:-1]  # Ignoring period
        governance_relations = observation.governance_relations[:-1]  # Ignoring period
        head_indices = [
            int(idx) - 1 for idx in observation.head_indices[:-1]
        ]  # Ignoring period and converting to 0-based index

        # Find root index
        root_indices = [
            idx for idx, head_index in enumerate(head_indices) if head_index == -1
        ]
        if len(root_indices) != 1:
            continue  # Skip observations without a single root
        root_index = root_indices[0]

        # Get structure indices and tokens
        structure_to_indices, structure_to_tokens = get_structure_indices(
            root_index, head_indices, governance_relations, tokens, include_descendants
        )

        # Check if any structure has only one token
        has_invalid_structure = any(len(v) <= 1 for v in structure_to_indices.values())
        if has_invalid_structure:
            print(
                f"Skipping observation due to invalid structure: {observation.sentence}"
            )
            continue  # Skip the observation

        # calculate global edges
        global_gold_edges = prims_matrix_to_edges(labels, tokens, ["X"] * len(tokens))
        global_pred_edges = prims_matrix_to_edges(
            prediction, tokens, ["X"] * len(tokens)
        )
        global_uuas = compute_uuas(global_gold_edges, global_pred_edges)

        # calculate metrics for each structure
        metrics_by_structure = {}
        for dep_name, indices in structure_to_indices.items():
            indices = sorted(set(indices))  # Ensure indices are unique and sorted
            span_labels = labels[np.ix_(indices, indices)]
            span_prediction = prediction[np.ix_(indices, indices)]
            triu_indices = np.triu_indices_from(span_labels, k=1)
            if len(triu_indices[0]) == 0:
                continue

            # Mean L1 Loss
            l1_loss = np.mean(
                np.abs(span_labels[triu_indices] - span_prediction[triu_indices])
            )

            # Spearman Correlation
            spearman_corr, _ = spearmanr(
                span_labels[triu_indices], span_prediction[triu_indices]
            )
            if np.isnan(spearman_corr):
                spearman_corr = None  # Handle NaN cases

            # Calculate UUAS
            # Calculate metrics from global_gold_edges and global_pred_edges
            filtered_gold_edges = [
                edge
                for edge in global_gold_edges
                if edge[0] in indices and edge[1] in indices
            ]
            filtered_pred_edges = [
                edge
                for edge in global_pred_edges
                if edge[0] in indices and edge[1] in indices
            ]

            # UUAS computation
            if filtered_gold_edges:
                uuas = compute_uuas(filtered_gold_edges, filtered_pred_edges)
            else:
                uuas = None  # Cannot compute UUAS without gold edges

            # UUAS
            words_in_span = [tokens[idx] for idx in indices]
            poses_in_span = ["X"] * len(indices)  # Placeholder POS tags
            gold_edges = prims_matrix_to_edges(
                span_labels, words_in_span, poses_in_span
            )
            pred_edges = prims_matrix_to_edges(
                span_prediction, words_in_span, poses_in_span
            )
            uuas_old = compute_uuas(gold_edges, pred_edges)

            metrics_by_structure[dep_name] = {
                "loss": l1_loss,
                "spearman": spearman_corr,
                "uuas": uuas,
                "uuas_old": uuas_old,
            }

        if not metrics_by_structure:
            continue

        all_observation_results.append(
            {
                "observation_index": i,
                "sentence": " ".join(tokens),
                "metrics_by_structure": metrics_by_structure,
                "structure_to_tokens": structure_to_tokens,
                "global_uuas": global_uuas,
            }
        )

    return all_observation_results


def process_experiment(experiment_name, include_descendants, target_test_set):
    n_layers = 24 if experiment_name.endswith(("bertlarge", "gpt2medium")) else 12
    layer_seed_infos = [
        (i, j, experiment_name, PROJECT_DIR, include_descendants, target_test_set)
        for i in range(n_layers)
        for j in range(1, 4)
    ]

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(
            tqdm(
                executor.map(process_layer_seed, layer_seed_infos),
                total=len(layer_seed_infos),
                desc=f"Processing {experiment_name}",
            )
        )

    results_by_layer = defaultdict(list)
    for i, j, layer_results in results:
        results_by_layer[i].append((i, j, layer_results))

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        print(f"Aggregating results for each layer in {experiment_name}...")
        aggregated_results = list(
            tqdm(
                executor.map(aggregate_layer_results, results_by_layer.items()),
                total=len(results_by_layer),
            )
        )

    results_list = [result for _, result in sorted(aggregated_results)]

    with open(
        PROJECT_DIR
        / f"results/{experiment_name}/{target_test_set}_results_list{'' if include_descendants else '_no_descendants'}.pkl",
        "wb",
    ) as f:
        pickle.dump(results_list, f)

    print(f"Results saved for {experiment_name}")


def process_layer_seed(layer_seed_info):
    i, j, experiment_name, project_dir, include_descendants, target_test_set = (
        layer_seed_info
    )
    prediction_path = (
        project_dir
        / f"results/{experiment_name}/layer{i}/seed{j}/{target_test_set}.predictions"
    )
    observation_path = (
        project_dir
        / f"results/{experiment_name}/layer{i}/seed{j}/{target_test_set}.observations"
    )
    results = compute_metrics(prediction_path, observation_path, include_descendants)
    return i, j, results


def aggregate_layer_results(layer_info):
    layer_index, layer_results = layer_info
    aggregated_structure_results = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    sentences = {}
    structure_to_tokens = {}

    for _, _, seed_results in layer_results:
        for obs in seed_results:
            obs_index = obs["observation_index"]
            sentences[obs_index] = obs["sentence"]
            structure_to_tokens[obs_index] = obs["structure_to_tokens"]

            for structure, metrics in obs["metrics_by_structure"].items():
                for metric_name, value in metrics.items():
                    if value is not None:
                        aggregated_structure_results[obs_index][structure][
                            metric_name
                        ].append(value)

            # global uuas
            aggregated_structure_results[obs_index]["global"]["uuas"].append(
                obs["global_uuas"]
            )

    layer_result = []
    for obs_index in aggregated_structure_results:
        mean_metrics_by_structure = {}
        global_uuas = np.mean(aggregated_structure_results[obs_index]["global"]["uuas"])
        global_uuas_std = np.std(
            aggregated_structure_results[obs_index]["global"]["uuas"]
        )
        for structure, metrics in aggregated_structure_results[obs_index].items():
            if structure != "global":
                mean_metrics = {}
                for metric_name, values in metrics.items():
                    mean_metrics[metric_name] = np.mean(values) if values else None
                mean_metrics_by_structure[structure] = mean_metrics

        layer_result.append(
            {
                "observation_index": obs_index,
                "sentence": sentences[obs_index],
                "mean_metrics_by_structure": mean_metrics_by_structure,
                "structure_to_tokens": structure_to_tokens[obs_index],
                "global_uuas": global_uuas,
                "global_uuas_std": global_uuas_std,
            }
        )

    return layer_index, layer_result


def main(experiment_prefix, models, include_descendants, target_test_set):
    experiment_names = [f"{experiment_prefix}_{model}" for model in models]

    for experiment_name in experiment_names:
        process_experiment(experiment_name, include_descendants, target_test_set)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate results for syntactic probing experiments"
    )
    parser.add_argument(
        "--experiment_prefix", type=str, help="Prefix for the experiment names"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt2small", "gpt2medium", "bertbase", "bertlarge"],
        help="List of models to analyze",
    )
    parser.add_argument(
        "--include_descendants",
        action="store_true",
        help="Include descendants in the structure indices",
    )
    parser.add_argument(
        "--target_test_set",
        type=str,
        default="test",
        help="Target test set to analyze",
    )
    args = parser.parse_args()

    main(
        args.experiment_prefix,
        args.models,
        args.include_descendants,
        args.target_test_set,
    )
