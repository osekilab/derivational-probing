from pathlib import Path
from collections import defaultdict, namedtuple
import random
import argparse
import json

from structural_probes.filter_data import (
    load_conll_dataset,
    is_valid_observation,
    get_primary_dep_relations,
    split_data,
    check_distribution,
    save_observations_to_conllx,
)
from structural_probes import DATA_DIR

# Set seed value to ensure reproducibility
random.seed(42)

# Settings
TRAIN_RATIO, DEV_RATIO, TEST_RATIO = 0.8, 0.1, 0.1  # Data split ratios
# Definition of Observation class
fieldnames = [
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
]
observation_class = namedtuple("Observation", fieldnames + ["embeddings"])


def process_config(config):
    OUTPUT_PREFIX = config["OUTPUT_PREFIX"]
    ignore_dep_list = config["ignore_dep_list"]
    punctuation_list = config["punctuation_list"]
    max_observation = config.get("max_observation", None)

    # Create output directory
    output_dir = Path(DATA_DIR) / OUTPUT_PREFIX
    output_dir.mkdir(parents=True, exist_ok=True)

    # Path for filtered observation data
    filtered_observations_path = output_dir / "filtered_observations.conllx"

    # Load filtered data if it exists
    if filtered_observations_path.exists():
        print(f"Loading filtered observations from {filtered_observations_path}")
        filtered_observation_count = sum(
            1
            for line in open(filtered_observations_path, "r", encoding="utf-8")
            if line.strip() == ""
        )
        print(f"Filtered observations: {filtered_observation_count}")
    else:
        all_observation_count = 0
        filtered_observation_count = 0

        # Write filtered data to file sequentially
        with open(filtered_observations_path, "w", encoding="utf-8") as out_f:
            for source_file in config["SOURCE_FILES"]:
                print(f"Processing {source_file}")
                source_path = Path(DATA_DIR) / source_file

                with open(source_path, "r", encoding="utf-8") as f:
                    lines = []
                    for line in f:
                        if line.strip() == "":
                            if lines:
                                all_observation_count += 1
                                observation = parse_observation(lines)
                                if is_valid_observation(
                                    observation, ignore_dep_list, punctuation_list
                                ):
                                    filtered_observation_count += 1
                                    out_f.writelines("\n".join(lines))
                                    out_f.write("\n\n")
                                lines = []
                        else:
                            lines.append(line.strip())
                    # Process last observation data
                    if lines:
                        all_observation_count += 1
                        observation = parse_observation(lines)
                        if is_valid_observation(
                            observation, ignore_dep_list, punctuation_list
                        ):
                            filtered_observation_count += 1
                            out_f.writelines("\n".join(lines))
                            out_f.write("\n\n")
                        lines = []

        print(f"Loaded {all_observation_count} observations in total.")
        print(f"Filtered observations: {filtered_observation_count}")

    # Load observation data into list
    all_observations = []
    for obs in generate_filtered_observations(filtered_observations_path):
        primary_deps = get_primary_dep_relations(obs)
        all_observations.append((primary_deps, obs))

    print(f"Total number of observations: {len(all_observations)}")

    # Get observation data classification and statistics
    classified_counts = defaultdict(int)
    for primary_deps, obs in all_observations:
        classified_counts[primary_deps] += 1

    threshold = int(len(all_observations) * config["threshold_ratio"])

    # Keep combinations exceeding threshold
    valid_primary_deps = {k for k, v in classified_counts.items() if v >= threshold}

    # Keep only valid observation data
    valid_observations = [
        obs
        for primary_deps, obs in all_observations
        if primary_deps in valid_primary_deps
    ]
    print(f"Total number of valid observations: {len(valid_observations)}")

    # Apply max_observation
    if max_observation is not None and max_observation < len(valid_observations):
        valid_observations = random.sample(valid_observations, max_observation)
        print(
            f"Randomly selected {max_observation} observations out of {len(valid_observations)}."
        )
    else:
        print(f"Using all {len(valid_observations)} observations")

    # Get observation data classification and statistics
    classified_counts = defaultdict(int)
    for obs in valid_observations:
        primary_deps = get_primary_dep_relations(obs)
        classified_counts[primary_deps] += 1

    print(f"Classification results (count >= {threshold}):")
    for primary_deps in sorted(
        classified_counts.keys(), key=lambda x: classified_counts[x], reverse=True
    ):
        print(f"{primary_deps}: {classified_counts[primary_deps]} observations")

    # Open data split files
    train_file = output_dir / "train.conllx"
    dev_file = output_dir / "dev.conllx"
    test_file = output_dir / "test.conllx"

    # Shuffle observation data
    random.shuffle(valid_observations)
    total = len(valid_observations)
    train_end = int(total * TRAIN_RATIO)
    dev_end = train_end + int(total * DEV_RATIO)

    train_obs = valid_observations[:train_end]
    dev_obs = valid_observations[train_end:dev_end]
    test_obs = valid_observations[dev_end:]

    # Write observation data
    with open(train_file, "w", encoding="utf-8") as train_f:
        for obs in train_obs:
            obs_text = format_observation(obs)
            train_f.write(obs_text)

    with open(dev_file, "w", encoding="utf-8") as dev_f:
        for obs in dev_obs:
            obs_text = format_observation(obs)
            dev_f.write(obs_text)

    with open(test_file, "w", encoding="utf-8") as test_f:
        for obs in test_obs:
            obs_text = format_observation(obs)
            test_f.write(obs_text)

    # Data set sizes
    split_counts = {
        "train": len(train_obs),
        "dev": len(dev_obs),
        "test": len(test_obs),
    }

    print(f"Data processing and saving completed for {OUTPUT_PREFIX}.")

    # Save results to summary.txt
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Source files: {config['SOURCE_FILES']}\n")
        f.write(f"Config: {config}\n")
        if "all_observation_count" in locals():
            f.write(f"Loaded {all_observation_count} observations in total.\n")
        f.write(f"Filtered observations: {filtered_observation_count}\n")
        f.write(f"Classification results (count >= {threshold}):\n")
        for primary_deps in sorted(
            valid_primary_deps, key=lambda x: classified_counts[x], reverse=True
        ):
            f.write(f"{primary_deps}: {classified_counts[primary_deps]} observations\n")
        f.write(f"Observations after applying threshold: {len(valid_observations)}\n")
        f.write(f"Train data: {split_counts['train']}\n")
        f.write(f"Dev data: {split_counts['dev']}\n")
        f.write(f"Test data: {split_counts['test']}\n")


def generate_filtered_observations(file_path):
    """Generate filtered observation data sequentially."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = []
        for line in f:
            if line.strip() == "":
                if lines:
                    observation = parse_observation(lines)
                    yield observation
                    lines = []
            else:
                lines.append(line.strip())
        if lines:
            observation = parse_observation(lines)
            yield observation


def parse_observation(lines):
    """Create Observation object from line data."""
    conllx_lines = [line.strip().split("\t") for line in lines]
    # Extract each field
    fields = list(zip(*conllx_lines))
    # Add embeddings field
    embeddings = [None] * len(conllx_lines)
    # Create Observation object
    observation = observation_class(*fields, embeddings)
    return observation


def format_observation(observation):
    """Convert Observation object to text format."""
    lines = []
    for i in range(len(observation.index)):
        line = [
            observation.index[i],
            observation.sentence[i],
            observation.lemma_sentence[i],
            observation.upos_sentence[i],
            observation.xpos_sentence[i],
            observation.morph[i],
            observation.head_indices[i],
            observation.governance_relations[i],
            observation.secondary_relations[i],
            observation.extra_info[i],
        ]
        lines.append("\t".join(line))
    return "\n".join(lines) + "\n\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=lambda p: Path(p).resolve(), required=True
    )
    args = parser.parse_args()

    # load config
    with open(args.config_path, "r") as f:
        config = json.load(f)

    print(f"Processing configuration: {config['OUTPUT_PREFIX']}")
    process_config(config)
    print("---")


if __name__ == "__main__":
    main()
