from collections import namedtuple, defaultdict
import random
from collections import defaultdict
from pathlib import Path

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
    "embeddings",
]
observation_class = namedtuple("Observation", fieldnames)


def generate_lines_for_sent(lines):
    """Yields batches of lines describing a sentence in conllx.

    Args:
        lines: Each line of a conllx file.
    Yields:
        a list of lines describing a single sentence in conllx.
    """
    buf = []
    for line in lines:
        if line.startswith("#"):
            continue
        if not line.strip():
            if buf:
                yield buf
                buf = []
            else:
                continue
        else:
            buf.append(line.strip())
    if buf:
        yield buf


def load_conll_dataset(filepath):
    """Reads in a conllx file; generates Observation objects

    For each sentence in a conllx file, generates a single Observation
    object.

    Args:
        filepath: the filesystem path to the conll dataset

    Returns:
        A list of Observations
    """
    observations = []
    lines = (x for x in open(filepath))
    for buf in generate_lines_for_sent(lines):
        conllx_lines = []
        for line in buf:
            conllx_lines.append(line.strip().split("\t"))
        embeddings = [None for x in range(len(conllx_lines))]
        observation = observation_class(*zip(*conllx_lines), embeddings)
        observations.append(observation)
    return observations


def has_only_one_root(head_indices):
    root_indices = [
        idx for idx, head_index in enumerate(head_indices) if head_index == -1
    ]
    if len(root_indices) != 1:
        return False
    return True


def does_not_have_invalid_dep_relations(
    primary_dep_relations, ignore_dep_list=["punct", "ccomp", "aux", "dep", "xcomp"]
):
    # Check if primary_dep_relations is included in ignore_dep_list
    for relation in primary_dep_relations:
        if relation in ignore_dep_list:
            return False
    return True


def has_subject(primary_dep_relations):
    # Check if primary_dep_relations contains 'subj'
    for relation in primary_dep_relations:
        if relation.endswith("subj"):
            return True
    return False


def every_primary_has_outgoing_indices(primary_to_outgoing_indices):
    for _, outgoing_indices in primary_to_outgoing_indices.items():
        if len(outgoing_indices) == 0:
            return False
    return True


def is_valid_observation(observation):
    governance_relations = observation.governance_relations[:-1]  # ignoring period
    head_indices = [
        int(idx) - 1 for idx in observation.head_indices[:-1]
    ]  # ignoring period and converting to 0-based index


def tokens_include_punctuations(tokens, punctuation_list):
    for token in tokens:
        if token in punctuation_list:
            return True
    return False


def is_question(sentence):
    return sentence[-1] == "?"


def includes_invalid_dep_relations(
    primary_dep_relations,
    ignore_dep_list=["punct", "ccomp", "aux", "dep", "xcomp", "advcl", "acl", "csubj"],
):
    # Check if primary_dep_relations is included in ignore_dep_list
    for relation in primary_dep_relations:
        if relation in ignore_dep_list:
            return True
    return False


def has_more_than_one_eos(sentence):
    # Invalid if there are two or more periods or question marks
    if sentence.count(".") > 1 or sentence.count("?") > 1:
        return True
    return False


def is_valid_observation(
    observation,
    ignore_dep_list=["punct", "ccomp", "aux", "dep", "xcomp", "advcl", "acl", "csubj"],
    punctuation_list=[",", ".", ":", ";", "-", "--", "...", "``", "''"],
):
    governance_relations = observation.governance_relations[:-1]  # ignoring period
    try:
        head_indices = [
            int(idx) - 1 for idx in observation.head_indices[:-1]
        ]  # ignoring period and converting to 0-based index
    except ValueError:
        print("invalid head_indices:", observation.head_indices)
        print(observation.sentence)
        return False
    tokens = observation.sentence[:-1]

    if includes_invalid_dep_relations(governance_relations, ignore_dep_list):
        return False
    if not has_only_one_root(head_indices):
        return False
    if has_more_than_one_eos(observation.sentence):
        return False
    if tokens_include_punctuations(tokens, punctuation_list):
        return False

    root_index = [
        idx for idx, head_index in enumerate(head_indices) if head_index == -1
    ][0]
    primary_dep_indices = [
        idx for idx, head_index in enumerate(head_indices) if head_index == root_index
    ]
    primary_dep_relations = [governance_relations[idx] for idx in primary_dep_indices]

    if not has_subject(primary_dep_relations):
        return False

    outgoing_indices_from_primary = {
        primary_dep_index: [
            idx
            for idx, head_index in enumerate(head_indices)
            if head_index == primary_dep_index
        ]
        for primary_dep_index in primary_dep_indices
    }
    primary_dep_to_outgoing_indices = {}
    relation_count = defaultdict(int)
    for primary_dep_index in primary_dep_indices:
        relation = governance_relations[primary_dep_index]
        relation_count[relation] += 1
        if relation_count[relation] > 1:
            relation = f"{relation}_{relation_count[relation]}"
        primary_dep_to_outgoing_indices[relation] = outgoing_indices_from_primary[
            primary_dep_index
        ]
    if not every_primary_has_outgoing_indices(primary_dep_to_outgoing_indices):
        return False

    if is_question(observation.sentence):
        return False
    return True


def get_primary_dep_relations(observation):
    governance_relations = observation.governance_relations[:-1]  # ignoring period
    head_indices = [
        int(idx) - 1 for idx in observation.head_indices[:-1]
    ]  # ignoring period and converting to 0-based index
    root_index = [
        idx for idx, head_index in enumerate(head_indices) if head_index == -1
    ][0]
    primary_dep_indices = [
        idx for idx, head_index in enumerate(head_indices) if head_index == root_index
    ]
    primary_dep_relations = []
    relation_count = defaultdict(int)
    for idx in primary_dep_indices:
        relation = governance_relations[idx]
        relation_count[relation] += 1
        if relation_count[relation] > 1:
            relation = f"{relation}_{relation_count[relation]}"
        primary_dep_relations.append(relation)

    if len(primary_dep_relations) != len(set(primary_dep_relations)):
        raise ValueError("primary_dep_relations contains duplicates")
    return frozenset(primary_dep_relations + ["root"])  # Add root and use frozenset


def split_data(observations, train_ratio, dev_ratio, test_ratio):
    random.shuffle(observations)
    n = len(observations)
    train_end = int(n * train_ratio)
    dev_end = int(n * (train_ratio + dev_ratio))
    return (
        observations[:train_end],
        observations[train_end:dev_end],
        observations[dev_end:],
    )


def check_distribution(dataset, name):
    distribution = defaultdict(int)
    for obs in dataset:
        primary_deps = get_primary_dep_relations(obs)
        distribution[primary_deps] += 1

    print(f"\nprimary_dep_relations in {name} set:")
    for primary_deps, count in sorted(
        distribution.items(), key=lambda x: x[1], reverse=True
    )[:5]:
        print(f"{primary_deps}: {count} observations ({count/len(dataset)*100:.2f}%)")


def save_observations_to_conllx(observations, filepath):
    """Saves Observation objects to a CoNLL-X formatted file.

    Args:
        observations: A list of Observation objects.
        filepath: The path where the CoNLL-X file will be saved.
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for obs in observations:
            for i in range(len(obs.index)):
                # CoNLL-X format: ID FORM LEMMA CPOSTAG POSTAG FEATS HEAD DEPREL PHEAD PDEPREL
                line = [
                    obs.index[i],
                    obs.sentence[i],
                    obs.lemma_sentence[i],
                    obs.upos_sentence[i],
                    obs.xpos_sentence[i],
                    obs.morph[i],
                    obs.head_indices[i],
                    obs.governance_relations[i],
                    "_",  # PHEAD (placeholder)
                    "_",  # PDEPREL (placeholder)
                ]
                f.write("\t".join(line) + "\n")
            f.write("\n")  # Empty line between sentences
