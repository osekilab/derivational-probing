import os
import random
import argparse
import logging
from collections import namedtuple
from tqdm import tqdm
import concurrent.futures
import spacy
from pathlib import Path

# Fix random seed
SEED = 42
random.seed(SEED)


# Logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Field name definitions
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
Observation = namedtuple("Observation", fieldnames)

logging.info("Initializing spacy pipeline")
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    logging.warning("Model 'en_core_web_lg' not found. Downloading it now...")
    spacy.cli.download("en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")


def parse_input_file(file_path):
    """Parse input file and return list of stripped lines."""
    logging.info(f"Parsing input file: {file_path}")
    with open(file_path, "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines


def create_observation(sentence):
    """Create an Observation object from a sentence using spaCy parsing."""
    doc = nlp(sentence)

    index = []
    sentence_tokens = []
    lemma_sentence = []
    upos_sentence = []
    xpos_sentence = []
    morph = []
    head_indices = []
    governance_relations = []

    for token in doc:
        index.append(str(token.i + 1))  # Add 1 because spaCy starts from 0
        sentence_tokens.append(token.text)
        lemma_sentence.append(token.lemma_)
        upos_sentence.append(token.pos_)
        xpos_sentence.append(token.tag_)
        morph.append(str(token.morph) if token.morph else "_")
        head_indices.append(str(token.head.i + 1) if token.head.i != token.i else "0")
        governance_relations.append(token.dep_)

    return Observation(
        index=index,
        sentence=sentence_tokens,
        lemma_sentence=lemma_sentence,
        upos_sentence=upos_sentence,
        xpos_sentence=xpos_sentence,
        morph=morph,
        head_indices=head_indices,
        governance_relations=governance_relations,
        secondary_relations=["_" for _ in index],
        extra_info=None,
        embeddings=None,
    )


def save_observations_to_conllx(observations, filepath):
    """Save observations to a CoNLL-X formatted file."""
    logging.info(f"Saving observations to {filepath}")
    with open(filepath, "w", encoding="utf-8") as f:
        for obs in observations:
            for i in range(len(obs.index)):
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
    logging.info(f"Saved {len(observations)} observations to {filepath}")


def create_single_observation(sentence):
    """Create a single observation and return it as a dictionary."""
    obs_dict = create_observation(sentence)._asdict()
    return obs_dict


def create_observations_parallel(
    sents, output_dir, prefix, max_workers=None, batch_size=1000
):
    """Create observations in parallel and save them in batches."""
    if max_workers is None:
        max_workers = os.cpu_count()
    logging.info(f"Creating observations using up to {max_workers} workers")

    total_sents = len(sents)
    all_observations = []

    for i in range(0, total_sents, batch_size):
        batch = sents[i : i + batch_size]
        batch_number = i // batch_size + 1
        total_batches = (total_sents - 1) // batch_size + 1
        logging.info(f"Processing batch {batch_number} of {total_batches}")

        batch_observations = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(create_single_observation, sent) for sent in batch
            ]
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(batch),
                desc=f"Creating observations (batch {batch_number})",
            ):
                obs_dict = future.result()
                batch_observations.append(Observation(**obs_dict))

        # Save batch observations
        batch_filepath = output_dir / f"{prefix}_batch_{batch_number}.conllx"
        save_observations_to_conllx(batch_observations, batch_filepath)

        all_observations.extend(batch_observations)
        logging.info(
            f"Saved batch {batch_number} with {len(batch_observations)} observations to {batch_filepath}"
        )

    logging.info(f"Created and saved a total of {len(all_observations)} observations")
    return all_observations


def main(input_files, output_dir):
    """Main function to process input files and create observations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_file in input_files:
        all_sents = parse_input_file(input_file)
        observations = []
        for sent in tqdm(all_sents, desc=f"Processing {input_file.name}"):
            obs = create_observation(sent)
            observations.append(obs)
        output_file = output_dir / f"{input_file.stem}.conllx"
        save_observations_to_conllx(observations, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert text files to CoNLL-X format using MARVIN")
    parser.add_argument("--input_files", type=str, nargs="+", required=True,
                      help="Input text files to process")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Output directory for CoNLL-X files")
    args = parser.parse_args()

    input_files = [Path(f).resolve() for f in args.input_files]
    output_dir = Path(args.output_dir).resolve()

    logging.info("Starting MARVIN to CoNLL-X conversion process")
    main(input_files, output_dir)
    logging.info("MARVIN to CoNLL-X conversion process completed")
