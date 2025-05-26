import os
import random
import argparse
import logging
from collections import namedtuple
from tqdm import tqdm
import concurrent.futures
import stanza
from datasets import load_dataset, concatenate_datasets
from pprint import pprint
import re
import spacy

# Fix random seed
SEED = 42
random.seed(SEED)

nlp_min = spacy.blank("en")
nlp_min.add_pipe("sentencizer")
# Logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
SECTION_HEADING_PATTERN = re.compile(r"(\n?=+[^=]+=+\n)", re.MULTILINE)

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
nlp = spacy.load("en_core_web_lg")


def parse_input_file(file_path):
    logging.info(f"Parsing input file: {file_path}")
    sentences = []
    with open(file_path, "r") as f:
        content = f.read().strip()
        blocks = content.split("\n\n")  # Split by empty lines

        for block in blocks:
            lines = block.split("\n")
            if len(lines) >= 3:
                sentence_id = lines[0].strip()
                sentence = lines[1].strip()  # This is the sentence to extract
                semantic_info = " ".join(
                    lines[2:]
                ).strip()  # Combine all lines from line 3 onwards
                sentences.append((sentence_id, sentence, semantic_info))
    logging.info(f"Parsed {len(sentences)} sentences from {file_path}")
    return sentences


def split_final_punctuation(tokens):
    """Split punctuation at the end of sentences"""
    result = []
    for token in tokens:
        if token[-1] in ".!?" and len(token) > 1:
            result.extend([token[:-1], token[-1]])
        else:
            result.append(token)
    return result


def create_observation(sentence):
    # Pre-tokenize sentences and split end-of-sentence punctuation
    pretokenized_sentence = split_final_punctuation(sentence.split())

    # Parse with spaCy
    doc = nlp(" ".join(pretokenized_sentence))

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


def process_files(file_paths):
    logging.info(f"Processing {len(file_paths)} files")
    all_sentences = []
    for file_path in tqdm(file_paths, desc="Processing files"):
        sentences = parse_input_file(file_path)
        all_sentences.extend(sentences)
    logging.info(f"Processed a total of {len(all_sentences)} sentences")
    return all_sentences


def create_single_observation(sentence):
    obs_dict = create_observation(sentence)._asdict()
    return obs_dict


def create_observations_parallel(
    sentences, output_dir, max_workers=None, batch_size=1000
):
    if max_workers is None:
        max_workers = os.cpu_count()
    logging.info(f"Creating observations using up to {max_workers} workers")

    total_sentences = len(sentences)
    all_observations = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(0, total_sentences, batch_size):
            batch = sentences[i : i + batch_size]
            futures.append(executor.submit(process_batch, batch, i // batch_size + 1))

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Processing batches",
        ):
            batch_observations = future.result()
            all_observations.extend(batch_observations)

    logging.info(f"Created a total of {len(all_observations)} observations")
    return all_observations


def process_batch(batch, batch_number):
    logging.info(f"Processing batch {batch_number}")
    batch_observations = []
    for sentence in tqdm(
        batch,
        desc=f"Creating observations (batch {batch_number})",
        leave=False,
    ):
        obs_dict = create_single_observation(sentence)
        batch_observations.append(Observation(**obs_dict))
    return batch_observations


def _wikitext_detokenizer(string):
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")

    # head space
    string = re.sub(r"^\s+", "", string)
    return string


def _split_wikitext_into_sections(text: str) -> tuple[list[str], list[str]]:
    sections = re.split(SECTION_HEADING_PATTERN, text)
    section_headings = sections[1::2]
    section_texts = sections[2::2]
    return section_headings, section_texts


def prepare_wikitext(split_text: str) -> tuple[list[str], list[str]]:
    detokenized_split_text = _wikitext_detokenizer(split_text)
    headings, texts = _split_wikitext_into_sections(detokenized_split_text)
    return headings, texts


def split_texts_into_sentences(texts: list[str], nlp_min: spacy.Language) -> list[str]:
    sentences = []
    for text in tqdm(texts, desc="Splitting texts into sentences"):
        paragraphs = text.split("\n")
        for paragraph in paragraphs:
            doc = nlp_min(paragraph)
            for sent in doc.sents:
                sentences.append(sent.text)
    return sentences


def main(args):
    global SEED
    SEED = args.seed
    random.seed(SEED)
    logging.info(f"Random seed set to {SEED}")
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    all_dataset = concatenate_datasets(
        [dataset["train"], dataset["validation"], dataset["test"]]
    )
    print(f"all: {len(all_dataset)}")
    if args.sample_ratio < 1.0:
        sample_size = int(len(all_dataset) * args.sample_ratio)
        all_dataset = all_dataset.shuffle(seed=args.seed).select(range(sample_size))
        print(f"all (after sampling): {len(all_dataset)}")
    if args.test:
        all_text = "".join(all_dataset["text"][:1000])
    else:
        all_text = "".join(all_dataset["text"])

    _, all_texts = prepare_wikitext(all_text)
    all_sentences = split_texts_into_sentences(all_texts, nlp_min)
    all_observations = create_observations_parallel(
        all_sentences, args.output_dir, args.max_workers, args.batch_size
    )

    # Save split datasets
    save_observations_to_conllx(
        all_observations, os.path.join(args.output_dir, f"all.conllx")
    )
    logging.info(f"Processing complete. Total sentences: {len(all_observations)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process wikitext files and convert to CoNLL-X format"
    )
    parser.add_argument(
        "--output_dir", type=str, help="Directory to save output CoNLL-X files"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Maximum number of worker processes",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Batch size for parallel processing",
    )
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--sample_ratio", type=float, default=1.0)

    args = parser.parse_args()

    logging.info("Starting PMB to CoNLL-X conversion process")
    main(args)
    logging.info("PMB to CoNLL-X conversion process completed")
