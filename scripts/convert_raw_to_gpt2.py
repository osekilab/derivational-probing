"""
Takes raw text and saves GPT-2 features for that text to disk

Adapted for GPT-2 using the Hugging Face Transformers library


"""

import torch
from transformers import GPT2Tokenizer, GPT2Model
from argparse import ArgumentParser
import h5py
import numpy as np
from tqdm import tqdm

argp = ArgumentParser()
argp.add_argument("input_path")
argp.add_argument("output_path")
argp.add_argument("gpt2_model", help="small or medium")
args = argp.parse_args()

# Load pre-trained model and tokenizer
if args.gpt2_model == "small":
    model_name = "gpt2"
elif args.gpt2_model == "medium":
    model_name = "gpt2-medium"
else:
    raise ValueError(f"Invalid GPT-2 model: {args.gpt2_model}")

print(f"Loading {model_name} model")
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2Model.from_pretrained(model_name)
print(f"Loaded {model_name} model")

# Set model configuration based on the chosen model
config = model.config
LAYER_COUNT = config.n_layer
FEATURE_COUNT = config.n_embd

model.eval()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

with h5py.File(args.output_path, "w") as fout:
    total_lines = sum(1 for _ in open(args.input_path))
    for index, line in tqdm(enumerate(open(args.input_path)), total=total_lines):
        line = line.strip()  # Remove trailing characters
        encoded_input = tokenizer(line, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**encoded_input, output_hidden_states=True)
        hidden_states = [
            states.squeeze(0).cpu() for states in outputs.hidden_states
        ]  # list of (seq_len, feature_count)
        # Create dataset and store the hidden states
        dset = fout.create_dataset(
            str(index), (LAYER_COUNT, hidden_states[0].shape[0], FEATURE_COUNT)
        )
        for layer in range(LAYER_COUNT):
            dset[layer, :, :] = hidden_states[layer].numpy()

print(f"Processed {index + 1} lines and saved to {args.output_path}")
