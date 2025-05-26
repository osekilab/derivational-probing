import argparse
from tqdm import tqdm


def postprocess_conllu(text):
    lines = text.split("\n")
    processed_lines = []

    for line in tqdm(lines):
        # Skip lines starting with '#'
        if line.startswith("#"):
            continue

        # Skip lines starting with digit-digit pattern
        if line.strip() and line.strip()[0].isdigit() and "-" in line.split()[0]:
            continue

        processed_lines.append(line)

    return "\n".join(processed_lines)


def main():
    parser = argparse.ArgumentParser(description="Postprocess CoNLL-U files")
    parser.add_argument("input_path", help="Path to the input file")
    parser.add_argument("output_path", help="Path to the output file")
    args = parser.parse_args()

    # Read input file
    print(f"Reading file from {args.input_path}")
    with open(args.input_path, "r", encoding="utf-8") as f:
        text = f.read()
    print("Done reading")
    # Process the text
    processed_text = postprocess_conllu(text)
    print("Done processing")
    # Write output file
    with open(args.output_path, "w", encoding="utf-8") as f:
        f.write(processed_text)

    print(f"Processed file saved to {args.output_path}")
    lens = len(text.split("\n"))
    print("Lines in original file: ", lens)
    lenp = len(processed_text.split("\n"))
    print("Lines in processed file: ", lenp)
    print("Difference: ", lens - lenp)


if __name__ == "__main__":
    main()
