"""Command-line interface for fuzzy matching."""

import argparse
import os
import sys

import pandas as pd

from .core import matrix_cosine, rapid_fuzz_wratio


def main():
    """CLI entry point for fuzzy matching."""
    parser = argparse.ArgumentParser(
        description="Fuzzy match records using TF-IDF cosine or Levenshtein WRatio."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mc", action="store_true", help="Use TF-IDF matrix cosine")
    group.add_argument("--rf", action="store_true", help="Use Levenshtein WRatio")
    args = parser.parse_args()

    input_dir = "input"
    output_dir = "output"

    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.")
        sys.exit(1)

    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    if len(files) != 1:
        print("Error: Exactly one CSV file required in input directory.")
        sys.exit(1)

    input_file = files[0]
    if not input_file.lower().endswith(".csv"):
        print("Error: Input file must be CSV.")
        sys.exit(1)

    input_path = os.path.join(input_dir, input_file)
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)

    # Parse CSV format
    if len(df.columns) == 1 and "to_match" in df.columns:
        to_match_list = [x for x in df["to_match"].astype(str).tolist() if x.strip()]
        to_match_to_list = to_match_list.copy()
        skip_flag = True
    elif len(df.columns) == 2 and set(df.columns) == {"to_match", "to_match_to"}:
        to_match_list = [x for x in df["to_match"].astype(str).tolist() if x.strip()]
        to_match_to_list = [
            x for x in df["to_match_to"].astype(str).tolist() if x.strip()
        ]
        skip_flag = False
    else:
        print("Error: CSV must have 'to_match' column or 'to_match'+'to_match_to' columns.")
        sys.exit(1)

    # Run matching
    if args.mc:
        result_df = matrix_cosine(to_match_list, to_match_to_list, skip_100=skip_flag)
    elif args.rf:
        result_df = rapid_fuzz_wratio(to_match_list, to_match_to_list, skip_100=skip_flag)

    # Write output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, input_file)
    try:
        result_df.to_csv(output_path, index=False)
        print(f"Results written to '{output_path}'")
    except Exception as e:
        print(f"Error writing output file: {e}")
        sys.exit(1)

    # Clean up input
    try:
        os.remove(input_path)
        print(f"Input file deleted.")
    except Exception as e:
        print(f"Error deleting input file: {e}")


if __name__ == "__main__":
    main()
