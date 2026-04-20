import gzip
import json
import argparse
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument("--file-path", type=str, required=True)
args = parser.parse_args()

file_path = args.file_path

with gzip.open(file_path, "rt") as f:
    first_line = next(f)
    example = json.loads(first_line)

    pprint({
        "question": example["question"],
        "answers": example["answers"],
        "num_ctxs": len(example["ctxs"])
    })

    print("\n--- GOLD POSITION CHECK ---")
    for i, doc in enumerate(example["ctxs"]):
        if doc.get("isgold"):
            print("Gold at index:", i)