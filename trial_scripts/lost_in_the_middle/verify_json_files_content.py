import gzip
import json
import argparse

def preview_jsonl_gz(path, max_items=10):

    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_items:
                break

            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"[riga {i}] JSON non valido:")
                print(line[:200])
                continue

            print(f"--- elemento {i} ---")
            print(json.dumps(obj, indent=2, ensure_ascii=False))
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Path al file .jsonl.gz")
    parser.add_argument("--max_items", type=int, default=10, help="Numero di righe da mostrare")
    args = parser.parse_args()

    preview_jsonl_gz(args.path, args.max_items)