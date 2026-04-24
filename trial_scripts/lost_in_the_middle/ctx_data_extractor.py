import json
import gzip
import sys

def validate_sample(item):
    errors = []

    if "question" not in item or not item["question"]:
        errors.append("Missing question")

    if "answers" not in item or not item["answers"]:
        errors.append("Missing answers")

    if "ctxs" not in item or len(item["ctxs"]) == 0:
        errors.append("Missing ctxs")

    gold_ctxs = [c for c in item.get("ctxs", []) if c.get("isgold", False)]
    if len(gold_ctxs) == 0:
        errors.append("No isgold context found")

    if errors:
        return False, errors

    return True, "No errors"

def stream_dataset(file_path, n=100):
    open_func = gzip.open if file_path.endswith(".gz") else open

    count = 0

    with open_func(file_path, "rt", encoding="utf-8") as f:
        for line in f:

            if count >= n:
                break

            item = json.loads(line)

            ok, info = validate_sample(item)

            if not ok:
                continue

            question = item["question"]
            answers = item["answers"]

            ctxs = item["ctxs"]
            #ctxs = [c for c in item["ctxs"] if c.get("isgold", True)] TESTING ONLY WITH THE FILE CONTAINING ONLY GOLD CONTEXTS
            contexts = [
                f"[DOC {i+1}]\n{c.get('text', '')}"
                for i, c in enumerate(ctxs)
            ]

            full_context = "\n\n".join(contexts)

            yield {
                "question": question,
                "answers": answers,
                "full_context": full_context
            }

            count += 1

if __name__ == "__main__":
    file_path = sys.argv[1]

    for sample in stream_dataset(file_path, n=100):
        # print JSON in one line (for easier bash processing)
        print(json.dumps(sample, ensure_ascii=False))