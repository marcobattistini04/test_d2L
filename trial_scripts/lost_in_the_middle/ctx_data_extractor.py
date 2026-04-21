import json
import gzip
import sys

def stream_dataset(file_path, n=10):
    open_func = gzip.open if file_path.endswith(".gz") else open

    with open_func(file_path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                return

            item = json.loads(line)

            question = item.get("question")
            
            answers = item.get("answers", [])

            ctxs = item.get("ctxs", [])
            contexts = [c.get("text") for c in ctxs]
            full_context = "\n".join(contexts)

            yield {
                "question": question,
                "answers": answers,
                "full_context": full_context
            }

if __name__ == "__main__":
    file_path = sys.argv[1]

    for sample in stream_dataset(file_path, n=10):
        # print JSON in one line (for easier bash processing)
        print(json.dumps(sample, ensure_ascii=False))