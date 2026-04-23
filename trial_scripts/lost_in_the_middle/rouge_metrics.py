import re
from collections import Counter

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()

# ROUGE-N (1, 2, ...)
def get_ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]


def rouge_n(pred, ref, n=1):
    pred_tokens = normalize(pred)
    ref_tokens = normalize(ref)

    pred_ngrams = Counter(get_ngrams(pred_tokens, n))
    ref_ngrams = Counter(get_ngrams(ref_tokens, n))

    overlap = pred_ngrams & ref_ngrams
    overlap_count = sum(overlap.values())

    pred_count = sum(pred_ngrams.values())
    ref_count = sum(ref_ngrams.values())

    if pred_count == 0 or ref_count == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = overlap_count / pred_count
    recall = overlap_count / ref_count

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ROUGE-L
def lcs_length(x, y):
    m, n = len(x), len(y)
    dp = [[0]*(n+1) for _ in range(m+1)]

    for i in range(m):
        for j in range(n):
            if x[i] == y[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])

    return dp[m][n]


def rouge_l(pred, ref):
    pred_tokens = normalize(pred)
    ref_tokens = normalize(ref)

    lcs = lcs_length(pred_tokens, ref_tokens)

    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def rouge_n_multi(pred, gold_list, n=1):
    best = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    for gold in gold_list:
        score = rouge_n(pred, gold, n)
        if score["f1"] > best["f1"]:
            best = score

    return best

def rouge_l_multi(pred, gold_list):
    best = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    for gold in gold_list:
        score = rouge_l(pred, gold)
        if score["f1"] > best["f1"]:
            best = score

    return best

def rouge_scores_multi(pred, gold_list):
    return {
        "ROUGE-1": rouge_n_multi(pred, gold_list, 1),
        "ROUGE-2": rouge_n_multi(pred, gold_list, 2),
        "ROUGE-L": rouge_l_multi(pred, gold_list)
    }