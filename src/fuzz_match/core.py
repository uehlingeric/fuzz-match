"""Fuzzy matching algorithms: TF-IDF cosine similarity and Levenshtein WRatio."""

import collections
import re
import time
import unicodedata

import pandas as pd
from rapidfuzz.fuzz import WRatio
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn


def _format_time(s):
    """Format seconds to human-readable string."""
    s = float(s)
    if s < 60:
        if s == int(s):
            seconds = int(s)
            return f"{seconds} second{'s' if seconds != 1 else ''}"
        else:
            formatted = f"{s:.2f}"
            return f"{formatted} second{'s' if float(formatted) != 1 else ''}"
    hours = int(s // 3600)
    minutes = int((s % 3600) // 60)
    seconds = int(s % 60)
    parts = []
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if seconds or not parts:
        parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")
    return ", ".join(parts)


def predict_matrix_cosine(n_left, n_right):
    """Predict execution time for matrix_cosine."""
    if n_left < 1000:
        return 2.8e-5 * (n_left / 10) ** 0.14 * n_right
    else:
        return 1.77e-7 * (n_left**0.82) * n_right


def predict_rapid_fuzz(n_left, n_right):
    """Predict execution time for rapid_fuzz_wratio."""
    product = n_left * n_right
    return 5.5e-6 * product


def normalize_text(text):
    """Normalize text: ASCII, lowercase, remove punctuation."""
    text = (
        unicodedata.normalize("NFKD", text)
        .encode("ASCII", "ignore")
        .decode("utf-8", "ignore")
    )
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text


def best_match(name, candidates, skip_100=False):
    """Find best WRatio match for name from candidates."""
    candidate_scores = []
    for c in candidates:
        score = WRatio(name, c)
        candidate_scores.append((c, score))
    if skip_100:
        filtered = [(c, score) for (c, score) in candidate_scores if score != 100]
        if filtered:
            best_candidate, best_score = max(filtered, key=lambda x: x[1])
        else:
            best_candidate, best_score = max(candidate_scores, key=lambda x: x[1])
    else:
        best_candidate, best_score = max(candidate_scores, key=lambda x: x[1])
    if best_candidate is None or best_candidate.strip() == "":
        best_candidate = name
        best_score = 100
    return best_candidate, best_score


def rapid_fuzz_wratio(to_match_list, to_match_to_list, skip_100=False):
    """Match items using RapidFuzz Levenshtein WRatio."""
    n_left = len(to_match_list)
    n_right = len(to_match_to_list)
    pred_time = predict_rapid_fuzz(n_left, n_right)
    print(
        f"Estimated time for rapid_fuzz_wratio with {n_left} left entities "
        f"and {n_right} right entities: {_format_time(pred_time)}."
    )
    t0 = time.time()
    results = []
    for name in to_match_list:
        m, s_ = best_match(name, to_match_to_list, skip_100=skip_100)
        if m is None or m.strip() == "":
            m = name
            s_ = 100
        results.append({"name": name, "matched_name": m, "score": round(s_, 2)})
    t1 = time.time()
    print(
        f"Rapid Fuzz WRatio finished with {n_left} left entities and "
        f"{n_right} right entities in {_format_time(t1 - t0)}."
    )
    return pd.DataFrame(results)


def awesome_cossim_top(A, B, ntop, lower_bound=0.0):
    """Calculate sparse dot product, returning top N results."""
    return sp_matmul_topn(A, B, ntop, lower_bound)


def matrix_cosine(to_match_list, to_match_to_list, topn=1, threshold=-1, skip_100=False):
    """Match items using TF-IDF vectorization and cosine similarity."""
    n_left = len(to_match_list)
    n_right = len(to_match_to_list)
    pred_time = predict_matrix_cosine(n_left, n_right)
    print(
        f"Estimated time for matrix_cosine with {n_left} left entities and "
        f"{n_right} right entities: {_format_time(pred_time)}."
    )

    t0 = time.time()
    combined = to_match_list + to_match_to_list
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 3), preprocessor=normalize_text
    )
    tfidf_matrix = vectorizer.fit_transform(combined)
    left_matrix = tfidf_matrix[:n_left]
    right_matrix = tfidf_matrix[n_left:]

    topn_val = topn if not skip_100 else max(min(n_right, 5), topn + 2)

    matches = awesome_cossim_top(left_matrix, right_matrix.T, topn_val, threshold)
    rows = matches.tocoo().row
    cols = matches.tocoo().col
    vals = matches.tocoo().data
    groups = collections.defaultdict(list)

    for i, j, v in zip(rows, cols, vals):
        score_value = round(v * 100, 2)
        groups[i].append((j, score_value))

    out = []
    for i in range(n_left):
        candidate_list = groups.get(i, [])

        if skip_100:
            normalized_current = normalize_text(to_match_list[i])

            filtered = [
                (j, score)
                for (j, score) in candidate_list
                if normalize_text(to_match_to_list[j]) != normalized_current
            ]

            if filtered:
                best_j, best_score = max(filtered, key=lambda x: x[1])
            else:
                best_j, best_score = None, 0
        else:
            if candidate_list:
                best_j, best_score = max(candidate_list, key=lambda x: x[1])
            else:
                best_j, best_score = None, 0

        if best_j is None:
            matched_name = to_match_list[i]
        else:
            matched_name = to_match_to_list[best_j]
            if matched_name.strip() == "":
                matched_name = to_match_list[i]

        out.append(
            {"name": to_match_list[i], "matched_name": matched_name, "score": best_score}
        )

    df_out = pd.DataFrame(out)
    t1 = time.time()
    print(
        f"Matrix Cosine finished with {n_left} left entities and {n_right} "
        f"right entities in {_format_time(t1 - t0)}."
    )
    return df_out
