import time
import re
import unicodedata
import pandas as pd
import argparse
import os
import sys
import collections
from rapidfuzz.fuzz import WRatio
from sparse_dot_topn import sp_matmul_topn
from sklearn.feature_extraction.text import TfidfVectorizer


def _format_time(s):
    """
    Format a time in seconds to a human-readable string.
    
    Args:
        s: Time in seconds.
        
    Returns:
        Formatted time string.
    """
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
    return ', '.join(parts)


def predict_matrix_cosine(n_left, n_right):
    """
    Predict execution time for matrix_cosine algorithm.
    
    Args:
        n_left: Number of items in the first list.
        n_right: Number of items in the second list.
        
    Returns:
        Predicted execution time in seconds.
    """
    if n_left < 1000:
        return 2.8e-5 * (n_left / 10) ** 0.14 * n_right
    else:
        return 1.77e-7 * (n_left**0.82) * n_right


def predict_rapid_fuzz(n_left, n_right):
    """
    Predict execution time for rapid_fuzz_wratio algorithm.
    
    Args:
        n_left: Number of items in the first list.
        n_right: Number of items in the second list.
        
    Returns:
        Predicted execution time in seconds.
    """
    product = n_left * n_right
    time_seconds = 5.5e-6 * product
    return time_seconds


def normalize_text(text):
    """
    Normalize text by converting to ASCII, lowercase, and removing punctuation.
    
    Args:
        text: Input text string.
        
    Returns:
        Normalized text string.
    """
    text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8", "ignore")
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text


def best_match(name, candidates, skip_100=False):
    """
    Find the best match for a name from a list of candidates using WRatio.
    
    Args:
        name: Name to match.
        candidates: List of candidate strings.
        skip_100: Whether to skip exact matches (score 100).
        
    Returns:
        Tuple of (best_candidate, best_score).
    """
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
    """
    Match items using the RapidFuzz WRatio algorithm.
    
    Args:
        to_match_list: List of items to match.
        to_match_to_list: List of items to match against.
        skip_100: Whether to skip exact matches (score 100).
        
    Returns:
        DataFrame with matching results.
    """
    n_left = len(to_match_list)
    n_right = len(to_match_to_list)
    pred_time_rapid = predict_rapid_fuzz(n_left, n_right)
    print(f"Estimated time for rapid_fuzz_wratio with {n_left} left entities and {n_right} right entities is {_format_time(pred_time_rapid)}.")
    t0 = time.time()
    results = []
    for name in to_match_list:
        m, s_ = best_match(name, to_match_to_list, skip_100=skip_100)
        if m is None or m.strip() == "":
            m = name
            s_ = 100
        results.append({"name": name, "matched_name": m, "score": round(s_, 2)})
    t1 = time.time()
    print(f"Rapid Fuzz WRatio Method with {n_left} left entities and {n_right} right entities finished in {_format_time(t1 - t0)}.")
    return pd.DataFrame(results)


def awesome_cossim_top(A, B, ntop, lower_bound=0.0):
    """
    Calculate sparse dot product between matrices A and B, returning only top N results.
    
    Args:
        A: First sparse matrix.
        B: Second sparse matrix (transposed).
        ntop: Number of top results to return.
        lower_bound: Minimum similarity threshold.
        
    Returns:
        Sparse matrix of top similarities.
    """
    return sp_matmul_topn(A, B, ntop, lower_bound)


def matrix_cosine(to_match_list, to_match_to_list, topn=1, threshold=-1, skip_100=False):
    """
    Match items using TF-IDF vectorization and cosine similarity.
    
    Args:
        to_match_list: List of items to match.
        to_match_to_list: List of items to match against.
        topn: Number of top matches to consider.
        threshold: Minimum similarity threshold.
        skip_100: Whether to skip self-matches.
        
    Returns:
        DataFrame with matching results.
    """
    n_left = len(to_match_list)
    n_right = len(to_match_to_list)
    pred_time_matrix = predict_matrix_cosine(n_left, n_right)
    print(f"Estimated time for matrix_cosine with {n_left} left entities and {n_right} right entities is {_format_time(pred_time_matrix)}.")
    
    t0 = time.time()
    combined = to_match_list + to_match_to_list
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), preprocessor=normalize_text)
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
        
        out.append({"name": to_match_list[i], "matched_name": matched_name, "score": best_score})
    
    df_out = pd.DataFrame(out)
    t1 = time.time()
    print(f"Sparse Matrix Multiplication Method with {n_left} left entities and {n_right} right entities finished in {_format_time(t1 - t0)}.")
    return df_out


def main():
    """
    Main function to parse arguments and execute the fuzzy matching.
    """
    parser = argparse.ArgumentParser(description="Fuzzy matching script using either matrix_cosine (--mc) or rapid_fuzz_wratio (--rf).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--mc', action='store_true', help="Use matrix_cosine matching")
    group.add_argument('--rf', action='store_true', help="Use rapid_fuzz_wratio matching")
    args = parser.parse_args()
    input_dir = "input"
    output_dir = "output"
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.")
        sys.exit(1)
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    if len(files) != 1:
        print("Error: There must be exactly one file in the input directory.")
        sys.exit(1)
    input_file = files[0]
    if not input_file.lower().endswith(".csv"):
        print("Error: The input file must be a CSV file.")
        sys.exit(1)
    input_path = os.path.join(input_dir, input_file)
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error reading the input file: {e}")
        sys.exit(1)
    if len(df.columns) == 1 and "to_match" in df.columns:
        to_match_list = [x for x in df["to_match"].astype(str).tolist() if x.strip() != ""]
        to_match_to_list = to_match_list.copy()
        skip_flag = True
    elif len(df.columns) == 2 and set(df.columns) == {"to_match", "to_match_to"}:
        to_match_list = [x for x in df["to_match"].astype(str).tolist() if x.strip() != ""]
        to_match_to_list = [x for x in df["to_match_to"].astype(str).tolist() if x.strip() != ""]
        skip_flag = False
    else:
        print("Error: CSV file must contain either exactly one column 'to_match' or exactly the columns 'to_match' and 'to_match_to'.")
        sys.exit(1)
    if args.mc:
        result_df = matrix_cosine(to_match_list, to_match_to_list, skip_100=skip_flag)
    elif args.rf:
        result_df = rapid_fuzz_wratio(to_match_list, to_match_to_list, skip_100=skip_flag)
    else:
        print("Error: No matching algorithm selected.")
        sys.exit(1)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, input_file)
    try:
        result_df.to_csv(output_path, index=False)
        print(f"Results written to '{output_path}'")
    except Exception as e:
        print(f"Error writing output file: {e}")
        sys.exit(1)
    try:
        os.remove(input_path)
        print(f"Input file '{input_path}' deleted.")
    except Exception as e:
        print(f"Error deleting input file: {e}")


if __name__ == "__main__":
    main()