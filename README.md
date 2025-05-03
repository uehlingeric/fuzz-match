# Fuzz-Match

A high-performance fuzzy string matching tool that helps you find and score similarities between text strings. This tool provides two different matching algorithms:

1. **Matrix Cosine** - Uses TF-IDF vectorization and cosine similarity for efficient large-scale matching
2. **RapidFuzz WRatio** - Uses the Levenshtein distance-based WRatio algorithm for accurate matching

## Features

- Compare one list against itself or two different lists
- Smart execution time prediction before running matching operations
- Two powerful matching algorithms:
  - Matrix Cosine (faster for large datasets)
  - RapidFuzz WRatio (more accurate for short strings)
- Output scores on a 0-100 scale (higher = better match)
- Simple CSV input/output format
- Automatic handling of Unicode, case sensitivity, and punctuation

## Installation

```bash
# Clone the repository
git clone https://github.com/uehlingeric/fuzz-match.git
cd fuzz-match

# Install dependencies
pip install -r requirements.txt
```

## Usage Options

### Option 1: For Non-Technical Users (Folder-Based Approach)

This approach is simple and requires no programming knowledge. It uses file directories and CSV files to process your data.

#### 1. Prepare your data

Create an `input` directory and place a CSV file inside it. The CSV file must use one of these formats:

**Single column** named `to_match` (will match each item against all others, excluding exact matches):
```
to_match
Apple Inc.
Appel
Apple Computers
...
```

**Two columns** named `to_match` and `to_match_to` (will match each item in the first column against items in the second column):
```
to_match,to_match_to
Apple Inc.,Apple Computers
Microsoft,Microsft Corp
Google,Alphabet
...
```

#### 2. Run the matching

Open a command prompt or terminal in the fuzz-match directory and run one of these commands:

```bash
# Run with Matrix Cosine method (recommended for large datasets)
python fuzz_match.py --mc

# OR

# Run with RapidFuzz WRatio method (recommended for short strings)
python fuzz_match.py --rf
```

#### 3. Get your results

Results will be written to the `output` directory as a CSV file with the same name as your input file. The output format is:

```
name,matched_name,score
Apple Inc.,Apple Computers,92.55
Microsoft,Microsft Corp,88.7
...
```

### Option 2: For Technical Users (API Approach)

If you're a developer who wants to integrate this directly into your Python code, you can import and use the functions directly.

#### Import the functions

```python
from fuzz_match import matrix_cosine, rapid_fuzz_wratio
```

#### Using Matrix Cosine

```python
import pandas as pd
from fuzz_match import matrix_cosine

# Prepare your data
to_match_list = ["Apple Inc.", "Microsoft", "Google"]
to_match_to_list = ["Apple Computers", "Microsft Corp", "Alphabet"]

# Run the matching algorithm
results_df = matrix_cosine(
    to_match_list, 
    to_match_to_list, 
    topn=1,          # Number of top matches to consider
    threshold=-1,    # Minimum similarity threshold (-1 = no threshold)
    skip_100=False   # Whether to skip exact matches
)

# Process your results
print(results_df)
```

#### Using RapidFuzz WRatio

```python
import pandas as pd
from fuzz_match import rapid_fuzz_wratio

# Prepare your data
to_match_list = ["Apple Inc.", "Microsoft", "Google"]
to_match_to_list = ["Apple Computers", "Microsft Corp", "Alphabet"]

# Run the matching algorithm
results_df = rapid_fuzz_wratio(
    to_match_list,
    to_match_to_list,
    skip_100=False   # Whether to skip exact matches
)

# Process your results
print(results_df)
```

#### Handling larger datasets

For large datasets, you can load data from files:

```python
import pandas as pd
from fuzz_match import matrix_cosine

# Load data from CSV files
data1 = pd.read_csv("path/to/first_list.csv")
data2 = pd.read_csv("path/to/second_list.csv")

# Convert to lists
to_match_list = data1["name_column"].tolist()
to_match_to_list = data2["name_column"].tolist()

# Run the matching algorithm
results_df = matrix_cosine(to_match_list, to_match_to_list)

# Save results
results_df.to_csv("path/to/output.csv", index=False)
```

## How it works

### Matrix Cosine Algorithm

This algorithm uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization with character n-grams to convert strings into numerical vectors. It then calculates the cosine similarity between these vectors to determine how similar the strings are.

The implementation uses sparse matrix multiplication for performance, allowing it to efficiently handle large datasets.

#### Best for:
- Large datasets (1,000+ records)
- Long text strings
- Situations where performance is critical

### RapidFuzz WRatio Algorithm

This algorithm uses the Levenshtein distance-based WRatio metric from the RapidFuzz library. It calculates a weighted ratio that combines different aspects of string similarity, which makes it particularly effective for short strings and human-readable text.

#### Best for:
- Smaller datasets
- Short strings like names or titles
- Situations where accuracy is more important than speed

## Algorithm Selection Guide

| Factor | Matrix Cosine | RapidFuzz WRatio |
|--------|--------------|-----------------|
| Dataset Size | Large (1,000+ items) | Small to medium (<1,000 items) |
| String Length | Works well with longer strings | Works best with shorter strings |
| Speed | Faster for large datasets | Faster for small datasets |
| Memory Usage | Lower | Higher |
| Accuracy | Good overall accuracy | Excellent for names and short text |

## Performance

The tool provides time estimates before running each algorithm, helping you decide which approach to use based on your dataset size.

Example performance estimates:
- Matrix Cosine: ~2 seconds for 1,000 x 1,000 comparisons
- RapidFuzz WRatio: ~5.5 seconds for 1,000 x 1,000 comparisons

Actual performance will vary based on hardware and string length.

## License

[MIT License](LICENSE)

## References

This project was inspired by techniques discussed in:
- https://medium.com/trusted-data-science-haleon/fuzzy-matching-at-scale-part-i-4621b0b36ba5

## Author

Eric Uehling