# Input Data

fuzz-match accepts CSV files in the `input/` directory.

## CSV Format

**Single-column format** (self-matching):
```
to_match
Apple Inc.
Appel Inc
...
```

**Two-column format** (cross-matching):
```
to_match,to_match_to
Apple Inc.,Apple Computers
Microsoft,Microsft Corp
...
```

Place exactly one CSV file in `input/` before running the matcher. Results appear in `output/` with the same filename.
