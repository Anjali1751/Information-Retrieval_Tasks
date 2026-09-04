# Multilingual Information Retrieval

A small offline preprocessing pipeline for English, Hindi, and Marathi text.
It tokenizes each document, removes language-specific stopwords, applies simple
language-appropriate normalization, and reports token statistics.

## Run

Requires Python 3.10 or newer. No third-party packages are required.

```bash
python multilingual_ir.py
python multilingual_ir.py --json results.json
```

The original notebook, `Multilingual__IR_1.ipynb`, demonstrates the same
pipeline interactively and imports the reusable module rather than installing
packages or downloading language models at runtime.