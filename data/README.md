# Data Directory

This directory stores datasets for training and evaluating the miniature GPT causal language model.

## Primary Dataset: Tiny Shakespeare
- **Source**: 40,000 lines of Shakespeare from various plays (Karpathy char-rnn / Hugging Face `tiny_shakespeare`).
- **File size**: ~1.11 MB, 1,115,394 characters.
- **Vocabulary**: 65 unique characters (letters, punctuation, whitespace).
- **Download**: Run `python data/prepare_data.py` to download `input.txt`.
