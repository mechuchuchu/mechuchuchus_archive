# GPT-2 full-gradient matching

`gradient_matching_gpt2.py` compares the flattened gradient of **all GPT-2 parameters** for three separate 512-token FineWeb-Edu documents with the gradient produced by continuous tensors of shape `[3, 512, vocab]`.

The continuous input is `softmax(x) @ WTE`; the continuous target is a soft distribution used in cross-entropy. Both are optimized to maximize cosine similarity to the real-data gradient. Results and the complete learning curve are written to JSON.

```bash
source /venv/main/bin/activate
python gradient_matching_gpt2.py --models gpt2 gpt2-medium --weights pretrained random --batch-size 3 --seq-len 512 --steps 30 --out results.json
```

For a smoke test use `--models gpt2 --weights pretrained --steps 2`. `random` keeps the Hugging Face GPT-2 architecture/configuration but reinitializes all weights. Larger models need substantially more GPU memory because second-order autograd is required.

Interpretation: cosine near 1 and relative L2 near 0 indicate matching of the full gradient vector. Compare final metrics across sizes and weight types; `history` contains the optimization trajectory.
