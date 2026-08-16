# Transfer experiment across GPT-2 sizes and weights

`transfer_gradient_matching_gpt2.py` optimizes continuous input/target tensors
only on pretrained GPT-2 small. The tensors have shape `[N, S, vocab]`. They are
then frozen and evaluated on pretrained and randomly initialized GPT-2 models.
For each evaluation, the script compares the full flattened parameter gradient
from the real FineWeb-Edu batch against the gradient induced by the optimized
continuous vectors.

```bash
source /venv/main/bin/activate
python transfer_gradient_matching_gpt2.py \
  --n 3 --s 512 --steps 30 \
  --models gpt2 gpt2-medium gpt2-large gpt2-xl \
  --weights pretrained random \
  --out transfer_results.json
```

The terminal and JSON output report cosine similarity, relative L2 error, and
gradient norm ratio for every `(model, weights)` pair. High cosine and low
relative L2 across models indicate transferability of the continuous gradient
matching solution. Only the source `gpt2` pretrained model receives optimization
updates; all evaluation vectors remain frozen.
