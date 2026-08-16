# Batch gradient to one continuous sequence

This experiment uses `N` real FineWeb-Edu samples of length `S` to compute one
batch-average GPT-2 gradient. It then optimizes only one continuous input/target
pair with shape `[1, S, vocab]` on pretrained GPT-2 small. The optimized pair is
frozen and tested against the same `N`-sample batch gradient on pretrained and
random GPT-2 models.

```bash
source /venv/main/bin/activate
python batch_to_single_transfer_gpt2.py \
  --n 3 --s 512 --steps 30 \
  --models gpt2 gpt2-medium gpt2-large gpt2-xl \
  --weights pretrained random \
  --out batch_to_single_results.json
```

The result reports full-gradient cosine, relative L2 error, and norm ratio for
each model/weight pair. The reference gradient always comes from the `N` real
examples; only the optimized vector has batch dimension 1.
