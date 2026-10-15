# Soft Prompt Memorization and Slot Transfer

## Setup

- **Model:** `IFM/K2-Horizon-0.9B`
- **Target sentence:** “At dusk, the paper moon drifted over the quiet harbor, carrying a small blue secret.”
- **Soft prompt:** 16 trainable continuous vectors; all model weights remained frozen.
- **Training objective:** predict the target sentence as the assistant response to “Recall exactly the sentence encoded by this learned prompt.”
- **Training result:** exact greedy recall succeeded in the training context. The final token loss was **0.0082** at step **242**.

The learned vectors were then inserted at `X` in six questions. Probes used the model's low-effort reasoning chat prefix, while training used a direct assistant-answer prefix. The generation limit was 512 tokens. A probe is marked as having no final answer when the model did not produce its answer-channel closing marker within that limit; internal reasoning text is omitted.

## Results

| Probe | Soft prompt at `X` | Literal sentence control |
|---|---|---|
| What is X? | No final answer | Repeated the sentence |
| Translate X into Korean. | No final answer | Repeated the English sentence without translating it |
| What feeling does X convey? | No final answer | Repeated the sentence |
| X는 뭐야? | No final answer | Repeated the sentence |
| X를 한국어로 번역해줘. | No final answer | Repetitive, incoherent Korean output |
| 이 X는 어떤 느낌이야? | No final answer | Repeated the sentence |

## Takeaway

The soft prompt reliably stored the target under its training context, but did not transfer as a usable phrase when moved into the middle of these questions. None of the six slot probes produced a completed answer within the generation limit. The literal-text controls also showed weak task following, especially for translation, so these results demonstrate this setup's transfer failure rather than a general capability comparison.

The training and probe chat prefixes differed, which is a limitation of this run. The experiment is reproducible with [`soft_prompt_probe.py`](./soft_prompt_probe.py); the trained vectors are in [`soft_prompt.pt`](./soft_prompt.pt), and the full outputs are in [`soft_prompt_probe_results.json`](./soft_prompt_probe_results.json).
