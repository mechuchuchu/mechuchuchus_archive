#!/usr/bin/env python3
"""Generate OLMo responses from statistic-matched random soft prompts."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
import time
import hashlib

# 현재 시각(ms)
timestamp_ms = time.time_ns() // 1_000_000

# SHA-256 해시
hash_value = hashlib.sha256(str(timestamp_ms).encode()).hexdigest()

print(timestamp_ms)
print(hash_value)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="allenai/Olmo-3-7B-Instruct")
    p.add_argument("--output", type=Path, default=Path(f"olmo_soft_prompt_outputs_{hash_value}.jsonl"))
    p.add_argument("--num-seeds", type=int, default=20)
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Exact per-prompt seeds to use, e.g. --seeds 11 22 33. "
            "When supplied, their count overrides --num-seeds."
        ),
    )
    p.add_argument("--soft-prompt-length", type=int, default=20)
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of soft prompts generated together on the GPU.",
    )
    p.add_argument(
        "--master-seed",
        type=int,
        default=None,
        help=(
            "Seed used to choose the per-prompt seeds. If omitted, seeds are "
            "chosen from the OS cryptographic random source."
        ),
    )
    p.add_argument(
        "--prompt",
        default="",
        help="Optional user text appended after the random soft prompt.",
    )
    p.add_argument(
        "--no-save-embeddings",
        action="store_true",
        help="Do not store the full soft-prompt matrix in JSONL.",
    )
    return p.parse_args()


def tensor_stats(x: torch.Tensor) -> dict[str, float]:
    x = x.float()
    return {
        "mean": x.mean().item(),
        "std": x.std(unbiased=False).item(),
        "rms": x.square().mean().sqrt().item(),
        "l2_norm": x.norm().item(),
    }


def main() -> None:
    args = parse_args()
    if args.num_seeds < 1 or args.soft_prompt_length < 1 or args.batch_size < 1:
        raise ValueError(
            "--num-seeds, --soft-prompt-length, and --batch-size must be positive"
        )
    if args.seeds is not None and any(seed < 0 or seed > 2**63 - 1 for seed in args.seeds):
        raise ValueError("Every value in --seeds must be between 0 and 2**63-1")

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this 7B model script.")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    embedding_layer = model.get_input_embeddings()
    device = embedding_layer.weight.device
    weight = embedding_layer.weight.detach().float()
    target_mean = weight.mean()
    target_std = weight.std(unbiased=False)
    model_embedding_stats = tensor_stats(weight)

    if args.seeds is not None:
        seeds = args.seeds
        seed_source = "explicit"
    else:
        # SystemRandom uses the OS cryptographic random source. Passing
        # --master-seed remains available when a repeatable list is desired.
        seed_rng = (
            random.SystemRandom()
            if args.master_seed is None
            else random.Random(args.master_seed)
        )
        seeds = seed_rng.sample(range(2**31), args.num_seeds)
        seed_source = "os_csprng" if args.master_seed is None else "master_seed"

    text_input_ids = None
    if args.prompt:
        messages = [{"role": "user", "content": args.prompt}]
        text_input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )["input_ids"].to(device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        progress = tqdm(total=len(seeds), desc="Generating", unit="prompt")
        for batch_start in range(0, len(seeds), args.batch_size):
            batch_seeds = seeds[batch_start : batch_start + args.batch_size]
            progress.set_postfix(
                batch=f"{batch_start // args.batch_size + 1}/"
                f"{(len(seeds) + args.batch_size - 1) // args.batch_size}"
            )
            soft_prompts = []
            for seed in batch_seeds:
                # CPU generation makes seed -> prompt independent of CUDA RNG state.
                generator = torch.Generator(device="cpu").manual_seed(seed)
                soft = torch.randn(
                    args.soft_prompt_length,
                    embedding_layer.embedding_dim,
                    generator=generator,
                    dtype=torch.float32,
                )
                # Match global mean and population std to the token embedding table.
                soft = (soft - soft.mean()) / soft.std(unbiased=False)
                soft = soft * target_std.cpu() + target_mean.cpu()
                soft_prompts.append(soft)

            soft_batch = torch.stack(soft_prompts).to(
                device=device, dtype=embedding_layer.weight.dtype
            )
            inputs_embeds = soft_batch
            if text_input_ids is not None:
                text_embeds = embedding_layer(text_input_ids)
                text_embeds = text_embeds.expand(len(batch_seeds), -1, -1)
                inputs_embeds = torch.cat((inputs_embeds, text_embeds), dim=1)

            attention_mask = torch.ones(
                inputs_embeds.shape[:2], dtype=torch.long, device=device
            )
            with torch.inference_mode():
                generated_ids = model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,  # greedy decoding
                    pad_token_id=tokenizer.eos_token_id,
                )
            output_texts = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            for offset, (seed, soft, output_text) in enumerate(
                zip(batch_seeds, soft_prompts, output_texts)
            ):
                index = batch_start + offset
                soft_bytes = soft.contiguous().numpy().tobytes()
                record = {
                    "index": index,
                    "seed": seed,
                    "seed_source": seed_source,
                    "master_seed": args.master_seed,
                    "num_seeds": len(seeds),
                    "input": {
                        "text": args.prompt,
                        "soft_prompt_length": args.soft_prompt_length,
                        "embedding_dim": embedding_layer.embedding_dim,
                        "soft_prompt_sha256": hashlib.sha256(soft_bytes).hexdigest(),
                        "soft_prompt_stats": tensor_stats(soft),
                        "model_embedding_stats": model_embedding_stats,
                    },
                    "output": output_text,
                    "generation": {
                        "method": "greedy",
                        "do_sample": False,
                        "max_new_tokens": args.max_new_tokens,
                        "batch_size": args.batch_size,
                        "actual_batch_size": len(batch_seeds),
                    },
                    "model": args.model,
                }
                if not args.no_save_embeddings:
                    record["input"]["soft_prompt_embeddings"] = soft.tolist()

                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
            progress.update(len(batch_seeds))
        progress.close()

    print(f"Done: {args.output.resolve()}")


if __name__ == "__main__":
    main()
