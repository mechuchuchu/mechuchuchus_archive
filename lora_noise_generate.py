#!/usr/bin/env python3
"""Generate sampled responses per seed using Gaussian-noise LoRA weights.

--prompt accepts either a plain string prompt, or a path to a .jsonl file.
JSONL lines can be:
  - a JSON string:            "Explain Gaussian distributions"
  - an object with "prompt":  {"prompt": "...", "system": "optional override"}
When multiple prompts are given, --batch-size controls batched generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_SYSTEM = "You are an accurate and helpful AI assistant."
DEFAULT_PROMPT = "Explain Gaussian distributions intuitively."
DEFAULT_TARGET_MODULES = ["q_proj", "v_proj"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def seed_all(seed: int) -> None:
    """Seed all RNGs used by the script and by Transformers generation."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def replace_lora_weights(
    model: torch.nn.Module,
    seed: int,
    noise_mean: float,
    noise_std: float,
) -> tuple[int, int]:
    """Replace every LoRA A/B parameter with Gaussian samples."""
    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(seed)
    tensor_count = 0
    parameter_count = 0

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_A" not in name and "lora_B" not in name:
                continue
            noise = torch.randn(
                parameter.shape,
                generator=cpu_generator,
                device="cpu",
                dtype=torch.float32,
            )
            noise = (noise * noise_std + noise_mean).to(
                device=parameter.device, dtype=parameter.dtype
            )
            parameter.copy_(noise)
            tensor_count += 1
            parameter_count += parameter.numel()

    if tensor_count == 0:
        raise RuntimeError("No LoRA A/B tensors were found to modify.")
    return tensor_count, parameter_count


def load_prompts(prompt_arg: str, default_system: str) -> tuple[list[dict[str, str]], str]:
    """Resolve --prompt into a list of {"system", "prompt"} entries.

    If prompt_arg is a path to an existing .jsonl file, each line is parsed.
    Otherwise the value is treated as a single literal prompt.
    Returns (entries, prompt_source).
    """
    path = Path(prompt_arg)
    if path.suffix.lower() == ".jsonl":
        if not path.is_file():
            raise ValueError(f"Prompt JSONL file not found: {path}")
        entries: list[dict[str, str]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}:{line_number}: invalid JSON ({exc.msg})"
                    ) from exc
                if isinstance(parsed, str):
                    entries.append({"system": default_system, "prompt": parsed})
                elif isinstance(parsed, dict):
                    if "prompt" not in parsed or not isinstance(parsed["prompt"], str):
                        raise ValueError(
                            f"{path}:{line_number}: object must contain a string 'prompt' key"
                        )
                    system = parsed.get("system", default_system)
                    if not isinstance(system, str):
                        raise ValueError(
                            f"{path}:{line_number}: 'system' must be a string"
                        )
                    entries.append({"system": system, "prompt": parsed["prompt"]})
                else:
                    raise ValueError(
                        f"{path}:{line_number}: line must be a JSON string or object"
                    )
        if not entries:
            raise ValueError(f"Prompt JSONL file is empty: {path}")
        return entries, str(path)
    return [{"system": default_system, "prompt": prompt_arg}], "inline"


def batched(items: list[Any], batch_size: int) -> list[list[Any]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="A literal prompt string, or a path to a .jsonl file of prompts.",
    )
    parser.add_argument("--system", default=DEFAULT_SYSTEM)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--noise-mean", type=float, default=0.0)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--num-random-seeds", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of prompts generated together per forward pass.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.noise_std < 0:
        raise ValueError("--noise-std must be non-negative")
    if args.num_random_seeds < 1:
        raise ValueError("--num-random-seeds must be at least 1")
    if args.rank < 1:
        raise ValueError("--rank must be at least 1")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.seeds is not None and args.num_random_seeds != 1:
        raise ValueError("Use either --seeds or --num-random-seeds, not both")


def make_jsonable_settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": args.model,
        "rank": args.rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": 0.0,
        "target_modules": DEFAULT_TARGET_MODULES,
        "noise_mean": args.noise_mean,
        "noise_std": args.noise_std,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
        prompt_entries, prompt_source = load_prompts(args.prompt, args.system)
        args.output_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    run_created = utc_now()
    timestamp = run_created.isoformat()
    filename_hash = hashlib.sha256(timestamp.encode("utf-8")).hexdigest()
    model_slug = args.model.replace("/", "__")
    output_path = (
        args.output_dir
        / f"random_lora_generations_{model_slug}_{filename_hash[:16]}.jsonl"
    )

    if args.seeds is None:
        seeds = [secrets.randbits(63) for _ in range(args.num_random_seeds)]
        seed_source = "random"
    else:
        seeds = args.seeds
        seed_source = "user"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only models must be left-padded for batched generation.
    tokenizer.padding_side = "left"

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        target_modules=DEFAULT_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.eval()

    # Pre-format every prompt once; reused across all seeds.
    formatted_entries: list[dict[str, Any]] = []
    for prompt_index, entry in enumerate(prompt_entries):
        messages = [
            {"role": "system", "content": entry["system"]},
            {"role": "user", "content": entry["prompt"]},
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        formatted_entries.append(
            {
                "prompt_index": prompt_index,
                "messages": messages,
                "formatted_prompt": formatted_prompt,
            }
        )

    batches = batched(formatted_entries, args.batch_size)
    settings = make_jsonable_settings(args)
    total_records = len(seeds) * len(formatted_entries)
    record_counter = 0

    run_info = {
        "run_created_at_utc": timestamp,
        "output_path": str(output_path),
        "device": str(device),
        "seed_source": seed_source,
        "seeds": seeds,
        "prompt_source": prompt_source,
        "num_prompts": len(formatted_entries),
        "num_batches": len(batches),
        "total_records": total_records,
        **settings,
    }
    tqdm.write("=== Run configuration ===")
    tqdm.write(json.dumps(run_info, ensure_ascii=False, indent=2))
    tqdm.write("=========================")

    progress = tqdm(
        total=total_records,
        unit="gen",
        desc=f"Generating [{args.model}]",
    )
    with output_path.open("w", encoding="utf-8") as output_file:
        for seed_index, seed in enumerate(seeds):
            seed_all(seed)
            tensor_count, parameter_count = replace_lora_weights(
                model, seed, args.noise_mean, args.noise_std
            )
            for batch in batches:
                batch_texts = [item["formatted_prompt"] for item in batch]
                encoded = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}

                with torch.inference_mode():
                    generated = model.generate(
                        **encoded,
                        do_sample=False,
                        max_new_tokens=args.max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                padded_input_length = encoded["input_ids"].shape[-1]
                attention_mask = encoded["attention_mask"]

                for row, item in enumerate(batch):
                    new_tokens = generated[row, padded_input_length:]
                    # Trailing pad tokens (finished-early sequences) are
                    # dropped by skip_special_tokens during decode.
                    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                    real_input_tokens = int(attention_mask[row].sum().item())
                    non_pad_output = int(
                        (new_tokens != tokenizer.pad_token_id).sum().item()
                    )
                    record = {
                        "run_created_at_utc": timestamp,
                        "generated_at_utc": utc_now().isoformat(),
                        "index": record_counter,
                        "seed_index": seed_index,
                        "seed": seed,
                        "seed_source": seed_source,
                        "prompt_index": item["prompt_index"],
                        "prompt_source": prompt_source,
                        "settings": settings,
                        "input": {
                            "messages": item["messages"],
                            "formatted_prompt": item["formatted_prompt"],
                        },
                        "output": {
                            "text": text,
                            "input_token_count": real_input_tokens,
                            "output_token_count": non_pad_output,
                        },
                        "lora": {
                            "tensor_count": tensor_count,
                            "parameter_count": parameter_count,
                        },
                    }
                    output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    record_counter += 1
                output_file.flush()
                progress.update(len(batch))
                progress.set_postfix(
                    seed=seed,
                    prompts=f"{batch[0]['prompt_index']}-{batch[-1]['prompt_index']}",
                )

    progress.close()
    print(f"Completed: {output_path}")


if __name__ == "__main__":
    main()
