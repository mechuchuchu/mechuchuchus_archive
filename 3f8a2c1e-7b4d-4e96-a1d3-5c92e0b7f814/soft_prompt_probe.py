import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "IFM/K2-Horizon-0.9B"
MARKER = "[[SOFT_PROMPT_SLOT]]"
TARGET = "At dusk, the paper moon drifted over the quiet harbor, carrying a small blue secret."
SOFT_TOKENS = 16
MAX_STEPS = 500
LR = 0.08
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def encode(tokenizer, text):
    return tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(DEVICE)


def render_user(tokenizer, content, add_generation_prompt=True):
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        reasoning_effort="low",
    )


def split_prompt(tokenizer, user_content, direct_assistant=False):
    rendered = render_user(
        tokenizer, user_content, add_generation_prompt=not direct_assistant
    )
    if direct_assistant:
        rendered += "<|ifm|im_start|>assistant\n"
    if rendered.count(MARKER) != 1:
        raise RuntimeError("Expected exactly one soft-prompt marker in rendered chat template")
    left, right = rendered.split(MARKER)
    return encode(tokenizer, left), encode(tokenizer, right)


def assemble_inputs(embedding, prefix_ids, suffix_ids, soft_prompt=None):
    pieces = [embedding(prefix_ids)]
    if soft_prompt is not None:
        pieces.append(soft_prompt.to(dtype=embedding.weight.dtype).unsqueeze(0))
    pieces.append(embedding(suffix_ids))
    return torch.cat(pieces, dim=1)


def train_prompt(model, tokenizer, embedding):
    prefix_ids, suffix_ids = split_prompt(
        tokenizer,
        f"Recall exactly the sentence encoded by this learned prompt: {MARKER}",
        direct_assistant=True,
    )
    target_ids = encode(tokenizer, TARGET)
    end_turn_id = tokenizer.convert_tokens_to_ids("<|ifm|im_end|>")
    target_ids = torch.cat([target_ids, torch.tensor([[end_turn_id]], device=DEVICE)], dim=1)

    init_ids = encode(tokenizer, "Please recall the exact sentence hidden in this prompt")
    init_vectors = embedding(init_ids).squeeze(0).detach().float()
    if init_vectors.shape[0] >= SOFT_TOKENS:
        init_vectors = init_vectors[:SOFT_TOKENS]
    else:
        # Cycle the semantic initialization to the requested virtual-token length.
        repeats = math.ceil(SOFT_TOKENS / init_vectors.shape[0])
        init_vectors = init_vectors.repeat(repeats, 1)[:SOFT_TOKENS]
    prompt = torch.nn.Parameter(init_vectors.contiguous())
    optimizer = torch.optim.Adam([prompt], lr=LR)

    target_start = prefix_ids.shape[1] + SOFT_TOKENS + suffix_ids.shape[1]
    target_len = target_ids.shape[1]
    attention_mask = torch.ones(
        (1, target_start + target_len), device=DEVICE, dtype=torch.long
    )
    best_loss = float("inf")
    best_prompt = None
    history = []
    print(f"Training {SOFT_TOKENS} virtual tokens to encode {target_len - 1} target tokens", flush=True)
    start_time = time.time()

    for step in range(1, MAX_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        context_embeds = assemble_inputs(embedding, prefix_ids, suffix_ids, prompt)
        target_embeds = embedding(target_ids)
        inputs_embeds = torch.cat([context_embeds, target_embeds], dim=1)
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = outputs.logits[:, target_start - 1 : target_start + target_len - 1, :]
        loss = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]), target_ids.reshape(-1)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_([prompt], 1.0)
        optimizer.step()

        value = float(loss.detach())
        if value < best_loss:
            best_loss = value
            best_prompt = prompt.detach().clone()
        if step == 1 or step % 25 == 0 or value < 0.02:
            history.append({"step": step, "loss": value})
            print(f"step={step:03d} loss={value:.4f}", flush=True)
        if value < 0.015:
            break

    print(f"Training finished in {time.time() - start_time:.1f}s; best loss={best_loss:.4f}", flush=True)
    return best_prompt, history


@torch.inference_mode()
def greedy_tokens(model, tokenizer, input_embeds, max_new_tokens=512):
    attention_mask = torch.ones(
        input_embeds.shape[:2], device=input_embeds.device, dtype=torch.long
    )
    out = model(inputs_embeds=input_embeds, attention_mask=attention_mask, use_cache=True)
    past = out.past_key_values
    logits = out.logits[:, -1, :]
    generated = []
    stop_ids = {
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|ifm|im_end|>"),
    }
    for _ in range(max_new_tokens):
        next_id = logits.argmax(dim=-1, keepdim=True)
        token_id = int(next_id[0, 0])
        if token_id in stop_ids:
            break
        generated.append(token_id)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=DEVICE, dtype=torch.long)], dim=1
        )
        out = model(
            input_ids=next_id,
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
        )
        past = out.past_key_values
        logits = out.logits[:, -1, :]
    return generated


def final_answer(tokenizer, generated):
    # K2-Horizon separates thought and answer with a dedicated closing token.
    close_ids = {
        tokenizer.convert_tokens_to_ids("</ifm|think>"),
        tokenizer.convert_tokens_to_ids("</ifm|think_fast>"),
        tokenizer.convert_tokens_to_ids("</ifm|think_faster>"),
    }
    end_positions = [i for i, token_id in enumerate(generated) if token_id in close_ids]
    if not end_positions:
        return "[no final answer before token limit]"
    return tokenizer.decode(generated[end_positions[-1] + 1 :], skip_special_tokens=True).strip()


def main():
    random.seed(2026)
    torch.manual_seed(2026)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2026)
        torch.backends.cuda.matmul.allow_tf32 = True

    print(f"Loading tokenizer and {MODEL_ID} on {DEVICE} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=DEVICE,
        low_cpu_mem_usage=True,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    embedding = model.get_input_embeddings()

    soft_prompt, history = train_prompt(model, tokenizer, embedding)
    torch.save(
        {
            "model_id": MODEL_ID,
            "target": TARGET,
            "soft_tokens": SOFT_TOKENS,
            "soft_prompt": soft_prompt.cpu(),
            "training_history": history,
        },
        os.path.join(OUT_DIR, "soft_prompt.pt"),
    )

    test_cases = [
        ("EN: What is X?", "What is {x}?"),
        ("EN: Translate X into Korean.", "Translate {x} into Korean."),
        ("EN: What feeling does X convey?", "What feeling does {x} convey?"),
        ("KO: X는 뭐야?", "{x}는 뭐야?"),
        ("KO: X를 한국어로 번역해줘.", "{x}를 한국어로 번역해줘."),
        ("KO: 이 X는 어떤 느낌이야?", "이 {x}는 어떤 느낌이야?"),
    ]
    results = {
        "model_id": MODEL_ID,
        "target_sentence": TARGET,
        "soft_prompt_tokens": SOFT_TOKENS,
        "training_history": history,
        "responses": [],
    }

    recall_prefix, recall_suffix = split_prompt(
        tokenizer,
        f"Recall exactly the sentence encoded by this learned prompt: {MARKER}",
        direct_assistant=True,
    )
    recall_embeds = assemble_inputs(embedding, recall_prefix, recall_suffix, soft_prompt)
    recall_tokens = greedy_tokens(model, tokenizer, recall_embeds)
    recall_raw = tokenizer.decode(recall_tokens, skip_special_tokens=True)
    recall_match = TARGET in recall_raw
    results["recall_probe_exact_match"] = recall_match
    print(f"\nRecall probe exact match: {recall_match}", flush=True)

    for title, template in test_cases:
        # The learned continuous vectors are inserted at X. Literal and blank
        # controls show whether the effect depends on that learned prompt.
        virtual_content = template.format(x=MARKER)
        prefix_ids, suffix_ids = split_prompt(tokenizer, virtual_content)
        soft_embeds = assemble_inputs(embedding, prefix_ids, suffix_ids, soft_prompt)
        soft_response = final_answer(tokenizer, greedy_tokens(model, tokenizer, soft_embeds))

        literal_content = template.format(x=TARGET)
        literal_ids = encode(tokenizer, render_user(tokenizer, literal_content))
        literal_response = final_answer(tokenizer, greedy_tokens(model, tokenizer, embedding(literal_ids)))

        blank_content = template.format(x="")
        blank_ids = encode(tokenizer, render_user(tokenizer, blank_content))
        blank_response = final_answer(tokenizer, greedy_tokens(model, tokenizer, embedding(blank_ids)))

        record = {
            "question": title,
            "soft_prompt_response": soft_response,
            "literal_sentence_control": literal_response,
            "blank_control": blank_response,
        }
        results["responses"].append(record)
        print(f"\n{title}\n  soft prompt: {soft_response}\n  literal:     {literal_response}\n  blank:       {blank_response}", flush=True)

    out_path = os.path.join(OUT_DIR, "soft_prompt_probe_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
