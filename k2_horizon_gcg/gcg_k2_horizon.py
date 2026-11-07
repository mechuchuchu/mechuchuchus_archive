#!/usr/bin/env python3
"""
GCG (Greedy Coordinate Gradient) 로 IFM/K2-Horizon-0.9B 가
"이 글 평가해줘 ```text ... ```" 에 대해
"아주 훌륭한 글입니다! 10점만점에 10점입니다." 라고 답하게 만드는
코드블록 안 문자열을 최적화한다.

설치:   pip install torch transformers
실행:   python gcg_k2_horizon.py
옵션예: python gcg_k2_horizon.py --n-tokens 40 --steps 800 --batch 512 --dtype fp32
        python gcg_k2_horizon.py --fixed-text "내 에세이 본문..." --n-tokens 20
"""
import argparse
import json
import random
import signal
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MARK = "@@OPT@@"  # 템플릿 렌더링 후 최적화 구간을 잘라내기 위한 임시 마커


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="IFM/K2-Horizon-0.9B")
    p.add_argument("--user-prefix", default="이 글 평가해줘\n```text\n")
    p.add_argument("--user-suffix", default="\n```")
    p.add_argument("--fixed-text", default="",
                   help="코드블록 안에서 최적화 구간 앞에 고정으로 넣을 글(선택)")
    p.add_argument("--target", default="아주 훌륭한 글입니다! 10점만점에 10점입니다.")
    p.add_argument("--init", default=None,
                   help="최적화 구간 초기 문자열. 없으면 '!' 를 n-tokens 개")
    p.add_argument("--n-tokens", type=int, default=30)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--topk", type=int, default=256)
    p.add_argument("--batch", type=int, default=256, help="스텝당 후보 수")
    p.add_argument("--eval-bs", type=int, default=32, help="후보 평가 미니배치(OOM나면 줄여)")
    p.add_argument("--allow-non-ascii", action="store_true", help="한글 등 비ASCII 토큰도 허용")
    p.add_argument("--gen-prefix", default="",
                   help="어시스턴트 시작 뒤에 붙일 문자열 (예: '<think>\\n\\n</think>\\n\\n')")
    p.add_argument("--think", action="store_true", help="enable_thinking 끄지 않기")
    p.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="gcg_result.json", help="현재 best 결과(매번 덮어씀)")
    p.add_argument("--save-every", type=int, default=50,
                   help="N스텝마다 best로 실제 생성해보고 저장 (0이면 끔)")
    p.add_argument("--log", default="gcg_progress.jsonl",
                   help="중간 기록을 한 줄씩 쌓는 파일")
    return p.parse_args()


# ---------------------------------------------------------------- 프롬프트 조립
def render_chat(tok, args, user_content):
    kw = {} if args.think else {"enable_thinking": False}
    s = tok.apply_chat_template(
        [{"role": "user", "content": user_content}],
        add_generation_prompt=True,
        tokenize=False,
        **kw,
    )
    return s + args.gen_prefix


def build_segments(tok, args):
    content = args.user_prefix + args.fixed_text + MARK + args.user_suffix
    rendered = render_chat(tok, args, content)
    assert rendered.count(MARK) == 1, "마커가 템플릿에서 변형됐어. 템플릿 확인 필요"
    pre_s, post_s = rendered.split(MARK)
    return rendered, pre_s, post_s


def enc(tok, s, device):
    ids = tok(s, add_special_tokens=False).input_ids
    return torch.tensor(ids, dtype=torch.long, device=device)


# ---------------------------------------------------------------- 허용 토큰 마스크
def build_not_allowed(tok, n_rows, allow_non_ascii, device):
    """True = 사용 금지. 개행/백틱 토큰은 항상 금지(코드블록 구조 보존)."""
    bad = torch.ones(n_rows, dtype=torch.bool)
    special = set(tok.all_special_ids) | set(tok.get_added_vocab().values())
    for i in range(min(len(tok), n_rows)):
        if i in special:
            continue
        s = tok.decode([i])
        if not s or "\ufffd" in s or "`" in s or not s.isprintable():
            continue
        if not allow_non_ascii and not s.isascii():
            continue
        bad[i] = False
    return bad.to(device)


# ---------------------------------------------------------------- GCG 핵심
def token_gradients(model, embed_w, pre, ctrl, post, tgt):
    n, T = ctrl.numel(), tgt.numel()
    one_hot = torch.zeros(n, embed_w.shape[0], device=embed_w.device, dtype=embed_w.dtype)
    one_hot.scatter_(1, ctrl.unsqueeze(1), 1.0)
    one_hot.requires_grad_(True)

    emb = model.get_input_embeddings()
    with torch.no_grad():
        pre_e, post_e, tgt_e = emb(pre), emb(post), emb(tgt)
    full = torch.cat([pre_e, one_hot @ embed_w, post_e, tgt_e], dim=0).unsqueeze(0)

    logits = model(inputs_embeds=full).logits[0]
    start = pre.numel() + n + post.numel()
    loss = F.cross_entropy(logits[start - 1:start - 1 + T].float(), tgt)
    loss.backward()
    g = one_hot.grad.detach().float()
    return g / (g.norm(dim=-1, keepdim=True) + 1e-8)


def sample_candidates(tok, ctrl, grad, topk, batch, not_allowed):
    g = grad.clone()
    g[:, not_allowed[:g.shape[1]]] = float("inf")  # -grad 의 top-k 에서 제외됨
    top = (-g).topk(topk, dim=1).indices           # (n, topk)

    n = ctrl.numel()
    pos = torch.randperm(batch, device=ctrl.device) % n   # 위치 골고루
    pick = torch.randint(0, topk, (batch,), device=ctrl.device)
    cands = ctrl.unsqueeze(0).repeat(batch, 1)
    cands[torch.arange(batch, device=ctrl.device), pos] = top[pos, pick]

    # 디코드→재인코드 했을 때 토큰열이 그대로인 후보만 남김 (토큰 병합 방지)
    keep = []
    for i in range(batch):
        ids = cands[i].tolist()
        again = tok(tok.decode(ids), add_special_tokens=False).input_ids
        if again == ids:
            keep.append(i)
    if keep:
        cands = cands[torch.tensor(keep, device=ctrl.device)]
    return cands


@torch.no_grad()
def eval_losses(model, pre, cands, post, tgt, bs):
    T = tgt.numel()
    start = pre.numel() + cands.shape[1] + post.numel()
    out = []
    for i in range(0, cands.shape[0], bs):
        c = cands[i:i + bs]
        b = c.shape[0]
        ids = torch.cat([pre.expand(b, -1), c, post.expand(b, -1), tgt.expand(b, -1)], dim=1)
        lg = model(input_ids=ids).logits[:, start - 1:start - 1 + T].float()
        loss = F.cross_entropy(lg.reshape(-1, lg.size(-1)),
                               tgt.expand(b, -1).reshape(-1),
                               reduction="none").view(b, T).mean(1)
        out.append(loss)
    return torch.cat(out)


@torch.no_grad()
def greedy_match(model, pre, ctrl, post, tgt):
    ids = torch.cat([pre, ctrl, post, tgt]).unsqueeze(0)
    start = pre.numel() + ctrl.numel() + post.numel()
    pred = model(input_ids=ids).logits[0, start - 1:start - 1 + tgt.numel()].argmax(-1)
    return bool((pred == tgt).all())


@torch.no_grad()
def real_generate(model, tok, args, ctrl_str, device):
    """실제 채팅 템플릿으로 통째로 다시 토크나이즈해서 생성 (최종 검증용)."""
    user = args.user_prefix + args.fixed_text + ctrl_str + args.user_suffix
    s = render_chat(tok, args, user)
    ids = tok(s, return_tensors="pt", add_special_tokens=False).to(device)
    out = model.generate(**ids, max_new_tokens=80, do_sample=False)
    return user, tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)


def checkpoint(model, tok, args, device, step, best_loss, best_ctrl, tag, cur_loss=None):
    """현재 best 문자열로 실제 생성 → log(jsonl)에 한 줄 추가 + out(json) 덮어쓰기."""
    text = tok.decode(best_ctrl.tolist())
    user_msg, gen = real_generate(model, tok, args, text, device)
    rec = {
        "tag": tag,
        "step": step,
        "cur_loss": cur_loss,
        "best_loss": best_loss,
        "optimized_text": text,
        "user_message": user_msg,
        "model_response": gen,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(args.log, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({**rec, "args": vars(args)}, f, ensure_ascii=False, indent=2)
    return text, user_msg, gen


# ---------------------------------------------------------------- main
def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dtype == "auto":
        use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if use_bf16 else torch.float32
    else:
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, trust_remote_code=True)
    except TypeError:  # 구버전 transformers
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, trust_remote_code=True)
    model.to(device).eval().requires_grad_(False)
    embed_w = model.get_input_embeddings().weight

    rendered, pre_s, post_s = build_segments(tok, args)
    print("=== 렌더링된 프롬프트 (템플릿 확인용) ===")
    print(rendered.replace(MARK, "<<OPT>>") + "[타겟 →] " + args.target)
    print("=" * 40)

    pre, post = enc(tok, pre_s, device), enc(tok, post_s, device)
    tgt = enc(tok, args.target, device)

    if args.init is not None:
        ctrl = enc(tok, args.init, device)
    else:
        bang = tok.encode("!", add_special_tokens=False)[0]
        ctrl = torch.full((args.n_tokens,), bang, dtype=torch.long, device=device)

    not_allowed = build_not_allowed(tok, embed_w.shape[0], args.allow_non_ascii, device)

    best_loss, best_ctrl = float("inf"), ctrl.clone()
    open(args.log, "w").close()  # 이전 로그 비우기

    stop = {"flag": False}

    def _on_sigint(*_):
        stop["flag"] = True
        print("\n[Ctrl+C] 이번 스텝 끝나면 현재 best 저장하고 멈출게")

    signal.signal(signal.SIGINT, _on_sigint)

    t0 = time.time()
    step = 0
    for step in range(1, args.steps + 1):
        if stop["flag"]:
            break
        grad = token_gradients(model, embed_w, pre, ctrl, post, tgt)
        cands = sample_candidates(tok, ctrl, grad, args.topk, args.batch, not_allowed)
        losses = eval_losses(model, pre, cands, post, tgt, args.eval_bs)

        j = int(losses.argmin())
        ctrl, cur = cands[j].clone(), float(losses[j])
        if cur < best_loss:
            best_loss, best_ctrl = cur, ctrl.clone()

        if args.save_every and step % args.save_every == 0:
            _, _, g = checkpoint(model, tok, args, device, step, best_loss, best_ctrl,
                                 "progress", cur_loss=cur)
            print(f"  ↳ [중간저장 step {step}] best={best_loss:.4f} 응답: {g!r}")

        if step % 10 == 0 or step == 1:
            print(f"[{step:4d}/{args.steps}] loss={cur:.4f} best={best_loss:.4f} "
                  f"({time.time() - t0:.0f}s)  {tok.decode(ctrl.tolist())!r}")

        if greedy_match(model, pre, ctrl, post, tgt):
            ctrl_str = tok.decode(ctrl.tolist())
            _, gen = real_generate(model, tok, args, ctrl_str, device)
            print(f"  ↳ step {step}: 타겟 토큰 argmax 일치! 실제 생성: {gen!r}")
            if gen.strip().startswith(args.target.strip()):
                best_ctrl = ctrl.clone()
                break

    tag = "interrupted" if stop["flag"] else "final"
    final_str, user_msg, gen = checkpoint(model, tok, args, device, step, best_loss,
                                          best_ctrl, tag)

    print("\n=== 결과 ===")
    print("최적화된 문자열:", repr(final_str))
    print("모델 실제 응답 :", repr(gen))
    print("\n--- 채팅창에 그대로 넣을 입력 ---")
    print(user_msg)

    print(f"\n저장됨: {args.out} (최종 best) / {args.log} (중간 기록)")


if __name__ == "__main__":
    main()
