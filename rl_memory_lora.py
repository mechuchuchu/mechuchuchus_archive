import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================
# Configuration
# =====================================================

INPUT_DIM = 128
MEMORY_DIM = 256
MEMORY_SIZE = 50000

CANDIDATE_K = 512
SELECT_K = 100
NOISE_RATIO = 0.05

HIDDEN_DIM = 256
NUM_CLASSES = 10
LORA_RANK = 8

LR = 1e-3
RL_COEF = 1e-3
ENTROPY_COEF = 1e-3

ATTENTION_TEMP = 0.1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42)


# =====================================================
# LoRA Linear
# =====================================================

class LoRALinear(nn.Module):

    def __init__(self, in_dim, out_dim, rank=8, alpha=16):
        super().__init__()

        self.base = nn.Linear(in_dim, out_dim)

        for p in self.base.parameters():
            p.requires_grad = False

        self.A = nn.Parameter(
            torch.randn(rank, in_dim) * 0.01
        )

        self.B = nn.Parameter(
            torch.zeros(out_dim, rank)
        )

        self.scale = alpha / rank

    def forward(self, x):

        base = self.base(x)

        lora = F.linear(
            F.linear(x, self.A),
            self.B
        )

        return base + self.scale * lora


# =====================================================
# Frozen Random Memory + RL Selection
# =====================================================

class RLExternalMemory(nn.Module):

    def __init__(self):

        super().__init__()

        # Frozen random memory
        memory = torch.randn(
            MEMORY_SIZE,
            MEMORY_DIM
        )

        memory = F.normalize(memory, dim=-1)

        self.register_buffer("memory", memory)

        # Selection policy (LoRA)
        # Input = Query + Candidate vector
        self.policy = nn.Sequential(
            LoRALinear(MEMORY_DIM * 2, 128, LORA_RANK),
            nn.GELU(),
            LoRALinear(128, 1, LORA_RANK)
        )

        # Learnable similarity temperature
        self.log_temperature = nn.Parameter(
            torch.tensor(0.0)
        )

    def forward(self, query):

        B = query.shape[0]

        query = F.normalize(query, dim=-1)

        # -----------------------------------------
        # 1. Candidate retrieval: 50k -> 512
        # -----------------------------------------

        similarity = query @ self.memory.T

        candidate_scores, candidate_indices = torch.topk(
            similarity,
            CANDIDATE_K,
            dim=-1
        )

        candidates = self.memory[candidate_indices]

        # -----------------------------------------
        # 2. Joint selection policy
        # -----------------------------------------

        q_expanded = query.unsqueeze(1).expand(
            -1,
            CANDIDATE_K,
            -1
        )

        policy_input = torch.cat(
            [q_expanded, candidates],
            dim=-1
        )

        policy_logits = self.policy(
            policy_input
        ).squeeze(-1)

        temperature = self.log_temperature.exp().clamp(
            0.05, 5.0
        )

        # Learnable policy + similarity prior
        policy_logits = (
            policy_logits
            + candidate_scores / temperature
        )

        # -----------------------------------------
        # 3. Select 100 vectors
        # -----------------------------------------

        selected_positions = []
        log_probs = []
        entropies = []

        if self.training:

            # REINFORCE sampling without replacement
            remaining = policy_logits

            for _ in range(SELECT_K):

                dist = torch.distributions.Categorical(
                    logits=remaining
                )

                action = dist.sample()

                log_probs.append(
                    dist.log_prob(action)
                )

                entropies.append(
                    dist.entropy()
                )

                selected_positions.append(action)

                # Remove already selected candidate
                remaining = remaining.scatter(
                    1,
                    action.unsqueeze(-1),
                    -1e9
                )

            log_probs = torch.stack(
                log_probs, dim=1
            )

            policy_entropy = torch.stack(
                entropies, dim=1
            ).mean()

        else:

            # Deterministic greedy selection at evaluation
            selected_positions = torch.topk(
                policy_logits,
                SELECT_K,
                dim=-1
            ).indices

            log_probs = None
            policy_entropy = None

        selected_positions = torch.stack(
            selected_positions,
            dim=1
        ) if self.training else selected_positions

        selected_indices = candidate_indices.gather(
            1,
            selected_positions
        )

        # -----------------------------------------
        # 4. Random 5% replacement noise
        # -----------------------------------------

        if self.training and NOISE_RATIO > 0:

            n_noise = max(
                1,
                int(SELECT_K * NOISE_RATIO)
            )

            # Random positions among selected 100
            noise_positions = torch.rand(
                B,
                SELECT_K,
                device=query.device
            ).argsort(dim=-1)[:, :n_noise]

            # Random memory vectors
            random_indices = torch.randint(
                0,
                MEMORY_SIZE,
                (B, n_noise),
                device=query.device
            )

            selected_indices = selected_indices.clone()

            selected_indices.scatter_(
                1,
                noise_positions,
                random_indices
            )

        # -----------------------------------------
        # 5. Attention aggregation
        # -----------------------------------------

        selected_memory = self.memory[selected_indices]

        selected_scores = torch.einsum(
            "bd,bkd->bk",
            query,
            selected_memory
        )

        attention = F.softmax(
            selected_scores / ATTENTION_TEMP,
            dim=-1
        )

        retrieved = (
            attention.unsqueeze(-1) * selected_memory
        ).sum(dim=1)

        attention_entropy = -(
            attention * (attention + 1e-9).log()
        ).sum(dim=-1).mean()

        return (
            retrieved,
            attention,
            selected_indices,
            log_probs,
            policy_entropy,
            attention_entropy
        )


# =====================================================
# Full Model
# =====================================================

class RLMemoryLoRAMLP(nn.Module):

    def __init__(self):

        super().__init__()

        # 128 -> 256
        self.query_encoder = LoRALinear(
            INPUT_DIM,
            MEMORY_DIM,
            LORA_RANK
        )

        self.memory = RLExternalMemory()

        # Original input + retrieved memory
        # 128 + 256 = 384

        self.fc1 = LoRALinear(
            INPUT_DIM + MEMORY_DIM,
            HIDDEN_DIM,
            LORA_RANK
        )

        self.fc2 = LoRALinear(
            HIDDEN_DIM,
            HIDDEN_DIM,
            LORA_RANK
        )

        self.fc3 = LoRALinear(
            HIDDEN_DIM,
            NUM_CLASSES,
            LORA_RANK
        )

    def forward(self, x):

        # Query encoder
        query = self.query_encoder(x)

        # RL memory retrieval
        (
            retrieved,
            attention,
            indices,
            log_probs,
            policy_entropy,
            attention_entropy
        ) = self.memory(query)

        # Combine input and memory
        h = torch.cat(
            [x, retrieved],
            dim=-1
        )

        # LoRA MLP
        h = F.gelu(self.fc1(h))
        h = F.gelu(self.fc2(h))

        logits = self.fc3(h)

        return {
            "logits": logits,
            "attention": attention,
            "indices": indices,
            "log_probs": log_probs,
            "policy_entropy": policy_entropy,
            "attention_entropy": attention_entropy
        }


# =====================================================
# Random Dataset
# =====================================================

def make_dataset(n):

    x = torch.randn(n, INPUT_DIM)

    y = torch.randint(
        0,
        NUM_CLASSES,
        (n,)
    )

    return x, y


# =====================================================
# Parameter Statistics
# =====================================================

def print_parameters(model):

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    memory_elements = model.memory.memory.numel()

    print("=" * 50)
    print(f"Trainable parameters: {trainable:,}")
    print(f"Model parameters:     {parameters:,}")
    print(f"Frozen memory:        {memory_elements:,}")
    print(f"Memory shape:         {tuple(model.memory.memory.shape)}")
    print("=" * 50)


# =====================================================
# Evaluation
# =====================================================

@torch.no_grad()
def evaluate(model, x, y, batch_size=32):

    model.eval()

    total_correct = 0
    total_loss = 0
    total_entropy = 0
    total_attention_entropy = 0
    total_count = 0

    for start in range(0, len(x), batch_size):

        xb = x[start:start + batch_size].to(DEVICE)
        yb = y[start:start + batch_size].to(DEVICE)

        out = model(xb)

        logits = out["logits"]

        loss = F.cross_entropy(
            logits,
            yb,
            reduction="sum"
        )

        total_loss += loss.item()

        total_correct += (
            logits.argmax(-1) == yb
        ).sum().item()

        total_attention_entropy += (
            out["attention_entropy"].item() * len(xb)
        )

        total_count += len(xb)

    return {
        "accuracy": total_correct / total_count,
        "loss": total_loss / total_count,
        "attention_entropy": (
            total_attention_entropy / total_count
        )
    }


# =====================================================
# Training
# =====================================================

def train_experiment(
    n_samples=1000,
    epochs=50,
    batch_size=32,
    log_every=5,
    lr=LR,
    use_compile=False
):

    model = RLMemoryLoRAMLP().to(DEVICE)

    # Keep actual training dataset
    x_train, y_train = make_dataset(n_samples)

    x_train = x_train.to(DEVICE)
    y_train = y_train.to(DEVICE)

    optimizer = torch.optim.AdamW(
        [
            p for p in model.parameters()
            if p.requires_grad
        ],
        lr=lr
    )

    print_parameters(model)

    # Compiled wrapper shares parameters with `model`
    # (evaluation still uses the original eager model)
    fwd = torch.compile(model) if use_compile else model

    for epoch in range(epochs):

        model.train()

        perm = torch.randperm(
            n_samples,
            device=DEVICE
        )

        total_ce = 0
        total_rl = 0
        total_correct = 0
        total_policy_entropy = 0
        total_attention_entropy = 0
        total_count = 0

        for idx in perm.split(batch_size):

            xb = x_train[idx]
            yb = y_train[idx]

            out = fwd(xb)

            logits = out["logits"]
            log_probs = out["log_probs"]

            # ---------------------------------
            # Classification loss
            # ---------------------------------

            ce_per_sample = F.cross_entropy(
                logits,
                yb,
                reduction="none"
            )

            ce_loss = ce_per_sample.mean()

            # ---------------------------------
            # Loss-based reward
            # ---------------------------------

            # Lower classification loss = higher reward
            reward = -ce_per_sample.detach()

            # Batch baseline
            advantage = reward - reward.mean()

            # Normalize advantage
            advantage = advantage / (
                advantage.std(unbiased=False) + 1e-6
            )

            # ---------------------------------
            # REINFORCE policy gradient
            # ---------------------------------

            joint_log_prob = log_probs.sum(dim=-1)

            reinforce_loss = -(
                advantage * joint_log_prob
            ).mean()

            # Entropy regularization
            entropy_bonus = (
                out["policy_entropy"]
            )

            policy_loss = (
                reinforce_loss
                - ENTROPY_COEF * entropy_bonus
            )

            # ---------------------------------
            # Combined objective
            # ---------------------------------

            loss = (
                ce_loss
                + RL_COEF * policy_loss
            )

            optimizer.zero_grad(set_to_none=True)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                [
                    p for p in model.parameters()
                    if p.requires_grad
                ],
                max_norm=1.0
            )

            optimizer.step()

            # ---------------------------------
            # Statistics
            # ---------------------------------

            total_ce += ce_loss.item() * len(idx)
            total_rl += reinforce_loss.item() * len(idx)

            total_correct += (
                logits.argmax(-1) == yb
            ).sum().item()

            total_policy_entropy += (
                entropy_bonus.item() * len(idx)
            )

            total_attention_entropy += (
                out["attention_entropy"].item() * len(idx)
            )

            total_count += len(idx)

        if (epoch + 1) % log_every == 0:

            print(
                f"Epoch {epoch+1:3d} | "
                f"CE: {total_ce/total_count:.4f} | "
                f"RL: {total_rl/total_count:.4f} | "
                f"Train Acc: {total_correct/total_count:.4f} | "
                f"Policy H: {total_policy_entropy/total_count:.4f} | "
                f"Attention H: {total_attention_entropy/total_count:.4f}"
            )

    return model, x_train, y_train


# =====================================================
# Main
# =====================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="RL memory + LoRA MLP experiment"
    )

    parser.add_argument(
        "--N", type=int, default=1000,
        help="number of training samples (default: 1000)"
    )
    parser.add_argument(
        "--n_test", type=int, default=1000,
        help="number of test samples (default: 1000)"
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="training batch size (default: 32)"
    )
    parser.add_argument(
        "--log_every", type=int, default=5,
        help="print training log every N epochs (default: 5)"
    )
    parser.add_argument(
        "--lr", type=float, default=LR,
        help="learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--compile", action="store_true",
        help="use torch.compile for the training forward pass"
    )

    args = parser.parse_args()

    if args.log_every < 1:
        parser.error("--log_every must be >= 1")

    return args


if __name__ == "__main__":

    args = parse_args()

    model, x_train, y_train = train_experiment(
        n_samples=args.N,
        epochs=args.epochs,
        batch_size=args.batch_size,
        log_every=args.log_every,
        lr=args.lr,
        use_compile=args.compile
    )

    # Independent test dataset
    x_test, y_test = make_dataset(args.n_test)

    print("\n========== TRAIN EVALUATION ==========")

    train_metrics = evaluate(
        model,
        x_train,
        y_train
    )

    print(train_metrics)

    print("\n========== TEST EVALUATION ==========")

    test_metrics = evaluate(
        model,
        x_test,
        y_test
    )

    print(test_metrics)