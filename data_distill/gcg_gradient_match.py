import os
import random
import math

import torch
import torch.nn.functional as F

from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "gpt2"
DATASET_NAME = "HuggingFaceFW/fineweb-edu"

SEQ_LEN = 512

# GCG
NUM_STEPS = 30
TOP_K = 64

# Number of candidate token replacements actually evaluated
# at each GCG step.
EVAL_BATCH_SIZE = 32

# How many coordinates to optimize per GCG step.
# 1 = standard coordinate-style GCG.
COORDS_PER_STEP = 1

# Candidate initialization:
# "random" or "sample"
INIT_MODE = "random"

SEED = 420

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# GPT-2 normally uses fp16/bf16 on CUDA safely for inference,
# but second-order gradients are much safer in float32.
DTYPE = torch.float32


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Load tokenizer / model
# ============================================================

print("Loading model...")

tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)

# GPT-2 has no pad token by default.
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
)

model.to(DEVICE)
model.eval()

# We do not train the model itself.
# GPT-2 전체는 freeze
for p in model.parameters():
    p.requires_grad_(False)

# Gradient fingerprint로 사용할 parameter만 enable
TARGET_PARAM_NAME = "transformer.h.11.ln_2.weight"

named_params = dict(model.named_parameters())

if TARGET_PARAM_NAME not in named_params:
    raise RuntimeError(
        f"Could not find {TARGET_PARAM_NAME}"
    )

target_param = named_params[TARGET_PARAM_NAME]
target_param.requires_grad_(True)

# ============================================================
# Dataset
# ============================================================

print("Loading FineWeb-Edu...")

dataset = load_dataset(
    DATASET_NAME,
    split="train",
    streaming=True,
)

iterator = iter(dataset)


# ============================================================
# Find two samples with >512 tokens
# ============================================================

def get_long_sample():
    """
    Get a FineWeb-Edu sample whose tokenized length > SEQ_LEN.
    """
    while True:
        example = next(iterator)

        text = example["text"]

        ids = tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0]

        if ids.numel() > SEQ_LEN:
            return text, ids


print("Searching for two samples longer than 512 tokens...")

text1, ids1 = get_long_sample()
text2, ids2 = get_long_sample()

ids1 = ids1[:SEQ_LEN].to(DEVICE)
ids2 = ids2[:SEQ_LEN].to(DEVICE)

print("Sample 1 tokens:", len(ids1))
print("Sample 2 tokens:", len(ids2))


# ============================================================
# LM loss
# ============================================================

def lm_loss_from_input_ids(input_ids):
    """
    Standard causal LM loss.

    input_ids: [1, 512]
    """
    outputs = model(
        input_ids=input_ids,
        labels=input_ids,
        use_cache=False,
    )

    return outputs.loss


# ============================================================
# Select gradient fingerprint
# ============================================================

# Using the complete GPT-2 parameter gradient would require
# enormous second-order computation during GCG.
#
# We therefore use one parameter tensor from the final block.
#
# You can experiment with:
#   model.transformer.h[-1].ln_2.weight
#   model.transformer.h[-1].ln_1.weight
#   model.transformer.h[-1].attn.c_attn.weight
# etc.

TARGET_PARAM_NAME = "transformer.h.11.ln_2.weight"

named_params = dict(model.named_parameters())

if TARGET_PARAM_NAME not in named_params:
    raise RuntimeError(
        f"Could not find {TARGET_PARAM_NAME}"
    )

target_param = named_params[TARGET_PARAM_NAME]

print(
    "Gradient fingerprint:",
    TARGET_PARAM_NAME,
    tuple(target_param.shape),
)


# ============================================================
# Compute reference gradient
# ============================================================

def compute_reference_gradient(input_ids):
    """
    Compute d(loss)/d(theta) for the selected GPT-2 parameter.

    This gradient is detached because the reference gradient
    itself is fixed during GCG.
    """

    input_ids = input_ids.unsqueeze(0)

    loss = lm_loss_from_input_ids(input_ids)

    grad = torch.autograd.grad(
        loss,
        target_param,
        create_graph=False,
        retain_graph=False,
    )[0]

    return grad.detach()


print("\nComputing reference gradient 1...")
g1 = compute_reference_gradient(ids1)

print("Computing reference gradient 2...")
g2 = compute_reference_gradient(ids2)

print("Gradient shape:", tuple(g1.shape))


# ============================================================
# Normalize reference gradients
# ============================================================

def flatten_normalize(x):
    x = x.reshape(-1)
    return x / (x.norm() + 1e-12)


g1_flat = flatten_normalize(g1)
g2_flat = flatten_normalize(g2)

# Combined target.
#
# We use the normalized mean so that the candidate gradient
# is encouraged to resemble BOTH reference gradients.
target_gradient = F.normalize(
    (g1_flat + g2_flat) / 2.0,
    dim=0,
).detach()


# ============================================================
# Embedding access
# ============================================================

embedding_layer = model.transformer.wte
embedding_matrix = embedding_layer.weight.detach()

VOCAB_SIZE = embedding_matrix.shape[0]

print("Vocabulary:", VOCAB_SIZE)


# ============================================================
# Candidate -> gradient
# ============================================================

def candidate_gradient_with_graph(candidate_ids):
    """
    Compute candidate gradient while KEEPING the computation graph.

    This is the key part that makes the optimization
    genuinely gradient-guided.

    We construct the model input through input embeddings,
    so that we can differentiate the gradient fingerprint
    with respect to the input embedding vectors.

    Returns:
        candidate_gradient_flat
        objective
        embedding_inputs
    """

    # [1, 512]
    candidate_ids = candidate_ids.unsqueeze(0)

    # Get embeddings.
    #
    # detach() + requires_grad makes these independent optimization
    # variables while the GPT-2 weights remain frozen.
    embeddings = embedding_layer(candidate_ids).detach()
    embeddings.requires_grad_(True)

    outputs = model(
        inputs_embeds=embeddings,
        labels=candidate_ids,
        use_cache=False,
    )

    loss = outputs.loss

    # First derivative:
    # d loss / d selected_model_parameter
    #
    # create_graph=True is required because the GCG objective
    # must subsequently be differentiated with respect to
    # the input embeddings.
    candidate_grad = torch.autograd.grad(
        loss,
        target_param,
        create_graph=True,
        retain_graph=True,
    )[0]

    candidate_grad_flat = candidate_grad.reshape(-1)

    candidate_grad_norm = (
        candidate_grad_flat.norm() + 1e-12
    )

    candidate_grad_normalized = (
        candidate_grad_flat / candidate_grad_norm
    )

    similarity = torch.sum(
        candidate_grad_normalized * target_gradient
    )

    return similarity, embeddings


# ============================================================
# Fast objective for evaluating concrete token sequences
# ============================================================

@torch.no_grad()
def candidate_similarity(candidate_ids):
    """
    Compute cosine similarity between the candidate's gradient
    and the target gradient.

    This is used to evaluate actual discrete token candidates.
    """

    candidate_ids = candidate_ids.unsqueeze(0)

    outputs = model(
        input_ids=candidate_ids,
        labels=candidate_ids,
        use_cache=False,
    )

    loss = outputs.loss

    grad = torch.autograd.grad(
        loss,
        target_param,
        create_graph=False,
        retain_graph=False,
    )[0]

    grad_flat = grad.reshape(-1)

    grad_flat = grad_flat / (
        grad_flat.norm() + 1e-12
    )

    return torch.sum(
        grad_flat * target_gradient
    ).item()


# ============================================================
# Important:
# torch.no_grad() cannot be used around autograd.grad().
# Replace the above evaluation implementation with this.
# ============================================================

def candidate_similarity(candidate_ids):
    """
    Exact discrete candidate objective.
    """

    candidate_ids = candidate_ids.unsqueeze(0)

    outputs = model(
        input_ids=candidate_ids,
        labels=candidate_ids,
        use_cache=False,
    )

    loss = outputs.loss

    grad = torch.autograd.grad(
        loss,
        target_param,
        create_graph=False,
        retain_graph=False,
    )[0]

    grad_flat = grad.reshape(-1)

    grad_flat = grad_flat / (
        grad_flat.norm() + 1e-12
    )

    similarity = torch.sum(
        grad_flat * target_gradient
    )

    return similarity.item()


# ============================================================
# Candidate initialization
# ============================================================

if INIT_MODE == "sample":
    # Use another FineWeb sample as initialization.
    text_init, init_ids = get_long_sample()
    candidate_ids = init_ids[:SEQ_LEN].clone().to(DEVICE)

else:
    # Random tokens.
    #
    # We avoid EOS as much as possible by sampling from the
    # vocabulary excluding special EOS token.
    candidate_ids = torch.randint(
        low=0,
        high=VOCAB_SIZE,
        size=(SEQ_LEN,),
        device=DEVICE,
        dtype=torch.long,
    )


print("\nInitial candidate similarity...")

current_score = candidate_similarity(candidate_ids)

print(
    f"Initial cosine similarity: {current_score:.6f}"
)


# ============================================================
# GCG coordinate selection
# ============================================================

# ============================================================
# Gradient similarity objective
# ============================================================

def compute_candidate_gradient(candidate_ids):
    """
    Compute:

        g_x = dL(x) / d(theta)

    for the selected fingerprint parameter.

    Returns a detached gradient.
    """

    input_ids = candidate_ids.unsqueeze(0)

    outputs = model(
        input_ids=input_ids,
        labels=input_ids,
        use_cache=False,
    )

    loss = outputs.loss

    grad = torch.autograd.grad(
        outputs=loss,
        inputs=target_param,
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )[0]

    return grad.detach()


def gradient_similarity_from_gradient(candidate_grad):
    """
    Objective:

        J(x) =
            0.5 * cos(g_x, g1)
          + 0.5 * cos(g_x, g2)
    """

    g = candidate_grad.reshape(-1)

    g = g / (
        g.norm() + 1e-12
    )

    sim1 = torch.sum(
        g * g1_flat
    )

    sim2 = torch.sum(
        g * g2_flat
    )

    objective = 0.5 * (
        sim1 + sim2
    )

    return objective


def candidate_objective(candidate_ids):
    """
    Exact objective for a discrete token sequence.

    This performs an actual forward + backward pass.
    """

    candidate_grad = compute_candidate_gradient(
        candidate_ids
    )

    objective = gradient_similarity_from_gradient(
        candidate_grad
    )

    return objective.item()


# ============================================================
# Candidate gradient WITH graph
# ============================================================

def candidate_objective_with_graph(candidate_ids):
    """
    Compute J(x) while retaining the graph needed for:

        dJ / d embedding

    This is the second-order gradient used by GCG.
    """

    input_ids = candidate_ids.unsqueeze(0)

    # [1, 512, hidden]
    embeddings = (
        embedding_layer(input_ids)
        .detach()
    )

    embeddings.requires_grad_(True)

    outputs = model(
        inputs_embeds=embeddings,
        labels=input_ids,
        use_cache=False,
    )

    loss = outputs.loss

    # g_x = dL/dtheta
    #
    # create_graph=True is essential because we need:
    #
    # d J / d embedding
    #
    # where J depends on g_x.
    candidate_grad = torch.autograd.grad(
        outputs=loss,
        inputs=target_param,
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )[0]

    g = candidate_grad.reshape(-1)

    g = g / (
        g.norm() + 1e-12
    )

    sim1 = torch.sum(
        g * g1_flat
    )

    sim2 = torch.sum(
        g * g2_flat
    )

    objective = 0.5 * (
        sim1 + sim2
    )

    # dJ / d embedding
    embedding_grad = torch.autograd.grad(
        outputs=objective,
        inputs=embeddings,
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )[0]

    # [1, 512, hidden]
    # ->
    # [512, hidden]
    embedding_grad = (
        embedding_grad.squeeze(0)
    )

    return (
        objective.detach().item(),
        embedding_grad.detach(),
    )


# ============================================================
# First-order vocabulary scores
# ============================================================

def get_token_scores(
    embedding_gradient,
    current_token,
):
    """
    For one coordinate i:

        J(e_i + delta)
        ≈
        J(e_i)
        +
        <dJ/de_i, delta>

    Therefore:

        score(token)
        =
        <dJ/de_i, E[token] - E[current]>

    The constant current-token term can be ignored
    for ranking.

    We compute scores over the ENTIRE vocabulary.
    """

    # embedding_gradient:
    #
    # [hidden]

    scores = torch.matmul(
        embedding_matrix,
        embedding_gradient,
    )

    # Do not propose the current token.
    scores[current_token] = -float("inf")

    return scores


# ============================================================
# Top-k proposals for ONE coordinate
# ============================================================

def get_coordinate_candidates(
    position,
    embedding_grads,
    top_k,
    candidate_ids,
):
    """
    Return top-k vocabulary candidates for a coordinate.
    """

    current_token = int(
        candidate_ids[position].item()
    )

    grad_i = embedding_grads[position]

    scores = get_token_scores(
        embedding_gradient=grad_i,
        current_token=current_token,
    )

    top_scores, top_tokens = torch.topk(
        scores,
        k=top_k,
        largest=True,
    )

    return (
        top_tokens,
        top_scores,
    )


# ============================================================
# Select multiple coordinates
# ============================================================

def select_coordinates(
    embedding_grads,
    candidate_ids,
    num_coordinates,
    excluded_positions=None,
):
    """
    Select coordinates with largest ||dJ/de_i||.

    excluded_positions:
        positions that have already been tried during the
        current greedy round.
    """

    position_scores = (
        embedding_grads.norm(dim=-1)
    )

    position_scores = position_scores.clone()

    if excluded_positions is not None:

        for p in excluded_positions:

            position_scores[p] = -float("inf")

    num_available = (
        position_scores
        .ne(-float("inf"))
        .sum()
        .item()
    )

    k = min(
        num_coordinates,
        num_available,
    )

    _, positions = torch.topk(
        position_scores,
        k=k,
    )

    return positions


# ============================================================
# Main multi-coordinate GCG
# ============================================================

print("\nStarting multi-coordinate GCG...\n")


# ------------------------------------------------------------
# Initial objective
# ------------------------------------------------------------

current_score = candidate_objective(
    candidate_ids
)

best_score = current_score
best_ids = candidate_ids.clone()

print(
    f"Initial objective: "
    f"{current_score:.8f}"
)


# ------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------

NUM_STEPS = 30

# Number of high-gradient coordinates considered
# in each greedy round.
NUM_COORDINATES = 16

# Number of vocabulary candidates per coordinate.
TOP_K = 32

# Maximum number of unsuccessful coordinate trials
# before declaring convergence.
MAX_FAILED_COORDINATES = 64


# ------------------------------------------------------------
# GCG iterations
# ------------------------------------------------------------

for step in range(NUM_STEPS):

    print()
    print("=" * 70)
    print(
        f"GCG ITERATION "
        f"{step + 1}/{NUM_STEPS}"
    )
    print("=" * 70)


    # ========================================================
    # Compute current second-order GCG gradient
    # ========================================================

    surrogate_score, embedding_grads = (
        candidate_objective_with_graph(
            candidate_ids
        )
    )

    print(
        f"Current objective: "
        f"{current_score:.8f}"
    )

    print(
        f"Surrogate objective: "
        f"{surrogate_score:.8f}"
    )


    # ========================================================
    # Greedy coordinate search
    # ========================================================

    tried_positions = set()

    iteration_improved = False

    failed_coordinates = 0


    while (
        len(tried_positions) < SEQ_LEN
        and
        failed_coordinates
        < MAX_FAILED_COORDINATES
    ):


        # ----------------------------------------------------
        # Choose several coordinates that have not yet
        # been attempted during this iteration.
        # ----------------------------------------------------

        positions = select_coordinates(
            embedding_grads=embedding_grads,
            candidate_ids=candidate_ids,
            num_coordinates=NUM_COORDINATES,
            excluded_positions=tried_positions,
        )


        if positions.numel() == 0:
            break


        # ====================================================
        # Build ALL top-k proposals
        # ====================================================

        proposals = []


        for position_tensor in positions:

            position = int(
                position_tensor.item()
            )

            tried_positions.add(position)

            top_tokens, top_scores = (
                get_coordinate_candidates(
                    position=position,
                    embedding_grads=embedding_grads,
                    top_k=TOP_K,
                    candidate_ids=candidate_ids,
                )
            )


            print(
                f"\nCoordinate {position}: "
                f"{top_tokens.numel()} proposals"
            )


            for rank in range(
                top_tokens.numel()
            ):

                token = int(
                    top_tokens[rank].item()
                )

                first_order_score = float(
                    top_scores[rank].item()
                )

                proposals.append(
                    (
                        position,
                        token,
                        first_order_score,
                    )
                )


        # ====================================================
        # Sort proposals by first-order score
        #
        # We still evaluate them EXACTLY below.
        # ====================================================

        proposals.sort(
            key=lambda x: x[2],
            reverse=True,
        )


        print(
            f"\nTotal discrete proposals: "
            f"{len(proposals)}"
        )


        # ====================================================
        # EXACT evaluation
        #
        # Every proposal gets:
        #
        #   forward
        #   loss
        #   dloss/dtheta
        #   cosine similarity
        #
        # No surrogate score is used for the final decision.
        # ====================================================

        best_local_score = current_score
        best_local_position = None
        best_local_token = None


        for proposal_idx, (
            position,
            token,
            first_order_score,
        ) in enumerate(proposals):


            trial_ids = (
                candidate_ids.clone()
            )

            old_token = int(
                trial_ids[position].item()
            )

            trial_ids[position] = token


            exact_score = candidate_objective(
                trial_ids
            )


            print(
                f"[{proposal_idx + 1:4d}/"
                f"{len(proposals):4d}] "
                f"pos={position:3d} "
                f"token={token:5d} "
                f"FO={first_order_score:+.6e} "
                f"EXACT={exact_score:+.8f}"
            )


            # ------------------------------------------------
            # Greedy best candidate
            # ------------------------------------------------

            if exact_score > best_local_score:

                best_local_score = exact_score

                best_local_position = (
                    position
                )

                best_local_token = (
                    token
                )


        # ====================================================
        # Commit BEST exact candidate
        # ====================================================

        if best_local_position is not None:

            position = (
                best_local_position
            )

            new_token = (
                best_local_token
            )

            old_token = int(
                candidate_ids[position].item()
            )


            candidate_ids[position] = (
                new_token
            )

            current_score = (
                best_local_score
            )

            iteration_improved = True


            print()
            print(
                "GREEDY UPDATE"
            )

            print(
                f"position: "
                f"{position}"
            )

            print(
                f"token: "
                f"{old_token} -> "
                f"{new_token}"
            )

            print(
                f"objective: "
                f"{current_score:.8f}"
            )


            # ------------------------------------------------
            # We changed the sequence.
            #
            # Therefore the gradient landscape changed.
            #
            # Recompute dJ/de for the NEW sequence.
            # ------------------------------------------------

            surrogate_score, embedding_grads = (
                candidate_objective_with_graph(
                    candidate_ids
                )
            )


            # ------------------------------------------------
            # Reset coordinate search because a token change
            # changes all coordinate gradients.
            # ------------------------------------------------

            tried_positions = set()

            failed_coordinates = 0


            # ------------------------------------------------
            # Update global best
            # ------------------------------------------------

            if current_score > best_score:

                best_score = (
                    current_score
                )

                best_ids = (
                    candidate_ids.clone()
                )

                print()
                print(
                    "***** NEW GLOBAL BEST *****"
                )

                print(
                    f"score = "
                    f"{best_score:.8f}"
                )


            # ------------------------------------------------
            # Greedy means one update per round.
            # Restart from the new sequence.
            # ------------------------------------------------

            break


        else:

            # No exact candidate among this group improved
            # the objective.
            #
            # DO NOT terminate.
            #
            # Continue with another set of coordinates.

            failed_coordinates += (
                len(positions)
            )

            print()
            print(
                "No improvement from "
                f"these {len(positions)} coordinates."
            )

            print(
                f"Trying different coordinates... "
                f"failed={failed_coordinates}"
            )


    # ========================================================
    # End of this GCG iteration
    # ========================================================

    print()
    print(
        f"Iteration {step + 1} finished."
    )

    print(
        f"Current score: "
        f"{current_score:.8f}"
    )

    print(
        f"Global best: "
        f"{best_score:.8f}"
    )


    # ========================================================
    # Decode current candidate
    # ========================================================

    current_text = tokenizer.decode(
        candidate_ids.detach().cpu().tolist(),
        skip_special_tokens=True,
    )

    print()
    print(
        "Current candidate:"
    )

    print(
        current_text[:1000]
    )


    # ========================================================
    # Convergence condition
    # ========================================================

    if not iteration_improved:

        print()
        print(
            "No coordinate in the entire search "
            "round improved the objective."
        )

        print(
            "GCG converged."
        )

        break


# ============================================================
# Final result
# ============================================================

print()
print("=" * 70)
print("GCG FINISHED")
print("=" * 70)

print(
    f"Best objective: "
    f"{best_score:.8f}"
)


final_text = tokenizer.decode(
    best_ids.detach().cpu().tolist(),
    skip_special_tokens=True,
)


print()
print("Final candidate:")
print(final_text)


# ============================================================
# Final verification
# ============================================================

final_grad = compute_candidate_gradient(
    best_ids
)

final_cos_1 = F.cosine_similarity(
    final_grad.reshape(1, -1),
    g1.reshape(1, -1),
).item()

final_cos_2 = F.cosine_similarity(
    final_grad.reshape(1, -1),
    g2.reshape(1, -1),
).item()


print()
print("=" * 70)
print("FINAL VERIFICATION")
print("=" * 70)

print(
    f"Cos(candidate, sample 1): "
    f"{final_cos_1:.8f}"
)

print(
    f"Cos(candidate, sample 2): "
    f"{final_cos_2:.8f}"
)

print(
    f"Mean cosine similarity: "
    f"{(final_cos_1 + final_cos_2) / 2:.8f}"
)
