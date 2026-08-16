# ==============================================================================
# GRU vs RWKV-7 on IMDB — single file
# init / lr schedule / optimizer groups: RWKV-LM/RWKV-v7/train_temp 공식 구현 반영
# kernel: wind_backstepping (fp32, inline hardcoded)
# ==============================================================================
# !pip install -q -U datasets huggingface-hub ninja
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import BertTokenizer
from tqdm import tqdm

BATCH_SIZE = 32
MAX_LENGTH = 256              # % CHUNK_LEN == 0
EPOCHS = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HEAD_SIZE = 64
CHUNK_LEN = 16

# --- train_temp demo 하이퍼파라미터 ---
LR_INIT = 6e-4                # train.py 기본값 (L12-D768 기준)
LR_FINAL = 1e-5
WARMUP_STEPS = 10
BETAS = (0.9, 0.99)
ADAM_EPS = 1e-18
WEIGHT_DECAY = 1e-3           # demo-training-run.sh

# ============== wind_backstepping CUDA kernel (fp32 하드코딩) ==============
CUDA_SRC = r"""
#include <assert.h>
using bf = float;
__device__ inline float to_float(const bf & u) { return u; }
__device__ inline bf to_bf(const float & u) { return u; }
typedef bf * __restrict__ F_;

__global__ void forward_kernel(int T, int H, F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_, bf* y_, float* s_, float* sa_) {
    constexpr int C = _C_;
    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;
    float state[C] = {0};
    __shared__ float q[C], k[C], w[C], a[C], b[C];
    for (int t = 0; t < T; t++) {
        int ind = bb*T*H*C + t*H*C + hh * C + i;
        __syncthreads();
        q[i] = to_float(q_[ind]);
        w[i] = __expf(-__expf(to_float(w_[ind])));
        k[i] = to_float(k_[ind]);
        a[i] = to_float(a_[ind]);
        b[i] = to_float(b_[ind]);
        __syncthreads();
        float sa = 0;
#pragma unroll
        for (int j = 0; j < C; j++) sa += a[j] * state[j];
        sa_[ind] = sa;
        float v = to_float(v_[ind]);
        float y = 0;
#pragma unroll
        for (int j = 0; j < C; j++) {
            float& s = state[j];
            s = s * w[j] + sa * b[j] + k[j] * v;
            y += s * q[j];
        }
        y_[ind] = to_bf(y);
        if ((t+1)%_CHUNK_LEN_ == 0) {
            int base = (bb*H+hh)*(T/_CHUNK_LEN_)*C*C + (t/_CHUNK_LEN_)*C*C + i;
#pragma unroll
            for (int j = 0; j < C; j++) s_[base + j*C] = state[j];
        }
    }
}

__global__ void backward_kernel(int T, int H, F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_, F_ dy_, float * __restrict__ s_, float * __restrict__ sa_, bf* dw_, bf* dq_, bf* dk_, bf* dv_, bf* da_, bf* db_) {
    constexpr int C = _C_;
    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;
    float stateT[C] = {0}, dstate[C] = {0}, dstateT[C] = {0};
    __shared__ float w[C], q[C], k[C], v[C], a[C], b[C], dy[C], sa[C], dSb_shared[C];
    float qi, wi, ki, ai, bi, dyi;
    for (int t = T-1; t >= 0; t--) {
        int ind = bb*T*H*C + t*H*C + hh * C + i;
        __syncthreads();
        q[i] = qi = to_float(q_[ind]);
        float wi_fac = -__expf(to_float(w_[ind]));
        w[i] = wi = __expf(wi_fac);
        k[i] = ki = to_float(k_[ind]);
        a[i] = ai = to_float(a_[ind]);
        b[i] = bi = to_float(b_[ind]);
        v[i] = to_float(v_[ind]);
        dy[i] = dyi = to_float(dy_[ind]);
        sa[i] = sa_[ind];
        __syncthreads();
        if ((t+1)%_CHUNK_LEN_ == 0) {
            int base = (bb*H+hh)*(T/_CHUNK_LEN_)*C*C + (t/_CHUNK_LEN_)*C*C + i*C;
#pragma unroll
            for (int j = 0; j < C; j++) stateT[j] = s_[base + j];
        }
        float dq = 0;
#pragma unroll
        for (int j = 0; j < C; j++) dq += stateT[j]*dy[j];
        dq_[ind] = to_bf(dq);
        float iwi = 1.0f/wi;
#pragma unroll
        for (int j = 0; j < C; j++) {
            stateT[j] = (stateT[j] - ki*v[j] - bi*sa[j]) * iwi;
            dstate[j] += dyi * q[j];
            dstateT[j] += qi * dy[j];
        }
        float dw = 0, dk = 0, dv = 0, db = 0, dSb = 0;
#pragma unroll
        for (int j = 0; j < C; j++) {
            dw += dstateT[j]*stateT[j];
            dk += dstateT[j]*v[j];
            dv += dstate[j]*k[j];
            dSb += dstate[j]*b[j];
            db += dstateT[j]*sa[j];
        }
        dw_[ind] = to_bf(dw * wi * wi_fac);
        dk_[ind] = to_bf(dk);
        dv_[ind] = to_bf(dv);
        db_[ind] = to_bf(db);
        __syncthreads();
        dSb_shared[i] = dSb;
        __syncthreads();
        float da = 0;
#pragma unroll
        for (int j = 0; j < C; j++) da += stateT[j]*dSb_shared[j];
        da_[ind] = to_bf(da);
#pragma unroll
        for (int j = 0; j < C; j++) {
            dstate[j] = dstate[j]*w[j] + dSb * a[j];
            dstateT[j] = dstateT[j]*wi + ai * dSb_shared[j];
        }
    }
}

void cuda_forward(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*z, bf*a, bf*y, float*s, float*sa) {
    forward_kernel<<<dim3(H,B), dim3(_C_)>>>(T,H,w,q,k,v,z,a,y,s,sa);
}
void cuda_backward(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*z, bf*a, bf*dy, float*s, float*sa, bf*dw, bf*dq, bf*dk, bf*dv, bf*dz, bf*da) {
    assert(T%_CHUNK_LEN_ == 0);
    backward_kernel<<<dim3(H,B), dim3(_C_)>>>(T,H,w,q,k,v,z,a,dy,s,sa,dw,dq,dk,dv,dz,da);
}
"""

CPP_SRC = r"""
#include <torch/extension.h>
using bf = float;
void cuda_forward(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*z, bf*a, bf*y, float*s, float*sa);
void forward(torch::Tensor &w, torch::Tensor &q, torch::Tensor &k, torch::Tensor &v, torch::Tensor &z, torch::Tensor &a, torch::Tensor &y, torch::Tensor &s, torch::Tensor &sa) {
    int B = w.sizes()[0], T = w.sizes()[1], H = w.sizes()[2];
    cuda_forward(B, T, H, (bf*)w.data_ptr(), (bf*)q.data_ptr(), (bf*)k.data_ptr(), (bf*)v.data_ptr(), (bf*)z.data_ptr(), (bf*)a.data_ptr(), (bf*)y.data_ptr(), (float*)s.data_ptr(), (float*)sa.data_ptr());
}
void cuda_backward(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*z, bf*a, bf*dy, float*s, float*sa, bf*dw, bf*dq, bf*dk, bf*dv, bf*dz, bf*da);
void backward(torch::Tensor &w, torch::Tensor &q, torch::Tensor &k, torch::Tensor &v, torch::Tensor &z, torch::Tensor &a, torch::Tensor &dy,
        torch::Tensor &s, torch::Tensor &sa, torch::Tensor &dw, torch::Tensor &dq, torch::Tensor &dk, torch::Tensor &dv, torch::Tensor &dz, torch::Tensor &da) {
    int B = w.sizes()[0], T = w.sizes()[1], H = w.sizes()[2];
    cuda_backward(B, T, H, (bf*)w.data_ptr(), (bf*)q.data_ptr(), (bf*)k.data_ptr(), (bf*)v.data_ptr(), (bf*)z.data_ptr(), (bf*)a.data_ptr(), (bf*)dy.data_ptr(),
            (float*)s.data_ptr(), (float*)sa.data_ptr(), (bf*)dw.data_ptr(), (bf*)dq.data_ptr(), (bf*)dk.data_ptr(), (bf*)dv.data_ptr(), (bf*)dz.data_ptr(), (bf*)da.data_ptr());
}
TORCH_LIBRARY(wind_backstepping, m) {
    m.def("forward(Tensor w, Tensor q, Tensor k, Tensor v, Tensor z, Tensor a, Tensor(a!) y, Tensor(b!) s, Tensor(c!) sa) -> ()");
    m.def("backward(Tensor w, Tensor q, Tensor k, Tensor v, Tensor z, Tensor a, Tensor dy, Tensor s, Tensor sa, Tensor(a!) dw, Tensor(b!) dq, Tensor(c!) dk, Tensor(d!) dv, Tensor(e!) dz, Tensor(f!) da) -> ()");
}
TORCH_LIBRARY_IMPL(wind_backstepping, CUDA, m) {
    m.impl("forward", &forward);
    m.impl("backward", &backward);
}
"""

USE_KERNEL = False
if torch.cuda.is_available():
    try:
        from torch.utils.cpp_extension import load_inline
        load_inline(
            name="wind_backstepping_fp32",
            cpp_sources=[CPP_SRC],
            cuda_sources=[CUDA_SRC],
            extra_cuda_cflags=["-O3", "--use_fast_math",
                               f"-D_C_={HEAD_SIZE}", f"-D_CHUNK_LEN_={CHUNK_LEN}"],
            is_python_module=False,
            verbose=False,
        )
        USE_KERNEL = True
        print("[kernel] wind_backstepping compiled, using CUDA path")
    except Exception as e:
        print(f"[kernel] compile failed ({e}), falling back to python loop")


class WindBackstepping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, q, k, v, z, b):
        B, T, H, C = w.shape
        assert T % CHUNK_LEN == 0
        assert all(t.dtype == torch.float32 and t.is_contiguous() for t in [w, q, k, v, z, b])
        y = torch.empty_like(v)
        s = torch.empty(B, H, T // CHUNK_LEN, C, C, dtype=torch.float32, device=w.device)
        sa = torch.empty(B, T, H, C, dtype=torch.float32, device=w.device)
        torch.ops.wind_backstepping.forward(w, q, k, v, z, b, y, s, sa)
        ctx.save_for_backward(w, q, k, v, z, b, s, sa)
        return y

    @staticmethod
    def backward(ctx, dy):
        w, q, k, v, z, b, s, sa = ctx.saved_tensors
        dy = dy.contiguous()
        dw, dq, dk, dv, dz, db = [torch.empty_like(x) for x in [w, q, k, v, z, b]]
        torch.ops.wind_backstepping.backward(w, q, k, v, z, b, dy, s, sa, dw, dq, dk, dv, dz, db)
        return dw, dq, dk, dv, dz, db


def rwkv7_attn(w_pre, r, k, v, kk, a, B, T, H, N):
    """decay = exp(-exp(w_pre)) (= 공식 clampw의 exp(-sigmoid(u)·e^-0.5), w_pre = -softplus(-u)-0.5)"""
    if USE_KERNEL:
        z = (-kk).contiguous()
        b = (kk * a).contiguous()
        y = WindBackstepping.apply(w_pre.contiguous(), r.contiguous(), k.contiguous(),
                                   v.contiguous(), z, b)
        return y.view(B, T, H * N)
    w = torch.exp(-torch.exp(w_pre))
    S = torch.zeros(B, H, N, N, device=r.device, dtype=r.dtype)
    ys = []
    for t in range(T):
        rt, wt, kt, vt, at, kkt = [x[:, t].unsqueeze(-1) for x in (r, w, k, v, a, kk)]
        S = S * wt.transpose(-1, -2) - (S @ kkt) @ (kkt * at).transpose(-1, -2) + vt @ kt.transpose(-1, -2)
        ys.append((S @ rt).view(B, H * N))
    return torch.stack(ys, dim=1)

# ============== DATASET ==============
class IMDBDataset(Dataset):
    def __init__(self, split="train", max_length=MAX_LENGTH):
        self.data = load_dataset("stanfordnlp/imdb", split=split)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        enc = self.tokenizer(item["text"], max_length=self.max_length,
                             padding="max_length", truncation=True, return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "label": torch.tensor(item["label"], dtype=torch.long)}

# ============== MODEL 1: Standard GRU (파라미터 매칭) ==============
class StandardGRU(nn.Module):
    def __init__(self, vocab_size, hidden_size=792, output_size=2, emb_dim=512, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        _, h_n = self.gru(x)
        return self.fc(h_n[-1])

# ============== MODEL 2: RWKV-7 (train_temp 공식 init) ==============
def ortho_init(x, scale):
    """train_temp의 ortho_init 그대로"""
    with torch.no_grad():
        shape = x.shape
        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
        nn.init.orthogonal_(x, gain=gain * scale)
    return x


class RWKV7TimeMix(nn.Module):
    def __init__(self, dim, layer_id, n_layer, head_size=HEAD_SIZE):
        super().__init__()
        assert dim % head_size == 0
        self.layer_id = layer_id
        self.n_head = dim // head_size
        self.head_size = head_size
        C, H, N = dim, self.n_head, head_size

        with torch.no_grad():
            ratio_0_to_1 = layer_id / max(n_layer - 1, 1)          # 0 → 1
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)        # 1 → ~0
            ddd = torch.arange(C, dtype=torch.float32) / C

            # token shift mix (공식 power-curve init)
            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_v = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            # decay/a/v 초기값 커브 (공식 www/zigzag/linear)
            n_idx = torch.arange(C, dtype=torch.float32)
            linear = n_idx / (C - 1) - 0.5
            zig = ((n_idx % N) - (N - 1) / 2) / ((N - 1) / 2)
            zigzag = zig * zig.abs()
            www = -6 + 6 * (n_idx / (C - 1)) ** (1 + 1 * ratio_0_to_1 ** 0.3)

            # LoRA dims (공식 suggestion 공식)
            D_DECAY = max(32, int(round((2.5 * C**0.5) / 32) * 32))
            D_AAA   = max(32, int(round((2.5 * C**0.5) / 32) * 32))
            D_MV    = max(32, int(round((1.7 * C**0.5) / 32) * 32))
            D_GATE  = max(32, int(round((5.0 * C**0.5) / 32) * 32))

            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY, C), 0.1))
            self.w0 = nn.Parameter(www + 0.5 + zigzag * 2.5)

            self.a1 = nn.Parameter(torch.zeros(C, D_AAA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(C) - 0.19 + zigzag * 0.3 + linear * 0.4)

            if layer_id > 0:
                self.v1 = nn.Parameter(torch.zeros(C, D_MV))
                self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV, C), 0.1))
                self.v0 = nn.Parameter(torch.zeros(C) + 0.73 - linear * 0.4)

            self.g1 = nn.Parameter(torch.zeros(C, D_GATE))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE, C), 0.1))

            self.k_k = nn.Parameter(torch.zeros(C) + 0.71 - linear * 0.1)
            self.k_a = nn.Parameter(torch.zeros(C) + 1.02)
            self.r_k = nn.Parameter(torch.zeros(H, N) - 0.04)

            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            self.ln_x = nn.GroupNorm(H, C, eps=64e-5)

            self.receptance.weight.data.uniform_(-0.5 / (C**0.5), 0.5 / (C**0.5))
            self.key.weight.data.uniform_(-0.05 / (C**0.5), 0.05 / (C**0.5))   # 10배 작음!
            self.value.weight.data.uniform_(-0.5 / (C**0.5), 0.5 / (C**0.5))
            self.output.weight.data.zero_()
            # ln_x.weight = ((1+layer)/n_layer)^0.7 (generate_init_weight)
            self.ln_x.weight.data.fill_(((1 + layer_id) / n_layer) ** 0.7)
            self.ln_x.bias.data.zero_()

    def forward(self, x, v_first):
        B, T, C = x.shape
        H, N = self.n_head, self.head_size

        xx = F.pad(x, (0, 0, 1, -1)) - x
        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        u = self.w0 + torch.tanh(xw @ self.w1) @ self.w2      # raw decay logits
        w_pre = -F.softplus(-u) - 0.5                          # ≡ 공식 clampw
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, N), dim=-1, p=2.0, eps=1e-12)
        k = k * (1 + (a - 1) * self.k_a)                       # 공식 형태

        r_, w_, k_, v_, a_ = [t.view(B, T, H, N) for t in (r, w_pre, k, v, a)]
        y = rwkv7_attn(w_, r_, k_, v_, kk, a_, B, T, H, N)

        y = self.ln_x(y.reshape(B * T, C)).view(B, T, C)
        y = y + ((r_ * k_ * self.r_k).sum(dim=-1, keepdim=True) * v_).view(B, T, C)
        return self.output(y * g), v_first


class RWKV7ChannelMix(nn.Module):
    def __init__(self, dim, layer_id, n_layer):
        super().__init__()
        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)
            ddd = torch.arange(dim, dtype=torch.float32) / dim
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0 ** 4))
        self.key = nn.Linear(dim, dim * 4, bias=False)
        self.value = nn.Linear(dim * 4, dim, bias=False)
        self.key.weight.data.uniform_(-0.5 / (dim**0.5), 0.5 / (dim**0.5))
        self.value.weight.data.zero_()

    def forward(self, x):
        xx = F.pad(x, (0, 0, 1, -1)) - x
        k = torch.relu(self.key(x + xx * self.x_k)) ** 2
        return self.value(k)


class RWKV7Block(nn.Module):
    def __init__(self, dim, layer_id, n_layer):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.att = RWKV7TimeMix(dim, layer_id, n_layer)
        self.ffn = RWKV7ChannelMix(dim, layer_id, n_layer)

    def forward(self, x, v_first):
        dx, v_first = self.att(self.ln1(x), v_first)
        x = x + dx
        x = x + self.ffn(self.ln2(x))
        return x, v_first


class RWKV7Classifier(nn.Module):
    def __init__(self, vocab_size, hidden_size=512, output_size=2, layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.ln0 = nn.LayerNorm(hidden_size)
        self.blocks = nn.ModuleList([RWKV7Block(hidden_size, i, layers) for i in range(layers)])
        self.ln_out = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, output_size, bias=False)
        with torch.no_grad():
            # emb: uniform ±1e-4 (뒤에 ln0가 스케일 복구)
            nn.init.uniform_(self.embedding.weight, a=-1e-4, b=1e-4)
            # head: ortho, vocab<n_embd 케이스라 gain 0.5
            nn.init.orthogonal_(self.head.weight, gain=0.5)

    def forward(self, x):
        x = self.ln0(self.embedding(x))
        v_first = None
        for block in self.blocks:
            x, v_first = block(x, v_first)
        x = self.ln_out(x)
        return self.head(x.mean(dim=1))

# ============== OPTIMIZER (공식 param group 정책) ==============
def configure_rwkv_optimizer(model, lr_init=LR_INIT, weight_decay=WEIGHT_DECAY):
    """att.w0 → 2x lr / 2D+ .weight → weight decay / 나머지 → 1x no-wd"""
    lr_1x, lr_2x, lr_decay = [], [], []
    for n, p in model.named_parameters():
        if "att.w0" in n:
            lr_2x.append(p)
        elif p.squeeze().dim() >= 2 and weight_decay > 0 and ".weight" in n:
            lr_decay.append(p)
        else:
            lr_1x.append(p)
    groups = [
        {"params": lr_1x, "weight_decay": 0.0, "lr_scale": 1.0},
        {"params": lr_2x, "weight_decay": 0.0, "lr_scale": 2.0},
        {"params": lr_decay, "weight_decay": weight_decay, "lr_scale": 1.0},
    ]
    return optim.AdamW(groups, lr=lr_init, betas=BETAS, eps=ADAM_EPS)


def lr_at_step(step, total_steps, lr_init=LR_INIT, lr_final=LR_FINAL, warmup=WARMUP_STEPS):
    """공식 trainer.py: warmup(0.01→1) 후 cosine lr_init→lr_final"""
    if step < warmup:
        return lr_init * (0.01 + 0.99 * step / warmup)
    progress = (step - warmup) / max(total_steps - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    ff = lr_final / lr_init
    mult = (0.5 + ff / 2) + (0.5 - ff / 2) * math.cos(math.pi * progress)
    return lr_init * mult


def set_lr(opt, lr):
    for g in opt.param_groups:
        g["lr"] = lr * g.get("lr_scale", 1.0)

# ============== TRAIN / EVAL ==============
def train_epoch(model, loader, opt, name, step_state=None, total_steps=None):
    model.train()
    total = 0
    for batch in tqdm(loader, desc=f"{name} train"):
        input_ids = batch["input_ids"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        if step_state is not None:  # RWKV: 공식 스케줄
            set_lr(opt, lr_at_step(step_state[0], total_steps))
            step_state[0] += 1
        opt.zero_grad()
        out = model(input_ids)
        loss = nn.CrossEntropyLoss()(out, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item()
    return total / len(loader)

@torch.no_grad()
def evaluate_model(model, loader, name):
    model.eval()
    correct = total = 0
    for batch in tqdm(loader, desc=f"{name} eval"):
        input_ids = batch["input_ids"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        preds = torch.argmax(model(input_ids), dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

# ============== RUN ==============
def main():
    train_ds = IMDBDataset("train")
    test_ds = IMDBDataset("test")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=2)

    vocab_size = train_ds.tokenizer.vocab_size
    total_steps = len(train_loader) * EPOCHS
    results = {}

    print("\n" + "=" * 50)
    print("1) STANDARD GRU (2-layer, hidden=792, emb=512)")
    m = StandardGRU(vocab_size).to(DEVICE)
    print(f"   params: {count_params(m)/1e6:.2f}M")
    opt = optim.AdamW(m.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY)
    for e in range(EPOCHS):
        loss = train_epoch(m, train_loader, opt, "GRU")
        acc = evaluate_model(m, test_loader, "GRU")
        results["Standard GRU"] = acc
        print(f"   Epoch {e+1} | Loss: {loss:.4f} | Acc: {acc:.2f}%")

    print("\n" + "=" * 50)
    print(f"2) RWKV-7 (official init/lr, 2-layer, hidden=512, kernel={USE_KERNEL})")
    m = RWKV7Classifier(vocab_size, hidden_size=512, layers=2).to(DEVICE)
    print(f"   params: {count_params(m)/1e6:.2f}M")
    opt = configure_rwkv_optimizer(m)
    step_state = [0]
    for e in range(EPOCHS):
        loss = train_epoch(m, train_loader, opt, "RWKV7", step_state, total_steps)
        acc = evaluate_model(m, test_loader, "RWKV7")
        results["RWKV-7"] = acc
        print(f"   Epoch {e+1} | Loss: {loss:.4f} | Acc: {acc:.2f}%")

    print("\n" + "=" * 50)
    print("FINAL COMPARISON (IMDB test accuracy, 7 epochs)")
    print("-" * 50)
    for k, v in results.items():
        print(f"{k:<25} -> {v:.2f}%")
    print("=" * 50)

if __name__ == "__main__":
    main()
