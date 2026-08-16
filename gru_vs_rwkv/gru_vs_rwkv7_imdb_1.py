# ============== SETUP ==============
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
MAX_LENGTH = 256          # CHUNK_LEN(16)의 배수여야 함
EPOCHS = 7
LR = 5e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HEAD_SIZE = 64
CHUNK_LEN = 16

# ============== wind_backstepping CUDA kernel (하드코딩, fp32 버전) ==============
CUDA_SRC = r"""
#include <cuda_bf16.h>
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
        for (int j = 0; j < C; j++) {
            sa += a[j] * state[j];
        }
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
            for (int j = 0; j < C; j++) {
                s_[base + j*C] = state[j];
            }
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
            for (int j = 0; j < C; j++) {
                stateT[j] = s_[base + j];
            }
        }

        float dq = 0;
#pragma unroll
        for (int j = 0; j < C; j++) {
            dq += stateT[j]*dy[j];
        }
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
        for (int j = 0; j < C; j++) {
            da += stateT[j]*dSb_shared[j];
        }
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
            extra_cuda_cflags=[
                "-O3", "--use_fast_math",
                f"-D_C_={HEAD_SIZE}", f"-D_CHUNK_LEN_={CHUNK_LEN}",
            ],
            is_python_module=False,
            verbose=False,
        )
        USE_KERNEL = True
        print("[kernel] wind_backstepping compiled, using CUDA path")
    except Exception as e:
        print(f"[kernel] compile failed ({e}), falling back to python loop")


class WindBackstepping(torch.autograd.Function):
    """입력 전부 (B,T,H,C) fp32 contiguous. w는 pre-decay: 실제 decay = exp(-exp(w))"""
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
    """
    S = S*diag(exp(-exp(w_pre))) - S@kk (kk*a)^T + v k^T ;  y = S@r
    커널 매핑: q=r, z=-kk, b=kk*a  (sa = z·state = -[S@kk])
    """
    if USE_KERNEL:
        z = (-kk).contiguous()
        b = (kk * a).contiguous()
        y = WindBackstepping.apply(w_pre.contiguous(), r.contiguous(), k.contiguous(),
                                   v.contiguous(), z, b)
        return y.view(B, T, H * N)
    # ---- python fallback (수학적으로 동일) ----
    w = torch.exp(-torch.exp(w_pre))
    S = torch.zeros(B, H, N, N, device=r.device, dtype=r.dtype)
    ys = []
    for t in range(T):
        rt, wt, kt, vt, at, kkt = [x[:, t].unsqueeze(-1) for x in (r, w, k, v, a, kk)]
        S = S * wt.transpose(-1, -2) \
            - (S @ kkt) @ (kkt * at).transpose(-1, -2) \
            + vt @ kt.transpose(-1, -2)
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
        enc = self.tokenizer(
            item["text"], max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "label": torch.tensor(item["label"], dtype=torch.long),
        }

# ============== MODEL 1: Standard GRU (파라미터 매칭: 22.50M) ==============
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

# ============== MODEL 2: RWKV-7 ==============
class RWKV7TimeMix(nn.Module):
    def __init__(self, dim, layer_id, head_size=HEAD_SIZE):
        super().__init__()
        assert dim % head_size == 0
        self.layer_id = layer_id
        self.n_head = dim // head_size
        self.head_size = head_size
        C, H, N = dim, self.n_head, head_size

        for name in ["x_r", "x_w", "x_k", "x_v", "x_a", "x_g"]:
            setattr(self, name, nn.Parameter(torch.full((C,), 0.5)))

        D_w, D_a, D_v, D_g = 64, 64, 32, 128
        self.Ww1 = nn.Parameter(torch.zeros(C, D_w))
        self.Ww2 = nn.Parameter(torch.randn(D_w, C) * 0.01)
        self.w_bias = nn.Parameter(torch.zeros(C))

        self.Wa1 = nn.Parameter(torch.zeros(C, D_a))
        self.Wa2 = nn.Parameter(torch.randn(D_a, C) * 0.01)
        self.a_bias = nn.Parameter(torch.zeros(C))

        if layer_id > 0:
            self.Wv1 = nn.Parameter(torch.zeros(C, D_v))
            self.Wv2 = nn.Parameter(torch.randn(D_v, C) * 0.01)
            self.v_bias = nn.Parameter(torch.zeros(C))

        self.Wg1 = nn.Parameter(torch.zeros(C, D_g))
        self.Wg2 = nn.Parameter(torch.randn(D_g, C) * 0.01)

        self.k_k = nn.Parameter(torch.full((C,), 0.85))
        self.k_a = nn.Parameter(torch.ones(C))
        self.r_k = nn.Parameter(torch.zeros(H, N))

        self.Wr = nn.Linear(C, C, bias=False)
        self.Wk = nn.Linear(C, C, bias=False)
        self.Wv = nn.Linear(C, C, bias=False)
        self.Wo = nn.Linear(C, C, bias=False)
        nn.init.zeros_(self.Wo.weight)

        self.ln = nn.GroupNorm(H, C, eps=64e-5)

    def forward(self, x, v0):
        B, T, C = x.shape
        H, N = self.n_head, self.head_size

        xx = F.pad(x, (0, 0, 1, -1)) - x
        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.Wr(xr)
        # pre-decay: 실제 decay = exp(-exp(w_pre)) = exp(-sigmoid(u)/sqrt(e))
        u = torch.tanh(xw @ self.Ww1) @ self.Ww2 + self.w_bias
        w_pre = -F.softplus(-u) - 0.5
        k = self.Wk(xk)
        v = self.Wv(xv)
        if self.layer_id == 0:
            v0 = v
        else:
            v = v + (v0 - v) * torch.sigmoid(xv @ self.Wv1 @ self.Wv2 + self.v_bias)
        a = torch.sigmoid(xa @ self.Wa1 @ self.Wa2 + self.a_bias)
        g = torch.sigmoid(xg @ self.Wg1) @ self.Wg2

        kk = k * self.k_k
        k = k + k * (a - 1) * self.k_a

        r_, w_, k_, v_, a_ = [t.view(B, T, H, N) for t in (r, w_pre, k, v, a)]
        kk_ = F.normalize(kk.view(B, T, H, N), dim=-1, eps=1e-12)

        y = rwkv7_attn(w_, r_, k_, v_, kk_, a_, B, T, H, N)

        y = self.ln(y.reshape(B * T, C)).view(B, T, C)
        bonus = ((r_ * k_ * self.r_k).sum(dim=-1, keepdim=True) * v_).view(B, T, C)
        y = y + bonus
        return self.Wo(y * g), v0


class RWKV7ChannelMix(nn.Module):
    def __init__(self, dim, hidden_mult=4):
        super().__init__()
        self.x_k = nn.Parameter(torch.full((dim,), 0.5))
        self.Wk = nn.Linear(dim, dim * hidden_mult, bias=False)
        self.Wv = nn.Linear(dim * hidden_mult, dim, bias=False)
        nn.init.zeros_(self.Wv.weight)

    def forward(self, x):
        xx = F.pad(x, (0, 0, 1, -1)) - x
        k = self.Wk(x + xx * self.x_k)
        return self.Wv(torch.relu(k) ** 2)


class RWKV7Block(nn.Module):
    def __init__(self, dim, layer_id):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.att = RWKV7TimeMix(dim, layer_id)
        self.ffn = RWKV7ChannelMix(dim)

    def forward(self, x, v0):
        dx, v0 = self.att(self.ln1(x), v0)
        x = x + dx
        x = x + self.ffn(self.ln2(x))
        return x, v0


class RWKV7Classifier(nn.Module):
    def __init__(self, vocab_size, hidden_size=512, output_size=2, layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.ln0 = nn.LayerNorm(hidden_size)
        self.blocks = nn.ModuleList([RWKV7Block(hidden_size, i) for i in range(layers)])
        self.ln_out = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.ln0(self.embedding(x))
        v0 = None
        for block in self.blocks:
            x, v0 = block(x, v0)
        x = self.ln_out(x)
        return self.head(x.mean(dim=1))

# ============== TRAIN / EVAL ==============
def train_epoch(model, loader, opt, name):
    model.train()
    total = 0
    for batch in tqdm(loader, desc=f"{name} train"):
        input_ids = batch["input_ids"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
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
    results = {}

    print("\n" + "=" * 50)
    print("1) STANDARD GRU (2-layer, hidden=792, emb=512)")
    m = StandardGRU(vocab_size).to(DEVICE)
    print(f"   params: {count_params(m)/1e6:.2f}M")
    opt = optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
    for e in range(EPOCHS):
        loss = train_epoch(m, train_loader, opt, "GRU")
        acc = evaluate_model(m, test_loader, "GRU")
        results["Standard GRU"] = acc
        print(f"   Epoch {e+1} | Loss: {loss:.4f} | Acc: {acc:.2f}%")

    print("\n" + "=" * 50)
    print(f"2) RWKV-7 (2-layer, hidden=512, head=64, kernel={USE_KERNEL})")
    m = RWKV7Classifier(vocab_size, hidden_size=512, layers=2).to(DEVICE)
    print(f"   params: {count_params(m)/1e6:.2f}M")
    opt = optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
    for e in range(EPOCHS):
        loss = train_epoch(m, train_loader, opt, "RWKV7")
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
