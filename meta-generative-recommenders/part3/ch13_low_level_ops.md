# 13장. 저수준 연산 코드 분석

---

## 13.1 HSTU Compute (`ops/hstu_compute.py`)

### hstu_compute_uqvk

```python
def hstu_compute_uqvk(x, norm_weight, norm_bias, ...):
    normed_x = layer_norm(x, weight, bias, eps)           # Step 1: LayerNorm
    uvqk = addmm(uvqk_bias, normed_x, uvqk_weight)       # Step 2: Linear
    u, v, q, k = torch.split(uvqk, [H*n, H*n, A*n, A*n]) # Step 3: Split
    u = F.silu(u)                                          # Step 4: SiLU gating
    q = q.view(-1, num_heads, attn_dim)                    # Step 5: Reshape
    k = k.view(-1, num_heads, attn_dim)
    v = v.view(-1, num_heads, hidden_dim)
    return u, q, k, v
```

## 13.2 HSTU Attention (`ops/hstu_attention.py`)

### 커널 디스패치

```python
def hstu_mha(q, k, v, seq_offsets, ...):
    if kernel == HammerKernel.PYTORCH:
        return pytorch_hstu_mha(...)     # Reference (느림, 정확)
    elif kernel == HammerKernel.TRITON:
        return triton_hstu_mha(...)      # GPU 최적화 (~10x)
    elif kernel == HammerKernel.CUDA:
        return cuda_hstu_mha(...)        # FlashAttn V3 (~100x)
```

## 13.3 Triton 커널 개요

```python
# ops/triton/triton_hstu_attention.py (~3000 lines)
@triton.jit
def _hstu_attn_fwd_kernel(Q, K, V, Out, ...):
    # Block-level tiling for GPU efficiency
    pid = tl.program_id(0)      # GPU thread block ID
    q_block = tl.load(Q + ...)  # Load Q tile from HBM
    k_block = tl.load(K + ...)  # Load K tile from HBM
    # Compute attention in SRAM (fast)
    qk = tl.dot(q_block, tl.trans(k_block))
    qk = tl.where(causal_mask, qk, 0)
    attn = silu(qk) / n
    out = tl.dot(attn, v_block)
    tl.store(Out + ..., out)    # Store back to HBM
```

> **DE 관점**: Spark에서 `explain()`으로 physical plan을 보는 것처럼, Triton 커널은 논리적 연산(attention)의 물리적 실행 계획.
> 메모리 접근 패턴을 최적화하여 같은 수학을 10~100배 빠르게 실행.

---

## 13.4 성능 계층

| Level | 파일 | 용도 |
|-------|------|------|
| `ops/pytorch/` | Reference | 디버깅, 정확성 검증 |
| `ops/triton/` | GPU optimized | 일반 학습/추론 |
| `ops/cpp/` | CUDA C++ | H100 프로덕션 (FlashAttn V3) |

---

[← 12장](ch12_core_modules.md) | [목차](../../README.md) | [14장 →](ch14_research_code.md)
