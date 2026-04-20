#!/usr/bin/env python3
"""Part 1 전체를 Markdown으로 생성"""
import os

OUT = '/home1/irteam/work/hstu-study-guide/part1'
FIG = '../figures'
os.makedirs(OUT, exist_ok=True)

def write(name, content):
    path = os.path.join(OUT, name)
    with open(path, 'w') as f:
        f.write(content)
    print(f'  {path}')

# ============================================================
# Chapter 1
# ============================================================
write('ch01_linear_algebra.md', f'''# 1장. 선형대수 기초

> 추천 시스템의 언어 -- 벡터, 행렬, 텐서로 유저와 아이템을 표현하는 법

---

## 1.1 스칼라, 벡터, 행렬, 텐서

![Scalar to Tensor]({FIG}/ch01_scalar_vec_mat_tensor.png)

*[그림 1-1] 스칼라 → 벡터 → 행렬 → 텐서: 차원이 하나씩 늘어난다*

> **DE 관점 비유**
> - Scalar = DataFrame의 한 셀
> - Vector = DataFrame의 한 Row
> - Matrix = 하나의 DataFrame 테이블
> - Tensor = 여러 테이블을 쌓아놓은 것 (Parquet 파티션과 유사)

### 텐서의 shape

PyTorch에서 모든 데이터는 `torch.Tensor`로 표현됩니다.

| Shape | 의미 | 코드 위치 |
|-------|------|----------|
| `(B, N, D)` | Batch × 시퀀스길이 × 임베딩차원 | Padded tensor (SASRec) |
| `(sum_i N_i, D)` | 전체시퀀스합 × 임베딩차원 | **Jagged tensor (HSTU)** |
| `(B, H, N, N)` | Batch × Head × Q길이 × K길이 | Attention score matrix |
| `(V, D)` | 어휘크기 × 임베딩차원 | Embedding table |

---

## 1.2 내적(Dot Product)과 유사도

![Dot Product & Cosine]({FIG}/ch01_dot_product_cosine.png)

*[그림 1-2] 왼쪽: 내적 = 한 벡터를 다른 벡터에 투영한 길이 / 오른쪽: 코사인 유사도 = 방향의 유사함*

> **HSTU 코드에서의 활용**
> - 추천 시스템에서 **유저 벡터와 아이템 벡터의 내적 = 선호도 점수**
> - `L2NormPostprocessor`: 벡터를 단위원 위로 투영 → 내적 = 코사인 유사도
> - `DotProduct` similarity: `user_emb @ item_emb.T` → 추천 점수 계산

### 1.2.3 행렬 곱셈

![Matrix Multiplication]({FIG}/ch01_matmul.png)

*[그림 1-3] 행렬 곱셈: A(2×3) @ B(3×2) = C(2×2). 행과 열의 내적.*

```python
# PyTorch에서 행렬 곱셈
import torch

A = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float)
B = torch.tensor([[7, 8], [9, 10], [11, 12]], dtype=torch.float)

C = torch.mm(A, B)        # (2,3) @ (3,2) -> (2,2)
C = A @ B                 # 동일 (연산자 오버로딩)
C = torch.bmm(A_3d, B_3d) # 배치 행렬곱 (3D 텐서)
```

---

## 1.3 정규화(Normalization)

![Normalization]({FIG}/ch01_normalization.png)

*[그림 1-4] 왼쪽: L2 Norm (방향만 유지) / 오른쪽: Layer Norm (분포 표준화)*

| 정규화 | 수식 | HSTU 코드 |
|--------|------|----------|
| L2 Norm | `x / ‖x‖₂` | `L2NormPostprocessor` |
| Layer Norm | `(x - mean) / std` | `LayerNorm`, `_input_norm_weight` |
| Swish LN | `LN(x × σ(x))` | `SwishLayerNorm` (MLP 내부) |

---

## 1장 핵심 요약

![Summary]({FIG}/ch01_summary.png)

*[그림 1-5] 추천 시스템의 핵심 흐름: ID → 벡터 → 내적 → 점수*

> **3줄 요약**
> 1. 추천 시스템은 유저와 아이템을 **벡터**로 표현하고, **내적**으로 선호도를 계산한다
> 2. **행렬곱** = 여러 내적을 한 번에 계산 (GPU가 빠른 이유)
> 3. **정규화**(L2 Norm, Layer Norm)는 학습 안정성과 유사도 계산의 핵심

---

[← 목차](../README.md) | [2장 →](ch02_ml_basics.md)
''')

# ============================================================
# Chapter 2
# ============================================================
write('ch02_ml_basics.md', f'''# 2장. 머신러닝 기초

> 모델이 학습하는 원리 -- Loss, Gradient, Optimizer의 동작 방식

---

## 2.1 Supervised Learning Pipeline

![Pipeline]({FIG}/ch02_supervised_pipeline.png)

*[그림 2-1] Forward (predict) → Loss (measure error) → Backward (update)*

> **DE Pipeline 비유**
> - Input = HDFS raw logs
> - Model = Spark transformation (but learned, not coded)
> - Prediction = Output table
> - Loss = Data quality check ("얼마나 틀렸는가?")
> - Update = transformation 규칙을 자동으로 수정

---

## 2.2 Loss Functions

![Loss Functions]({FIG}/ch02_loss_functions.png)

*[그림 2-2] BCE: 클릭 예측 / MSE: 시청 시간 / Sampled Softmax: 네거티브 중 랭킹*

```python
# HSTU code: MultitaskModule loss calculation
# Classification task (click/no-click)
bce_loss = F.binary_cross_entropy_with_logits(preds, labels)

# Regression task (watch time)
mse_loss = F.mse_loss(preds, labels)

# Sampled Softmax (research/modeling/sequential/autoregressive_losses.py)
logits = torch.cat([pos_logit, neg_logits], dim=-1)  # [B, 1+N]
loss = -F.log_softmax(logits / temperature, dim=-1)[:, 0]  # maximize pos
```

---

## 2.3 Gradient Descent

![Gradient Descent]({FIG}/ch02_gradient_descent.png)

*[그림 2-3] 왼쪽: gradient를 따라 loss surface를 내려간다 / 오른쪽: learning rate 효과*

| Optimizer | Key Idea | HSTU Config |
|-----------|----------|-------------|
| SGD | 단순 gradient step | Not used |
| Adam | Adaptive LR + momentum | Common baseline |
| **AdamW** | Adam + decoupled weight decay | `weight_decay=1e-3` |
| LR Warmup | 시작 시 LR을 점진적으로 올림 | `num_warmup_steps` |

---

## 2.4 Backpropagation & Checkpointing

![Backprop]({FIG}/ch02_backprop.png)

*[그림 2-4] Forward pass: 출력 계산 / Backward pass: gradient 전파. RECOMPUTE = activation checkpointing*

```python
# STU Layer config (modules/stu.py:STULayerConfig)
recompute_normed_x: bool = True   # LayerNorm 출력을 재계산
recompute_uvqk: bool = True       # U,V,Q,K projection을 재계산
recompute_y: bool = True          # Attention 출력을 재계산

# Why? 저장 대신 재계산 → GPU 메모리 ~40% 절약
```

---

## 2.5 Overfitting & Regularization

![Overfitting & Dropout]({FIG}/ch02_overfitting_dropout.png)

*[그림 2-5] 왼쪽: overfitting = val loss 상승 / 오른쪽: dropout으로 뉴런을 랜덤 비활성화*

| Technique | Effect | HSTU Config |
|-----------|--------|-------------|
| Dropout | 랜덤 뉴런 비활성화 | `output_dropout_ratio=0.3`, `input_dropout=0.2` |
| Weight Decay | 큰 가중치 페널티 | `weight_decay=1e-3` |
| Early Stopping | val 성능 하락 시 중단 | HR@10을 epoch마다 모니터링 |

---

## 2장 핵심 요약

> **Core Training Loop**
> 1. **Forward**: input → model → prediction
> 2. **Loss**: prediction vs label 비교 (BCE for clicks, MSE for watch time)
> 3. **Backward**: chain rule로 gradient 계산
> 4. **Update**: AdamW로 가중치 조정 (`lr=1e-3`, `weight_decay=1e-3`)
> 5. **Repeat** for `num_epochs` (HSTU config: 101)

---

[← 1장](ch01_linear_algebra.md) | [목차](../README.md) | [3장 →](ch03_deep_learning.md)
''')

# ============================================================
# Chapter 3
# ============================================================
write('ch03_deep_learning.md', f'''# 3장. 딥러닝 기초

> Neural Networks, Activation Functions, PyTorch Fundamentals

---

## 3.1 Activation Functions

![Activations]({FIG}/ch03_activations.png)

*[그림 3-1] HSTU는 SiLU (Swish) 사용: smooth, 미분 가능, 음수값도 일부 통과*

> **Why SiLU in HSTU?**
> - `SiLU = x × sigmoid(x)`: smooth gating으로 정보 흐름을 제어
> - ReLU는 모든 음수를 죽임 → 정보 손실
> - STU Layer에서: `u = F.silu(u)` → valve처럼 정보 흐름을 조절

---

## 3.2 MLP Architecture

![MLP]({FIG}/ch03_mlp.png)

*[그림 3-2] HSTU Preprocessor의 MLP: Linear → SwishLayerNorm → Linear → LayerNorm*

```python
# HSTU Preprocessor MLP (modules/preprocessors.py)
_content_embedding_mlp = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),     # 512 → 256
    SwishLayerNorm(hidden_dim),            # SiLU + LayerNorm
    nn.Linear(hidden_dim, output_dim),     # 256 → 512
    nn.LayerNorm(output_dim),              # normalize
)
```

---

## 3.3 PyTorch nn.Module

![Module Hierarchy]({FIG}/ch03_module_hierarchy.png)

*[그림 3-3] HSTU 코드베이스의 PyTorch Module 계층 구조*

```python
# HSTU의 핵심 패턴
class STULayer(HammerModule):
    def __init__(self, config: STULayerConfig):
        super().__init__()
        # nn.Parameter = 학습 가능한 가중치
        self._uvqk_weight = nn.Parameter(
            torch.empty(D, (H*2 + A*2) * num_heads)
        )

    def forward(self, x, x_lengths, x_offsets, ...):
        # model(input) 호출 시 자동 실행
        u, attn, k, v = hstu_preprocess_and_attention(...)
        return hstu_compute_output(attn, u, x, ...)
```

---

## 3.4 Training vs Inference Mode

![Train vs Eval]({FIG}/ch03_train_vs_eval.png)

*[그림 3-4] Training과 Inference의 핵심 차이점*

---

## 3.5 Distributed Data Parallel (DDP)

![DDP]({FIG}/ch03_ddp.png)

*[그림 3-5] DDP: 각 GPU가 다른 배치를 처리하고, gradient를 평균*

```python
# HSTU DDP setup (main.py)
num_gpus = torch.cuda.device_count()  # e.g., 4
mp.spawn(train_fn, args=(num_gpus, master_port),
         nprocs=num_gpus)  # GPU당 1 process

# Inside train_fn:
dist.init_process_group("nccl", rank=rank, world_size=world_size)
model = DDP(model, device_ids=[rank])
```

---

## 3장 핵심 요약

> 1. Neural Network = Linear + Activation 레이어의 스택
> 2. HSTU는 **SiLU gating** 사용 (표준 softmax 대신) → 정보 흐름 제어
> 3. PyTorch: `nn.Module` + `nn.Parameter` + `forward()` = 모든 모델의 기본 블록

---

[← 2장](ch02_ml_basics.md) | [목차](../README.md) | [4장 →](ch04_embedding.md)
''')

# ============================================================
# Chapter 4
# ============================================================
write('ch04_embedding.md', f'''# 4장. Embedding

> Discrete IDs to Dense Vectors -- 추천 시스템의 기반

---

## 4.1 One-Hot vs Embedding

![One-Hot vs Embedding]({FIG}/ch04_onehot_vs_embedding.png)

*[그림 4-1] One-Hot: sparse, huge / Embedding: dense, compact. Lookup = table[item_id]*

---

## 4.2 Embedding Lookup

![Embedding Lookup]({FIG}/ch04_embedding_lookup.png)

*[그림 4-2] Embedding Lookup: ID가 들어가면 dense vector가 나온다. 연산 없이 테이블 조회.*

```python
# HSTU embedding (research/modeling/sequential/embedding_modules.py)
class LocalEmbeddingModule(EmbeddingModule):
    def __init__(self, num_items, item_embedding_dim):
        self._item_emb = nn.Embedding(
            num_items + 1,
            item_embedding_dim,
            padding_idx=0  # ID=0 means "no item"
        )

    def get_item_embeddings(self, item_ids):
        return self._item_emb(item_ids)  # [B, N] → [B, N, D]
```

---

## 4.3 Learned Embedding Space

![Embedding Space]({FIG}/ch04_embedding_space.png)

*[그림 4-3] 학습 후 임베딩이 자동으로 조직화된다. 비슷한 아이템 = 가까운 벡터.*

---

## 4.4 Large-Scale: Hash Embedding

![Hash Embedding]({FIG}/ch04_hash_embedding.png)

*[그림 4-4] Hash Embedding: 100M 아이템을 10M 버킷에 매핑. 일부 충돌이 있지만 10배 메모리 절약.*

| Config | Embedding Dim | Hash Size |
|--------|---------------|-----------|
| Research (ML-1M) | 50 | ~4K items (no hash) |
| Research (ML-20M) | 50 | ~27K items (no hash) |
| DLRMv3 (small) | 512 | `HASH_SIZE = 10M` |
| DLRMv3 (large) | 512 | `HASH_SIZE_1B = 1B` |

```python
# DLRMv3 Embedding Config (dlrm_v3/configs.py)
EmbeddingConfig(
    name="item_id",
    embedding_dim=HSTU_EMBEDDING_DIM,  # 512
    num_embeddings=HASH_SIZE,           # 10_000_000
    data_type=DataType.FP16,            # half precision → 메모리 절반
)
```

---

## 4장 핵심 요약

> 1. **Embedding** = ID를 dense vector로 매핑하는 lookup table
> 2. 학습이 진행되면 비슷한 아이템이 **가까운 벡터**로 모인다
> 3. **Hash trick**: 100M 아이템 → 10M 버킷 매핑 (충돌 OK)
> 4. HSTU는 `TorchRec EmbeddingCollection`으로 분산 임베딩 처리

---

[← 3장](ch03_deep_learning.md) | [목차](../README.md) | [5장 →](ch05_attention.md)
''')

# ============================================================
# Chapter 5
# ============================================================
write('ch05_attention.md', f'''# 5장. Attention Mechanism

> Self-Attention에서 Transformer까지 -- HSTU를 이해하기 위한 핵심

---

## 5.1 Self-Attention: QKV

![Self-Attention]({FIG}/ch05_self_attention.png)

*[그림 5-1] Self-Attention: 각 토큰이 다른 모든 토큰에 "질의"하여 관련 정보를 가져온다*

> **SQL 비유**
> - **Q (Query)** = WHERE 조건절
> - **K (Key)** = INDEX (각 행의 매칭 키)
> - **V (Value)** = SELECT 컬럼
> - **Attention** = soft JOIN: 모든 매칭 행의 weighted sum을 반환

---

## 5.2 Causal Masking

![Causal Mask]({FIG}/ch05_causal_mask.png)

*[그림 5-2] Causal masking: t3는 t1,t2,t3만 참조 가능. t4,t5는 볼 수 없다.*

> **Why Causal?**
> - 추천 = 다음 클릭 예측
> - User viewed: shoes → sports → watch → **???**
> - **???** 을 예측할 때 미래 정보를 참조하면 안 됨
> - Causal mask가 이를 강제: `causal=True` (STULayerConfig)

---

## 5.3 Multi-Head Attention

![Multi-Head]({FIG}/ch05_multihead.png)

*[그림 5-3] Multi-Head: D를 H개 head로 분할, 각각 독립 attention, 결과를 concat*

| Config | num_heads | D_total | D_per_head |
|--------|-----------|---------|------------|
| Research (small) | 2 | 50 | 25 |
| Research (large) | 2 | 50 | 25 |
| DLRMv3 | 4 | 512 | 128 |

---

## 5.4 Transformer Block vs STU Layer

![Transformer vs STU]({FIG}/ch05_transformer_vs_stu.png)

*[그림 5-4] 핵심 차이: HSTU는 FFN을 SiLU gating (u × attn)으로 대체. 파라미터↓, 표현력 유지.*

> **STU Layer vs Transformer: 3가지 핵심 차이**
> 1. **UVQK** (4개 projection) vs QKV (3개) → U가 gating 역할
> 2. **SiLU(U) × Attention**이 FFN을 대체 → 2-layer MLP 대신 gating
> 3. **상대적 시간+위치 인코딩** vs 절대 위치 인코딩

---

## 5.5 Positional Encoding

![Positional Encoding]({FIG}/ch05_positional_encoding.png)

*[그림 5-5] 왼쪽: 표준 sinusoidal (고정 패턴) / 오른쪽: HSTU는 log-bucketed 시간 차이 사용*

```python
# HSTU time bucketization (research/modeling/sequential/hstu.py)
bucketization_fn = lambda x: (
    torch.log(torch.abs(x).clamp(min=1)) / 0.301
).long()

# 0.301 = log(2), 즉 bucket = log2(time_diff)
# 10초 → bucket 3, 1시간 → bucket 12, 1주 → bucket 20
```

---

## 5장 핵심 요약

> 1. **Attention** = soft lookup: Q가 질문, K가 매칭, V가 정보 반환
> 2. **Causal masking**: 과거만 참조 (next-item prediction에 필수)
> 3. **Multi-head**: 여러 "관점"에서 동시에 attention
> 4. **HSTU STU Layer**: FFN 대신 SiLU gating (더 효율적)
> 5. **시간 인코딩**: log-bucketed 시간 차이로 "최신성"과 "주기성" 포착

---

[← 4장](ch04_embedding.md) | [목차](../README.md) | [6장 →](ch06_recsys.md)
''')

# ============================================================
# Chapter 6
# ============================================================
write('ch06_recsys.md', f'''# 6장. 추천 시스템 기초

> Collaborative Filtering에서 Sequential Recommendation까지 -- 진화의 역사

---

## 6.1 Collaborative Filtering & Matrix Factorization

![CF & MF]({FIG}/ch06_cf_mf.png)

*[그림 6-1] 왼쪽: sparse 평점 행렬 / 오른쪽: user & item 임베딩으로 분해. 이것이 임베딩의 기원!*

---

## 6.2 Sequential Recommendation

![Sequential]({FIG}/ch06_sequential.png)

*[그림 6-2] 순서와 시간이 중요하다. HSTU는 short-term (최근 클릭)과 long-term (선호) 패턴을 동시에 포착.*

> **Place Service 매핑**
> - User sequence: search → click → visit → save → click → **???**
> - HSTU `ActionEncoder`: 행동 TYPE을 인코딩 (search=1, click=2, visit=4, save=8)
> - HSTU `TimestampEncoder`: 각 행동의 시간 차이를 포착
> - `Target-Aware Attention`: 후보 장소 정보를 반영하여 이력을 인코딩

---

## 6.3 Evaluation Metrics

![Metrics]({FIG}/ch06_metrics.png)

*[그림 6-3] HR@K: 정답이 top K에 있나? (binary) / NDCG@K: 정답이 얼마나 높은 순위인가? (위치 가중)*

| Metric | 의미 | HSTU benchmark |
|--------|------|---------------|
| **HR@10** | 정답이 top-10에 포함? | HSTU-large: **+56.7%** vs SASRec |
| **NDCG@10** | 위치 가중 추천 품질 | HSTU-large: **+60.7%** vs SASRec |
| MRR | 첫 정답의 역순위 평균 | eval loop에서 리포트 |
| GAUC | 유저별 Group AUC | DLRMv3 프로덕션 메트릭 |

---

## 6.4 HSTU Benchmark Results

![Benchmark]({FIG}/ch06_benchmark.png)

*[그림 6-4] HSTU-large가 모든 데이터셋에서 SASRec을 큰 폭으로 앞선다.*

---

## 6.5 Evolution of RecSys Models

![Evolution]({FIG}/ch06_evolution.png)

*[그림 6-5] MF (2009) → Deep RecSys (2016) → SASRec (2018) → HSTU (2024) → DLRM-v3 (production)*

---

## 6장 핵심 요약 & Part 1 마무리

> **Part 1에서 배운 것 총정리**
> 1. **CF + MF** = 임베딩의 기원 (user/item 벡터 + 내적)
> 2. **Sequential 모델**은 유저 행동의 순서를 포착
> 3. **SASRec** = Transformer를 RecSys에 적용 (baseline)
> 4. **HSTU** = SiLU gating + 시간 인코딩 + jagged tensors (SOTA)
> 5. **평가**: HR@K (맞았나?), NDCG@K (순위가 높은가?)
>
> **이제 HSTU 코드를 이해할 모든 기초 지식을 갖추었습니다!**
> Part 2에서 아키텍처를 심층 분석합니다.

---

[← 5장](ch05_attention.md) | [목차](../README.md)
''')

print('All chapters generated!')
