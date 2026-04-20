#!/usr/bin/env python3
"""Part 3 (ch10-15) + Part 4 (ch16-18) figures + markdown"""
import numpy as np, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BASE = '/home1/irteam/work/hstu-study-guide/meta-generative-recommenders'
FIG = f'{BASE}/figures'
os.makedirs(f'{BASE}/part3', exist_ok=True)
os.makedirs(f'{BASE}/part4', exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','axes.unicode_minus':False,'figure.dpi':150})

def sf(fig,n):
    p=f'{FIG}/{n}.png'; fig.savefig(p,bbox_inches='tight',facecolor='white',edgecolor='none'); plt.close(fig); return p

# ============ FIGURES ============

# Ch10: Repo structure
fig,ax=plt.subplots(figsize=(12,7)); ax.axis('off')
tree = [
    (0.5, 6.5, 'generative_recommenders/', '#333', 14, True),
    (1.5, 5.8, 'research/', '#1565C0', 11, True),
    (2.5, 5.3, 'data/ (dataset, eval, features)', '#42A5F5', 9, False),
    (2.5, 4.8, 'modeling/sequential/ (hstu.py, sasrec.py, losses)', '#42A5F5', 9, False),
    (2.5, 4.3, 'trainer/ (train.py, data_loader.py)', '#42A5F5', 9, False),
    (2.5, 3.8, 'rails/ (indexing, similarities)', '#42A5F5', 9, False),
    (1.5, 3.1, 'modules/', '#2E7D32', 11, True),
    (2.5, 2.6, 'stu.py, hstu_transducer.py, dlrm_hstu.py', '#66BB6A', 9, False),
    (2.5, 2.1, 'preprocessors.py, postprocessors.py, action_encoder.py', '#66BB6A', 9, False),
    (1.5, 1.4, 'ops/', '#E65100', 11, True),
    (2.5, 0.9, 'pytorch/ (reference), triton/ (GPU), cpp/ (CUDA)', '#FFA726', 9, False),
    (2.5, 0.4, 'hstu_attention.py, hstu_compute.py, jagged_tensors.py', '#FFA726', 9, False),
    (1.5, -0.3, 'dlrm_v3/', '#6A1B9A', 11, True),
    (2.5, -0.8, 'train/, inference/, datasets/, configs.py', '#AB47BC', 9, False),
]
for x,y,t,c,s,b in tree:
    ax.text(x,y,t,fontsize=s,color=c,fontweight='bold' if b else 'normal',family='monospace')
# Labels
labels = [
    (7, 5.5, 'Paper reproduction\n(padded tensors, gin config)', '#1565C0'),
    (7, 2.3, 'Optimized production\n(jagged tensors, HammerKernel)', '#2E7D32'),
    (7, 0.6, 'GPU kernels\n(Triton, CUDA, FlashAttn V3)', '#E65100'),
    (7, -0.5, 'End-to-end production\n(TorchRec, MLPerf, DDP)', '#6A1B9A'),
]
for x,y,t,c in labels:
    ax.text(x,y,t,fontsize=10,color=c,ha='center',
            bbox=dict(boxstyle='round',facecolor='white',edgecolor=c,lw=1.5))
    ax.annotate('',xy=(5,y+0.1),xytext=(x-1.5,y+0.1),
                arrowprops=dict(arrowstyle='->',lw=1.5,color=c,linestyle='--'))
ax.set_xlim(-0.2,10); ax.set_ylim(-1.5,7.2)
sf(fig,'ch10_repo_structure')

# Ch11: Data pipeline flow
fig,ax=plt.subplots(figsize=(13,4)); ax.axis('off')
steps=[
    (1,'CSV\nratings.csv','#E3F2FD','#1565C0'),
    (3.2,'preprocess\nsort by time\nsplit train/eval','#E8F5E9','#2E7D32'),
    (5.8,'DatasetV2\nper-user\nsequences','#FFF3E0','#E65100'),
    (8.2,'DataLoader\nbatch + shuffle\nprefetch=128','#F3E5F5','#6A1B9A'),
    (10.8,'seq_features\npast_ids\npast_lengths\npast_payloads','#FFEBEE','#C62828'),
]
for x,t,fc,ec in steps:
    rect=patches.FancyBboxPatch((x-0.9,0.5),1.8,2,boxstyle="round,pad=0.1",facecolor=fc,edgecolor=ec,lw=2)
    ax.add_patch(rect); ax.text(x,1.5,t,ha='center',va='center',fontsize=9,fontweight='bold',color=ec)
for i in range(len(steps)-1):
    ax.annotate('',xy=(steps[i+1][0]-1,1.5),xytext=(steps[i][0]+1,1.5),
                arrowprops=dict(arrowstyle='->',lw=2,color='#666'))
ax.set_xlim(-0.5,12.5); ax.set_ylim(-0.2,3)
sf(fig,'ch11_data_pipeline')

# Ch12: Module composition
fig,ax=plt.subplots(figsize=(12,6)); ax.axis('off')
mods=[
    (6,5.5,'DlrmHSTU (top-level model)','#E3F2FD','#1565C0',8,0.6),
    (3,4.2,'EmbeddingCollection\n(TorchRec)','#E8F5E9','#2E7D32',3,0.7),
    (7,4.2,'HSTUTransducer','#FFF3E0','#E65100',3,0.7),
    (11,4.2,'MultitaskModule\n(click+watchtime)','#F3E5F5','#6A1B9A',3,0.7),
    (4.5,2.8,'ContextualPreprocessor','#FFEBEE','#C62828',3,0.6),
    (8,2.8,'STUStack\n[STULayer x N]','#FFEBEE','#C62828',2.5,0.6),
    (11,2.8,'PostProcessor\n(L2Norm/LN)','#FFEBEE','#C62828',2.5,0.6),
    (4.5,1.5,'ContentEncoder\n+ ActionEncoder','#E0F7FA','#00695C',3,0.6),
    (8,1.5,'PositionalEncoder\n(time+position)','#E0F7FA','#00695C',2.5,0.6),
]
for x,y,t,fc,ec,w,h in mods:
    rect=patches.FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle="round,pad=0.08",facecolor=fc,edgecolor=ec,lw=2)
    ax.add_patch(rect); ax.text(x,y,t,ha='center',va='center',fontsize=8,fontweight='bold',color=ec)
conns=[(0,1),(0,2),(0,3),(2,4),(2,5),(2,6),(4,7),(2,8)]
for a,b in conns:
    ax.annotate('',xy=(mods[b][0],mods[b][1]+mods[b][6]/2),
                xytext=(mods[a][0],mods[a][1]-mods[a][6]/2),
                arrowprops=dict(arrowstyle='->',lw=1.5,color='#888'))
ax.set_xlim(0,13); ax.set_ylim(0.8,6.2)
sf(fig,'ch12_module_composition')

# Ch14: Training loop
fig,ax=plt.subplots(figsize=(12,5)); ax.axis('off')
loop=[
    (1,2.5,'Load batch\nseq_features','#E3F2FD','#1565C0'),
    (3.5,2.5,'Embed items\nget_item_embeddings','#E8F5E9','#2E7D32'),
    (6,2.5,'Encode\nmodel(features)','#FFF3E0','#E65100'),
    (8.5,2.5,'Loss\nar_loss(logits)','#FFEBEE','#C62828'),
    (10.5,2.5,'Backward\nloss.backward()','#F3E5F5','#6A1B9A'),
    (10.5,0.8,'Update\nopt.step()','#E0F7FA','#00695C'),
    (6,0.8,'Eval\nHR@K, NDCG@K','#FFFDE7','#F57F17'),
    (1,0.8,'TensorBoard\nlog metrics','#F5F5F5','#555'),
]
for x,y,t,fc,ec in loop:
    rect=patches.FancyBboxPatch((x-1,y-0.5),2,1,boxstyle="round,pad=0.1",facecolor=fc,edgecolor=ec,lw=2)
    ax.add_patch(rect); ax.text(x,y,t,ha='center',va='center',fontsize=9,fontweight='bold',color=ec)
arrows=[(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7)]
for a,b in arrows:
    ax.annotate('',xy=(loop[b][0]-1 if loop[b][0]>loop[a][0] else loop[b][0]+1,loop[b][1]),
                xytext=(loop[a][0]+1 if loop[a][0]<loop[b][0] else loop[a][0]-1,loop[a][1]),
                arrowprops=dict(arrowstyle='->',lw=2,color='#666'))
ax.annotate('',xy=(1,1.3),xytext=(1,2),arrowprops=dict(arrowstyle='->',lw=2,color='#666'))
ax.text(6,3.8,'Training Loop (research/trainer/train.py)',ha='center',fontsize=12,fontweight='bold')
ax.set_xlim(-0.8,12.5); ax.set_ylim(-0.2,4.3)
sf(fig,'ch14_training_loop')

# Ch15: DLRMv3 inference
fig,ax=plt.subplots(figsize=(12,4.5)); ax.axis('off')
inf=[
    (1.5,2.5,'Query\nRequest','#E3F2FD','#1565C0'),
    (4,2.5,'DataProducer\n(multi-thread)','#E8F5E9','#2E7D32'),
    (6.5,3.5,'CPU: Sparse\nEmbedding Lookup','#FFF3E0','#E65100'),
    (6.5,1.5,'GPU: HSTU\nTransducer','#F3E5F5','#6A1B9A'),
    (9,2.5,'MultitaskModule\nscoring','#FFEBEE','#C62828'),
    (11.5,2.5,'Top-K\nranked items','#E0F7FA','#00695C'),
]
for x,y,t,fc,ec in inf:
    rect=patches.FancyBboxPatch((x-1.1,y-0.5),2.2,1,boxstyle="round,pad=0.1",facecolor=fc,edgecolor=ec,lw=2)
    ax.add_patch(rect); ax.text(x,y,t,ha='center',va='center',fontsize=9,fontweight='bold',color=ec)
for a,b in [(0,1),(1,2),(2,3),(3,4),(4,5)]:
    ax.annotate('',xy=(inf[b][0]-1.1,inf[b][1]),xytext=(inf[a][0]+1.1,inf[a][1]),
                arrowprops=dict(arrowstyle='->',lw=2,color='#666'))
ax.text(6.5,0.3,'KV Cache: reuse prev computations for speed',ha='center',fontsize=10,
        color='#6A1B9A',style='italic',bbox=dict(boxstyle='round',facecolor='#F3E5F5',alpha=0.5))
ax.set_xlim(-0.5,13); ax.set_ylim(-0.3,4.5)
sf(fig,'ch15_inference_pipeline')

# Ch17: Hyperparameters radar
fig,ax=plt.subplots(figsize=(10,5))
params=['embed_dim','num_layers','num_heads','seq_length','batch_size','negatives']
research=[50,8,2,200,128,128]
prod=[512,5,4,16384,256,512]
x=np.arange(len(params)); w=0.35
ax.bar(x-w/2,research,w,color='#42A5F5',label='Research (ML-1M)',edgecolor='white')
ax.bar(x+w/2,prod,w,color='#E65100',label='DLRMv3 (Production)',edgecolor='white')
ax.set_xticks(x); ax.set_xticklabels(params,fontsize=9)
ax.set_ylabel('Value',fontsize=11); ax.set_yscale('log')
ax.set_title('Research vs Production Hyperparameters',fontsize=12,fontweight='bold')
ax.legend(fontsize=10); ax.grid(True,alpha=0.3,axis='y')
sf(fig,'ch17_hyperparams')

# Ch18: Service mapping
fig,ax=plt.subplots(figsize=(13,5)); ax.axis('off')
maps=[
    (2,4,'HSTU Component','#F5F5F5','#333'),
    (6.5,4,'Place Service','#F5F5F5','#333'),
    (10.5,4,'Content Feed','#F5F5F5','#333'),
    (2,3,'item_embedding','#E3F2FD','#1565C0'),
    (6.5,3,'Place ID + Category','#E8F5E9','#2E7D32'),
    (10.5,3,'Content ID + Type','#E8F5E9','#2E7D32'),
    (2,2,'UIH sequence','#E3F2FD','#1565C0'),
    (6.5,2,'visit/search/save history','#FFF3E0','#E65100'),
    (10.5,2,'view/click/share history','#FFF3E0','#E65100'),
    (2,1,'ActionEncoder','#E3F2FD','#1565C0'),
    (6.5,1,'search=1,click=2\nvisit=4,save=8','#FFEBEE','#C62828'),
    (10.5,1,'view=1,click=2\nlike=4,share=8','#FFEBEE','#C62828'),
    (2,0,'TimestampEncoder','#E3F2FD','#1565C0'),
    (6.5,0,'visit time patterns\n(weekday/weekend)','#F3E5F5','#6A1B9A'),
    (10.5,0,'consumption time\n(morning/evening)','#F3E5F5','#6A1B9A'),
]
for x,y,t,fc,ec in maps:
    w=3.2 if x>3 else 2.5
    rect=patches.FancyBboxPatch((x-w/2,y-0.35),w,0.7,boxstyle="round,pad=0.06",
         facecolor=fc,edgecolor=ec,lw=1.5 if y==4 else 1)
    ax.add_patch(rect)
    ax.text(x,y,t,ha='center',va='center',fontsize=8 if y<4 else 9,
            fontweight='bold' if y==4 else 'normal',color=ec)
for i in range(3,15,3):
    ax.annotate('',xy=(maps[i+1][0]-1.5,maps[i+1][1]),xytext=(maps[i][0]+1.3,maps[i][1]),
                arrowprops=dict(arrowstyle='->',lw=1.5,color='#999'))
    ax.annotate('',xy=(maps[i+2][0]-1.5,maps[i+2][1]),xytext=(maps[i][0]+1.3,maps[i][1]),
                arrowprops=dict(arrowstyle='->',lw=1.5,color='#999'))
ax.set_xlim(-0.3,12.5); ax.set_ylim(-0.7,4.7)
sf(fig,'ch18_service_mapping')

print("All Part 3+4 figures done!")

# ============ MARKDOWN ============
F='../figures'

def w(path,content):
    with open(path,'w') as f: f.write(content)
    print(f'  {path}')

# --- Ch10 ---
w(f'{BASE}/part3/ch10_repo_structure.md', f'''# 10장. 레포지토리 구조와 코드 컨벤션

---

## 10.1 디렉토리 구조

![Repo Structure]({F}/ch10_repo_structure.png)

*[그림 10-1] 4개 레벨: research (논문재현) → modules (최적화) → ops (GPU커널) → dlrm_v3 (프로덕션)*

| 디렉토리 | 역할 | Tensor 타입 | 대상 |
|----------|------|------------|------|
| `research/` | 논문 재현, 실험 | Padded | 연구자 |
| `modules/` | 최적화된 모델 | **Jagged** | 엔지니어 |
| `ops/` | GPU 커널 (Triton/CUDA) | Jagged | GPU 전문가 |
| `dlrm_v3/` | E2E 프로덕션 | Jagged + TorchRec | 프로덕션 |

---

## 10.2 주요 의존성

```
torch>=2.6.0          # PyTorch (핵심)
fbgemm_gpu>=1.1.0     # Meta GPU 최적화 (jagged tensor ops)
torchrec>=1.1.0       # Meta 추천 라이브러리 (분산 임베딩)
gin_config>=0.5.0     # Google 설정 프레임워크
```

> **DE 관점**: Spark의 Catalyst optimizer처럼, HSTU는 같은 로직을 여러 백엔드(PyTorch/Triton/CUDA)로 실행 가능.
> `HammerKernel` enum이 런타임에 최적 커널을 선택.

---

## 10.3 코드 컨벤션

```python
# 1. HammerModule: 모든 모듈의 기반 클래스
class STULayer(HammerModule):  # not nn.Module

# 2. @torch.fx.wrap: 모델 트레이싱/컴파일 지원
@torch.fx.wrap
def _update_kv_cache(...):

# 3. record_function: 프로파일링 마커
with record_function("hstu_attention"):

# 4. Gin configuration: 하이퍼파라미터 주입
@gin.configurable
def train_fn(rank, world_size, master_port, dataset_name="ml-1m", ...):
```

---

[← 9장](../part2/ch09_jagged_tensor.md) | [목차](../../README.md) | [11장 →](ch11_data_pipeline.md)
''')

# --- Ch11 ---
w(f'{BASE}/part3/ch11_data_pipeline.md', f'''# 11장. 데이터 파이프라인

---

## 11.1 전체 흐름

![Data Pipeline]({F}/ch11_data_pipeline.png)

*[그림 11-1] CSV → 전처리 → Dataset → DataLoader → Features*

---

## 11.2 데이터 전처리

```bash
# 1단계: 데이터 다운로드 + 변환
mkdir -p tmp/
python3 preprocess_public_data.py

# 결과:
# tmp/ml-1m/ratings.csv     (userId, movieId, rating, timestamp)
# tmp/ml-20m/ratings.csv
# tmp/amzn-books/ratings.csv
```

## 11.3 Dataset 클래스

```python
# research/data/dataset.py
class DatasetV2:
    def __init__(self, ratings_file, item_features_file=None):
        self._data = pd.read_csv(ratings_file)
        # 유저별로 그룹화하여 시퀀스 구성 (시간순)

# research/data/reco_dataset.py
train_dataset = RecoDataset(data, ignore_last_n=1)  # 마지막=target
eval_dataset  = RecoDataset(data, ignore_last_n=0)  # 전체 사용
```

## 11.4 Feature Engineering

```python
# research/modeling/sequential/features.py
def movielens_seq_features_from_row(row, max_length):
    return SequentialFeatures(
        past_ids=row["movie_ids"][-max_length:],    # 아이템 ID 시퀀스
        past_lengths=len(past_ids),                  # 시퀀스 길이
        past_payloads={{
            "timestamps": row["timestamps"],          # 행동 시간
            "ratings": row["ratings"],                # 평점
        }},
    )
```

> **Spark ETL 비유**
> - `preprocess_public_data.py` = raw data ingestion job
> - `DatasetV2` = Spark DataFrame partitioned by user_id
> - `movielens_seq_features_from_row` = UDF that transforms each row
> - `DataLoader(prefetch_factor=128)` = Spark의 broadcast + coalesce

---

[← 10장](ch10_repo_structure.md) | [목차](../../README.md) | [12장 →](ch12_core_modules.md)
''')

# --- Ch12 ---
w(f'{BASE}/part3/ch12_core_modules.md', f'''# 12장. 핵심 모듈 코드 분석

---

## 12.1 모듈 구성도

![Module Composition]({F}/ch12_module_composition.png)

*[그림 12-1] DlrmHSTU가 최상위. 하위에 Embedding, Transducer, Multitask이 조합됨.*

---

## 12.2 STULayer (`modules/stu.py`)

### 가중치 Shape 정리

| Parameter | Shape | 역할 |
|-----------|-------|------|
| `_uvqk_weight` | `(D, (H*2+A*2)*heads)` | UVQK 한번에 계산 |
| `_uvqk_beta` | `((H*2+A*2)*heads,)` | UVQK bias |
| `_input_norm_weight/bias` | `(D,)` | LayerNorm |
| `_output_weight` | `(H*heads*3, D)` | concat(u,attn,u*attn) → D |
| `_output_norm_weight/bias` | `(H*heads,)` | Output LayerNorm |

### forward 핵심 코드

```python
def forward(self, x, x_lengths, x_offsets, max_seq_len, num_targets, ...):
    # Step 1: UVQK + Attention (fused)
    u, attn, k, v = hstu_preprocess_and_attention(
        x, self._input_norm_weight, self._input_norm_bias,
        self._uvqk_weight, self._uvqk_beta, ...)

    # Step 2: KV Cache update (for inference)
    self.k_cache, self.v_cache, ... = _update_kv_cache(...)

    # Step 3: Output = LN(attn) * u → concat → Linear + residual
    return hstu_compute_output(
        attn, u, x, self._output_norm_weight, self._output_norm_bias,
        self._output_weight, ...)
```

---

## 12.3 HSTUTransducer (`modules/hstu_transducer.py`)

```
forward(x, ...) =
  1. _preprocess: InputPreprocessor → PositionalEncoder → Dropout
  2. _hstu_compute: STUStack.forward(x) → N layers of STULayer
  3. _postprocess: split_2D_jagged → OutputPostprocessor
  → returns (candidate_embeddings, full_embeddings)
```

---

## 12.4 DlrmHSTU (`modules/dlrm_hstu.py`)

```
main_forward(x, ...) =
  1. preprocess: EmbeddingCollection → KeyedJaggedTensor lookup
  2. _user_forward: concat UIH+candidates → HSTUTransducer → user_emb
  3. _item_forward: concat item features → item_embedding_mlp → item_emb
  4. MultitaskModule: user_emb * item_emb → predictions + losses
```

---

[← 11장](ch11_data_pipeline.md) | [목차](../../README.md) | [13장 →](ch13_low_level_ops.md)
''')

# --- Ch13 ---
w(f'{BASE}/part3/ch13_low_level_ops.md', f'''# 13장. 저수준 연산 코드 분석

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
''')

# --- Ch14 ---
w(f'{BASE}/part3/ch14_research_code.md', f'''# 14장. Research 코드 워크스루

---

## 14.1 Training Loop

![Training Loop]({F}/ch14_training_loop.png)

*[그림 14-1] Research 학습 루프: batch → embed → encode → loss → backward → update → eval*

### train_fn 전체 구조

```python
@gin.configurable
def train_fn(rank, world_size, master_port,
             dataset_name="ml-1m", max_sequence_length=200,
             local_batch_size=128, main_module="HSTU",
             learning_rate=1e-3, num_epochs=101, ...):

    # 1. DDP 초기화
    setup(rank, world_size, master_port)

    # 2. 데이터 로드
    dataset, eval_dataset = get_reco_dataset(dataset_name)
    train_loader = create_data_loader(dataset, batch_size, ...)

    # 3. 모델 생성
    model = get_sequential_encoder(main_module, ...)  # HSTU or SASRec
    model = DDP(model, device_ids=[rank])

    # 4. Loss + Sampler
    ar_loss = SampledSoftmaxLoss(num_negatives, temperature, ...)
    sampler = LocalNegativesSampler(num_items, item_emb, ...)

    # 5. Training loop
    for epoch in range(num_epochs):
        for batch in train_loader:
            features = movielens_seq_features_from_row(batch, max_length)
            user_emb = model(features)
            loss = ar_loss(user_emb, item_emb, sampler)
            loss.backward()
            optimizer.step()

        # 6. Evaluation
        if epoch % eval_interval == 0:
            hr, ndcg, mrr = eval_metrics_v2_from_tensors(...)
            writer.add_scalar("hr@10", hr)
```

---

## 14.2 Gin Config 해석

```ini
# configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin
train_fn.dataset_name = "ml-1m"
train_fn.max_sequence_length = 200      # 최근 200개 행동
train_fn.local_batch_size = 128         # GPU당 128 시퀀스
train_fn.main_module = "HSTU"           # (or "SASRec" for baseline)
train_fn.item_embedding_dim = 50        # 아이템 벡터 차원
train_fn.dropout_rate = 0.2

hstu_encoder.num_blocks = 8             # STU Layer 8개 스택
hstu_encoder.num_heads = 2              # 2-head attention
hstu_encoder.dqk = 25                   # Query/Key dim per head
hstu_encoder.dv = 25                    # Value dim per head

train_fn.loss_module = "SampledSoftmaxLoss"
train_fn.num_negatives = 128            # 128개 네거티브 샘플
train_fn.temperature = 0.05             # Softmax temperature
train_fn.learning_rate = 1e-3           # AdamW learning rate
```

---

## 14.3 SASRec vs HSTU 코드 비교

| 측면 | SASRec | HSTU |
|------|--------|------|
| Attention | `softmax(QK^T/sqrt(d))` | `SiLU(QK^T)/n` |
| FFN | `Conv1D → ReLU → Conv1D` | `SiLU(U) × attn` (gating) |
| Linear | QKV (3 projections) | UVQK (4 projections) |
| Time encoding | None | `RelativeBucketedTimeAndPositionBasedBias` |
| Config | `sasrec-sampled-softmax-n128-final.gin` | `hstu-sampled-softmax-n128-large-final.gin` |

---

[← 13장](ch13_low_level_ops.md) | [목차](../../README.md) | [15장 →](ch15_dlrmv3_production.md)
''')

# --- Ch15 ---
w(f'{BASE}/part3/ch15_dlrmv3_production.md', f'''# 15장. DLRMv3 프로덕션 코드

---

## 15.1 학습 진입점

```bash
# 4-GPU 학습
LOCAL_WORLD_SIZE=4 WORLD_SIZE=4 python3 \\
  generative_recommenders/dlrm_v3/train/train_ranker.py \\
  --dataset movielens-1m --mode train
```

## 15.2 추론 파이프라인

![Inference]({F}/ch15_inference_pipeline.png)

*[그림 15-1] CPU에서 임베딩 조회 → GPU에서 HSTU 인코딩 → 점수 계산 → Top-K 반환*

### KV Caching

```python
# modules/stu.py:STULayer.cached_forward
def cached_forward(self, delta_x, num_targets, ...):
    # 새 토큰의 UVQK만 계산
    u, q, k, v = hstu_compute_uqvk(delta_x, ...)

    # 이전 캐시 + 새 KV 결합
    full_k, full_v, ... = _construct_full_kv(
        delta_k=k, delta_v=v,
        k_cache=self.k_cache, v_cache=self.v_cache, ...)

    # 새 Q로 전체 KV에 attention
    attn = delta_hstu_mha(delta_q=q, k=full_k, v=full_v, ...)
    return hstu_compute_output(attn, u, delta_x, ...)
```

> **KV Cache 효과**: 시퀀스 길이 N일 때
> - Without cache: O(N²) attention per request
> - With cache: O(N) per new token (이전 KV 재사용)

---

## 15.3 Research vs DLRMv3 비교

| 측면 | Research | DLRMv3 |
|------|----------|--------|
| Tensor | Padded | Jagged |
| Embedding | `nn.Embedding` | `TorchRec EmbeddingCollection` |
| Config | Gin (simple) | Gin + dataclass |
| Training | Single script | `train_ranker.py` (4 modes) |
| Inference | None | MLPerf loadgen |
| KV Cache | Limited | Full support |

---

[← 14장](ch14_research_code.md) | [목차](../../README.md) | [16장 →](../part4/ch16_environment.md)
''')

# --- Ch16 ---
w(f'{BASE}/part4/ch16_environment.md', f'''# 16장. 환경 구축과 실행

---

## 16.1 시스템 요구사항

| 항목 | 최소 | 권장 |
|------|------|------|
| GPU | 24GB HBM (A100) | H100 80GB |
| CUDA | 12.1 | 12.4 |
| Python | 3.10 | 3.10 |
| OS | Ubuntu 20.04 | Ubuntu 22.04 |

## 16.2 설치

```bash
# 1. 레포 클론
git clone https://github.com/meta-recsys/generative-recommenders.git
cd generative-recommenders

# 2. 의존성 설치
pip3 install -r requirements.txt
# torch>=2.6.0, fbgemm_gpu>=1.1.0, torchrec>=1.1.0, gin_config>=0.5.0

# 3. 데이터 전처리
mkdir -p tmp/
python3 preprocess_public_data.py
```

## 16.3 Research 실험 재현

```bash
# MovieLens-1M + HSTU-large (single GPU)
CUDA_VISIBLE_DEVICES=0 python3 main.py \\
  --gin_config_file=configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin \\
  --master_port=12345

# TensorBoard 모니터링
tensorboard --logdir exps/ml-1m-l200/ --port 24001
```

### 예상 결과 (ML-1M, HSTU-large)

| Metric | Expected | Epochs |
|--------|----------|--------|
| HR@10 | ~0.33 | ~80 |
| NDCG@10 | ~0.185 | ~80 |
| Training time | ~30min | 101 (single A100) |

## 16.4 DLRMv3 실행

```bash
# 4-GPU debug 학습
LOCAL_WORLD_SIZE=4 WORLD_SIZE=4 python3 \\
  generative_recommenders/dlrm_v3/train/train_ranker.py \\
  --dataset debug --mode train

# 4-GPU 추론 벤치마크
LOCAL_WORLD_SIZE=4 WORLD_SIZE=4 python3 \\
  generative_recommenders/dlrm_v3/inference/main.py \\
  --dataset debug
```

---

[← 15장](../part3/ch15_dlrmv3_production.md) | [목차](../../README.md) | [17장 →](ch17_hyperparameters.md)
''')

# --- Ch17 ---
w(f'{BASE}/part4/ch17_hyperparameters.md', f'''# 17장. 하이퍼파라미터 튜닝 가이드

---

## 17.1 Research vs Production

![Hyperparameters]({F}/ch17_hyperparams.png)

*[그림 17-1] Research (ML-1M)와 Production (DLRMv3)의 하이퍼파라미터 차이 (log scale)*

---

## 17.2 아키텍처 파라미터

| 파라미터 | Research | Production | 영향 |
|----------|----------|------------|------|
| `item_embedding_dim` | 50 | 512 | 표현력↑, 메모리↑ |
| `num_blocks` (layers) | 8 | 5 | 깊이↑ = 복잡 패턴, 학습 느림 |
| `num_heads` | 2 | 4 | 관점↑, 연산↑ |
| `dqk` (Q/K dim) | 25 | 128 | attention 해상도↑ |
| `max_sequence_length` | 200 | 16,384 | 장기 이력↑, 메모리↑ |

## 17.3 학습 파라미터

| 파라미터 | 값 | 효과 |
|----------|---|------|
| `learning_rate` | 1e-3 | 너무 높으면 불안정, 너무 낮으면 느림 |
| `weight_decay` | 0 ~ 1e-3 | 과적합 방지 |
| `num_negatives` | 128 ~ 512 | ↑ = 더 정확한 loss, 더 느린 학습 |
| `temperature` | 0.05 | ↓ = hard negative에 집중 |
| `dropout_rate` | 0.2 | ↑ = 강한 정규화 |
| `num_warmup_steps` | 0 | >0이면 학습 초기 안정성↑ |

## 17.4 튜닝 가이드

```
Step 1: 작은 모델로 빠르게 실험
  → ML-1M + HSTU (not large) + 30 epochs

Step 2: 학습률 탐색
  → lr = [5e-4, 1e-3, 2e-3] 비교

Step 3: 모델 크기 조절
  → num_blocks = [4, 6, 8], num_heads = [1, 2, 4]

Step 4: 정규화 조절
  → dropout = [0.1, 0.2, 0.3], weight_decay = [0, 1e-4, 1e-3]

Step 5: 대규모 데이터로 확장
  → ML-20M → Amazon Books → 자체 데이터
```

---

[← 16장](ch16_environment.md) | [목차](../../README.md) | [18장 →](ch18_practical_application.md)
''')

# --- Ch18 ---
w(f'{BASE}/part4/ch18_practical_application.md', f'''# 18장. 실무 적용 전략

---

## 18.1 HSTU → 실무 매핑

![Service Mapping]({F}/ch18_service_mapping.png)

*[그림 18-1] HSTU 컴포넌트를 장소추천/콘텐츠피드에 매핑*

---

## 18.2 콘텐츠피드 초개인화

### 데이터 파이프라인 설계

```
HDFS (user action logs)
  → Spark ETL (시간순 정렬, 시퀀스 구성)
  → Parquet (train/eval split)
  → HSTU DataLoader
  → GPU Training (Argo Workflow)
  → Model Checkpoint → Feature Store (user embeddings)
```

### Multi-Task 설정

```python
tasks = [
    TaskConfig(name="click", task_weight=2, task_type=BINARY_CLASSIFICATION),
    TaskConfig(name="dwell_time", task_weight=1, task_type=REGRESSION),
    TaskConfig(name="share", task_weight=4, task_type=BINARY_CLASSIFICATION),
]
# causal_multitask_weights = 0.2 (auxiliary loss weight)
```

---

## 18.3 장소추천 랭커 개선

### HSTU의 장점

| Feature | 활용 |
|---------|------|
| `TimestampEncoder` | 방문 시점 패턴 (평일 점심 vs 주말 저녁) |
| `ActionEncoder` | 검색/클릭/방문/저장 행동 구분 |
| `Target-Aware Attention` | 후보 장소가 관련 이력에 집중 |
| `Multi-Task` | 클릭 + 방문 + 저장 동시 예측 |

### TimestampLayerNormPostprocessor

```python
# 시간대/요일별 추천 패턴 학습
time_duration_features = [
    (3600, 24),   # 시간대 (24시간 주기)
    (86400, 7),   # 요일 (7일 주기)
]
# → sin/cos 인코딩으로 주기적 패턴 포착
```

---

## 18.4 유저 페르소나

```python
# HSTU 인코더를 페르소나 생성기로 활용
user_embeddings = model.encode(user_sequence)  # (B, D)
# → 이 벡터를 Feature Store에 저장
# → 세그먼테이션, 타겟팅, 개인화에 활용

# Argo Workflow: 주기적 배치 추론
# 1. HDFS에서 최신 유저 시퀀스 로드
# 2. HSTU 인코더로 임베딩 생성
# 3. Feature Store 업데이트
```

---

## 18.5 인프라 설계

```
┌─────────────────────────────────────────────┐
│  Argo Workflow (N3R Kubernetes)              │
│                                             │
│  ┌─────────┐  ┌──────────┐  ┌───────────┐  │
│  │ Spark   │→│ GPU      │→│ Feature   │  │
│  │ ETL     │  │ Training │  │ Store     │  │
│  │ (HDFS)  │  │ (DDP x4) │  │ (serving) │  │
│  └─────────┘  └──────────┘  └───────────┘  │
│                                             │
│  Monitoring: TensorBoard + Grafana          │
│  Metrics: HR@K, NDCG@K → CTR, dwell_time   │
└─────────────────────────────────────────────┘
```

---

## 18장 핵심 요약 & 전체 마무리

> **즉시 적용 가능한 패턴**
> 1. `ActionEncoder`: 다중 행동 유형을 비트마스크로 인코딩
> 2. `TimestampEncoder`: 시간 패턴 포착 (log-bucket)
> 3. `Target-Aware Attention`: 후보가 이력에서 관련 정보를 직접 조회
> 4. `Multi-Task`: 클릭 + 체류시간 + 공유를 동시 학습
> 5. `Jagged Tensor`: 가변 길이 시퀀스의 메모리 효율적 처리

> **어렵거나 불필요한 부분**
> - Triton/CUDA 커널 직접 작성 (fbgemm_gpu 라이브러리 활용으로 충분)
> - Trillion-parameter 스케일링 (현재 규모에서는 불필요)
> - MLPerf 벤치마크 통합 (내부 벤치마크 체계 사용)

---

[← 17장](ch17_hyperparameters.md) | [목차](../../README.md)
''')

print("All Part 3+4 markdown done!")
