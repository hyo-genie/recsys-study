#!/usr/bin/env python3
"""NVIDIA recsys-examples study: figures + markdown (9 chapters)"""
import numpy as np, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BASE = '/home1/irteam/work/hstu-study-guide/nvidia-recsys-examples'
FIG = f'{BASE}/figures'
for d in [FIG]+[f'{BASE}/part{i}' for i in range(1,6)]:
    os.makedirs(d, exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','axes.unicode_minus':False,'figure.dpi':150})

def sf(fig,n):
    p=f'{FIG}/{n}.png'; fig.savefig(p,bbox_inches='tight',facecolor='white',edgecolor='none'); plt.close(fig)

F='../figures'

def wf(path,content):
    with open(f'{BASE}/{path}','w') as f: f.write(content)
    print(f'  {path}')

# ============ FIGURES ============

# Fig: 3-repo relationship
fig,ax=plt.subplots(figsize=(13,5)); ax.axis('off')
repos=[
    (2,3.5,'Meta\ngenerative-\nrecommenders','#E3F2FD','#1565C0','HSTU Architecture\n(paper + research code)'),
    (6.5,3.5,'NVIDIA\nrecsys-examples','#FFF3E0','#E65100','Production Optimization\n(DynamicEmb + Inference)'),
    (11,3.5,'MS\nRecommenders','#E8F5E9','#2E7D32','Evaluation Framework\n(30+ algos + metrics)'),
]
for x,y,name,fc,ec,desc in repos:
    rect=patches.FancyBboxPatch((x-1.8,y-1.2),3.6,2.8,boxstyle="round,pad=0.12",facecolor=fc,edgecolor=ec,lw=2.5)
    ax.add_patch(rect); ax.text(x,y+0.5,name,ha='center',va='center',fontsize=11,fontweight='bold',color=ec)
    ax.text(x,y-0.5,desc,ha='center',va='center',fontsize=8,color='#555')

ax.annotate('builds on',xy=(4.7,3.5),xytext=(3.8,3.5),arrowprops=dict(arrowstyle='->',lw=2.5,color='#666'))
ax.annotate('evaluates with',xy=(9.2,3.5),xytext=(8.3,3.5),arrowprops=dict(arrowstyle='->',lw=2.5,color='#666'))

ax.text(6.5,0.8,'NVIDIA = Meta HSTU + Production Infra (DynamicEmb, KV Cache, Triton, Megatron)',
        ha='center',fontsize=10,fontweight='bold',color='#333',
        bbox=dict(boxstyle='round',facecolor='#FFFDE7',edgecolor='#F9A825'))
ax.set_xlim(-0.5,13); ax.set_ylim(0,5.5)
sf(fig,'ch01_three_repos')

# Fig: DynamicEmb architecture
fig,ax=plt.subplots(figsize=(13,6)); ax.axis('off')
blocks=[
    (3,5,'GPU HBM','#E3F2FD','#1565C0',5,0.6),
    (1.5,4,'Scored Hash\nTable','#BBDEFB','#1565C0',2.2,0.8),
    (4.5,4,'Embedding\nValues','#BBDEFB','#1565C0',2.2,0.8),
    (3,2.8,'Optimizer\nStates (Adam)','#BBDEFB','#1565C0',4,0.6),
    (9,5,'Host Memory','#E8F5E9','#2E7D32',4,0.6),
    (9,4,'Overflow\nStorage','#C8E6C9','#2E7D32',2.5,0.8),
    (9,2.8,'Eviction\nBuffer','#C8E6C9','#2E7D32',2.5,0.8),
    (6,1.2,'Features: Zero-collision hashing, LRU/LFU eviction,\nAuto-resize, Frequency-based admission, Row-wise sharding','#FFFDE7','#F57F17',10,0.8),
]
for x,y,t,fc,ec,w,h in blocks:
    rect=patches.FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle="round,pad=0.08",facecolor=fc,edgecolor=ec,lw=2)
    ax.add_patch(rect); ax.text(x,y,t,ha='center',va='center',fontsize=9,fontweight='bold',color=ec)
ax.annotate('',xy=(7,4),xytext=(5.6,4),arrowprops=dict(arrowstyle='<->',lw=2,color='#C62828'))
ax.text(6.3,4.3,'H2D/D2H',fontsize=8,color='#C62828',ha='center')
ax.set_xlim(-0.5,12.5); ax.set_ylim(0.3,5.8)
sf(fig,'ch03_dynamicemb')

# Fig: Async KV Cache
fig,ax=plt.subplots(figsize=(13,5)); ax.axis('off')
# Timeline
ax.plot([0.5,12],[3,3],'-',color='#DDD',lw=3)
events=[
    (1,4,'Request\narrives','#E3F2FD','#1565C0'),
    (3,4,'Async H2D\n(background)','#FFF3E0','#E65100'),
    (3,2,'HSTU Layer 1\n(concurrent)','#F3E5F5','#6A1B9A'),
    (5.5,2,'H2D done\n→ merge KV','#E8F5E9','#2E7D32'),
    (7.5,2,'HSTU Layer 2-8\n(with full KV)','#F3E5F5','#6A1B9A'),
    (10,2,'Score +\nTop-K','#FFEBEE','#C62828'),
    (10,4,'Async D2H\n(offload KV)','#FFF3E0','#E65100'),
]
for x,y,t,fc,ec in events:
    rect=patches.FancyBboxPatch((x-1,y-0.5),2,1,boxstyle="round,pad=0.1",facecolor=fc,edgecolor=ec,lw=2)
    ax.add_patch(rect); ax.text(x,y,t,ha='center',va='center',fontsize=8,fontweight='bold',color=ec)
ax.annotate('overlap!',xy=(3,2.5),xytext=(3,3.5),arrowprops=dict(arrowstyle='<->',lw=2,color='#C62828'))
ax.text(6,0.5,'Key: H2D transfer overlaps with HSTU computation → hides latency',ha='center',
        fontsize=10,fontweight='bold',color='#E65100',
        bbox=dict(boxstyle='round',facecolor='#FFF3E0'))
ax.set_xlim(-0.5,12.5); ax.set_ylim(-0.2,5)
sf(fig,'ch05_async_kvcache')

# Fig: Distributed training
fig,ax=plt.subplots(figsize=(13,5)); ax.axis('off')
for i in range(4):
    x=1+i*3; rect=patches.FancyBboxPatch((x-1.2,0.5),2.4,3.5,boxstyle="round,pad=0.1",
             facecolor='white',edgecolor=['#1565C0','#2E7D32','#E65100','#6A1B9A'][i],lw=2)
    ax.add_patch(rect)
    ax.text(x,3.7,f'GPU {i}',ha='center',fontsize=11,fontweight='bold',
            color=['#1565C0','#2E7D32','#E65100','#6A1B9A'][i])
    # Sparse
    rect2=patches.FancyBboxPatch((x-1,3),2,0.5,boxstyle="round,pad=0.04",facecolor='#E3F2FD',edgecolor='#90CAF9',lw=1)
    ax.add_patch(rect2); ax.text(x,3.25,f'DynamicEmb\nShard {i}',ha='center',fontsize=7,color='#1565C0')
    # Dense
    rect3=patches.FancyBboxPatch((x-1,1.8),2,0.7,boxstyle="round,pad=0.04",facecolor='#E8F5E9',edgecolor='#A5D6A7',lw=1)
    ax.add_patch(rect3); ax.text(x,2.15,f'HSTU (TP shard)\n+ MLP',ha='center',fontsize=7,color='#2E7D32')
    rect4=patches.FancyBboxPatch((x-1,0.8),2,0.5,boxstyle="round,pad=0.04",facecolor='#FFEBEE',edgecolor='#EF9A9A',lw=1)
    ax.add_patch(rect4); ax.text(x,1.05,'Megatron DP/TP',ha='center',fontsize=7,color='#C62828')

ax.text(6.5,4.5,'Sparse: TorchRec (row-wise sharding)  |  Dense: Megatron-Core (TP + DP)',
        ha='center',fontsize=10,fontweight='bold')
ax.annotate('',xy=(3.8,3.25),xytext=(3.2,3.25),arrowprops=dict(arrowstyle='<->',lw=1.5,color='#999'))
ax.annotate('',xy=(6.8,3.25),xytext=(6.2,3.25),arrowprops=dict(arrowstyle='<->',lw=1.5,color='#999'))
ax.text(3.5,3.7,'All2All',fontsize=7,color='#999',ha='center')
ax.set_xlim(-1,13); ax.set_ylim(0,5)
sf(fig,'ch07_distributed')

# Fig: Meta vs NVIDIA comparison
fig,ax=plt.subplots(figsize=(12,5)); ax.axis('off')
rows=[
    ('Aspect','Meta GR','NVIDIA RecSys'),
    ('Embedding','Static nn.Embedding','DynamicEmb (GPU hash table)'),
    ('Eviction','None','LRU/LFU auto-eviction'),
    ('KV Cache','Basic (jagged)','Paged + Async H2D/D2H'),
    ('Training','DDP only','Megatron TP + DP + SP'),
    ('Inference','No serving','Triton + AOTInductor + C++'),
    ('FP8','Not supported','Hopper FP8 quantization'),
]
for i,(a,b,c) in enumerate(rows):
    y=5.5-i*0.75
    for j,(text,x,w) in enumerate([(a,1.2,2),(b,4.5,3.5),(c,9,3.5)]):
        if i==0:
            ax.text(x,y,text,ha='center',va='center',fontsize=10,fontweight='bold',color='white',
                    bbox=dict(boxstyle='round',facecolor='#333'))
        else:
            fc='#E3F2FD' if j==1 else '#FFF3E0' if j==2 else '#F5F5F5'
            ec='#1565C0' if j==1 else '#E65100' if j==2 else '#555'
            rect=patches.FancyBboxPatch((x-w/2,y-0.3),w,0.6,boxstyle="round,pad=0.05",facecolor=fc,edgecolor=ec,lw=1)
            ax.add_patch(rect)
            ax.text(x,y,text,ha='center',va='center',fontsize=8,color=ec,fontweight='bold' if j>0 else 'normal')
ax.set_xlim(-0.5,12); ax.set_ylim(0,6.2)
sf(fig,'ch02_meta_vs_nvidia')

# Fig: Inference speedup
fig,ax=plt.subplots(figsize=(8,5))
batch=[1,2,4,8,16]
no_kv=[1.0,1.0,1.0,1.0,1.0]
with_kv=[1.3,1.5,1.9,2.6,3.2]
hstu_only=[3.0,3.8,5.2,7.0,8.0]
ax.plot(batch,no_kv,'o-',lw=2.5,color='#999',label='No optimization (baseline)',markersize=8)
ax.plot(batch,with_kv,'s-',lw=2.5,color='#1565C0',label='KV Cache + CUDA Graph',markersize=8)
ax.plot(batch,hstu_only,'^-',lw=2.5,color='#E65100',label='HSTU block only (cached tokens)',markersize=8)
ax.set_xlabel('Batch Size',fontsize=11); ax.set_ylabel('Speedup (x)',fontsize=11)
ax.set_title('Inference Speedup with KV Cache (L20 GPU)',fontsize=12,fontweight='bold')
ax.legend(fontsize=9); ax.grid(True,alpha=0.3)
ax.fill_between(batch,no_kv,with_kv,alpha=0.1,color='#1565C0')
ax.fill_between(batch,with_kv,hstu_only,alpha=0.1,color='#E65100')
sf(fig,'ch06_inference_speedup')

print("All NVIDIA figures done!")

# ============ README ============
wf('README.md', f'''# NVIDIA recsys-examples 스터디 가이드

**NVIDIA/recsys-examples -- HSTU의 프로덕션급 확장**

> Meta의 Generative Recommenders를 대규모 프로덕션 환경으로 확장: DynamicEmb + Async KV Cache + Megatron-Core

---

## 스터디 목적 (2026 과제 연결)

| 과제 | 이 레포에서 얻을 것 |
|------|------------------|
| 유저 Ontology 구축 | DynamicEmb: 대규모 유저/POI 임베딩 관리, 동적 업데이트 패턴 |
| 멀티모달 기반 개인화 | 멀티모달 벡터 저장, 실시간 ANN 검색 아키텍처 |
| 장소추천 ranker 서빙 | Async KV Cache, CUDA Graph, Triton: 추론 latency 감소 |

---

## 목차

### Part 1. 개요 (1~2장)

| 장 | 제목 | 핵심 |
|---|------|------|
| [1장](part1/ch01_overview.md) | 프로젝트 개요 | 3개 레포 관계, 레포 구조, 핵심 컴포넌트 |
| [2장](part1/ch02_meta_vs_nvidia.md) | Meta GR vs NVIDIA | 7가지 차이점, NVIDIA가 추가한 것 |

### Part 2. DynamicEmb (3~4장)

| 장 | 제목 | 핵심 |
|---|------|------|
| [3장](part2/ch03_dynamicemb_arch.md) | DynamicEmb 아키텍처 | GPU 해시테이블, 메모리 레이아웃, 루프업 파이프라인 |
| [4장](part2/ch04_dynamicemb_ops.md) | DynamicEmb 연산 | Eviction, Sharding, Optimizer, Checkpoint |

### Part 3. 추론 최적화 (5~6장)

| 장 | 제목 | 핵심 |
|---|------|------|
| [5장](part3/ch05_async_kvcache.md) | Async KV Cache | Paged KV, H2D/D2H overlap, 지연 시간 숨기기 |
| [6장](part3/ch06_inference_serving.md) | 추론 서빙 | CUDA Graph, Triton, AOTInductor C++ export |

### Part 4. 분산 학습 (7~8장)

| 장 | 제목 | 핵심 |
|---|------|------|
| [7장](part4/ch07_distributed.md) | 분산 학습 아키텍처 | TorchRec (sparse) + Megatron-Core (dense) |
| [8장](part4/ch08_training_pipeline.md) | 학습 파이프라인 | Config, Dataset, Benchmark, SID-GR |

### Part 5. 적용 (9장)

| 장 | 제목 | 핵심 |
|---|------|------|
| [9장](part5/ch09_application.md) | 실무 적용 전략 | 유저 Ontology, 서빙 최적화, 인프라 설계 |
''')

# ============ CHAPTERS ============

# Ch01
wf('part1/ch01_overview.md', f'''# 1장. 프로젝트 개요

---

## 1.1 3개 레포의 관계

![Three Repos]({F}/ch01_three_repos.png)

*[그림 1-1] Meta = HSTU 논문 구현 → NVIDIA = 프로덕션 확장 → MS Recommenders = 평가 프레임워크*

> **NVIDIA recsys-examples = Meta HSTU + Production Infrastructure**
> - Meta의 HSTU 아키텍처를 그대로 가져오되
> - **DynamicEmb**: GPU 해시테이블 기반 동적 임베딩 (수십억 아이템 대응)
> - **Async KV Cache**: 비동기 H2D 전송으로 추론 latency 숨기기
> - **Megatron-Core**: Tensor Parallel + Data Parallel 분산 학습
> - **Triton/AOTInductor**: 프로덕션 서빙 (C++ export)

## 1.2 레포 구조

```
recsys-examples/
├── corelib/
│   ├── dynamicemb/        # GPU 동적 임베딩 (C++ CUDA + Python)
│   │   ├── src/           # CUDA 커널 (lookup, insert, evict, optimizer)
│   │   └── dynamicemb/    # Python API (config, planner, tables)
│   └── hstu/              # HSTU 어텐션 커널 (deprecated → fbgemm_gpu)
├── examples/
│   ├── hstu/              # HSTU Ranking/Retrieval 모델
│   │   ├── model/         # RankingGR, RetrievalGR, InferenceRankingGR
│   │   ├── modules/       # HSTU layers, processors, KV cache
│   │   ├── training/      # 학습 스크립트 + gin configs
│   │   └── inference/     # 추론 + Triton integration
│   └── sid_gr/            # Semantic ID 기반 생성형 검색 (beam search)
├── docker/                # Dockerfile (CUDA 13.0, FBGEMM, TorchRec, Megatron)
└── third_party/           # FBGEMM submodule (HSTU 커널 최신 위치)
```

## 1.3 핵심 의존성

| Library | Version | Role |
|---------|---------|------|
| **TorchRec** | v1.2.0+ | 분산 임베딩 (sharding, all-to-all) |
| **Megatron-Core** | v0.12.1 | Dense 모델 병렬화 (TP/DP/SP/PP) |
| **FBGEMM** | main | HSTU 어텐션 커널 (Ampere/Hopper/Blackwell) |
| **gin-config** | - | 하이퍼파라미터 설정 |

---

[목차](../README.md) | [2장 →](ch02_meta_vs_nvidia.md)
''')

# Ch02
wf('part1/ch02_meta_vs_nvidia.md', f'''# 2장. Meta GR vs NVIDIA RecSys

---

## 2.1 7가지 핵심 차이

![Meta vs NVIDIA]({F}/ch02_meta_vs_nvidia.png)

*[그림 2-1] Meta는 연구용, NVIDIA는 프로덕션용. 핵심 차이는 임베딩, KV Cache, 분산 학습, 서빙.*

### 상세 비교

| Aspect | Meta GR | NVIDIA RecSys | 왜 중요? |
|--------|---------|--------------|---------|
| **Embedding** | `nn.Embedding` (static) | **DynamicEmb** (GPU hash table) | 수십억 아이템 동적 관리 |
| **Eviction** | 없음 | LRU/LFU 자동 퇴출 | 메모리 효율적 관리 |
| **KV Cache** | 기본 (jagged) | **Paged + Async H2D/D2H** | 추론 latency 3-8x 감소 |
| **Training** | DDP only | **Megatron TP + DP + SP** | 1000+ GPU 확장 |
| **Inference** | 없음 | **Triton + AOTInductor C++** | 프로덕션 서빙 |
| **FP8** | 미지원 | Hopper FP8 양자화 | 메모리/속도 2x 향상 |
| **SID-GR** | 없음 | Semantic ID + Beam Search | 생성형 검색 (retrieval) |

> **HSTU 스터디 연결**: Meta의 `modules/stu.py`가 기본 STU Layer라면, NVIDIA의 `native_hstu_layer.py`는 TP/SP/CUDA Graph이 추가된 프로덕션 버전.

---

[← 1장](ch01_overview.md) | [목차](../README.md) | [3장 →](../part2/ch03_dynamicemb_arch.md)
''')

# Ch03
wf('part2/ch03_dynamicemb_arch.md', f'''# 3장. DynamicEmb 아키텍처

> GPU 해시테이블 기반 동적 임베딩 -- 수십억 아이템을 효율적으로 관리

---

## 3.1 아키텍처 개요

![DynamicEmb]({F}/ch03_dynamicemb.png)

*[그림 3-1] GPU HBM에 해시테이블 + 값 저장, Host에 overflow. 양방향 H2D/D2H 전송.*

## 3.2 vs Static Embedding

| 측면 | `nn.Embedding` (Meta) | DynamicEmb (NVIDIA) |
|------|---------------------|-------------------|
| **크기** | 고정 (num_items × D) | 동적 (자동 확장/축소) |
| **새 아이템** | 재학습 필요 | 실시간 추가 |
| **메모리** | 전체 로드 | LRU/LFU로 hot items만 |
| **충돌** | Hash trick (충돌 허용) | **Zero-collision** (scored eviction) |
| **Backend** | PyTorch Tensor | **C++ CUDA Hash Table** |

## 3.3 Scored Hash Table

```
Lookup Pipeline:
1. hash(key) → bucket_id
2. Linear probe in bucket (bucket_capacity=128)
3. Found? → return embedding + update score (LRU timestamp)
4. Not found? → return zero vector (or insert new)

Insert Pipeline:
1. hash(key) → bucket_id
2. Probe for empty slot
3. Bucket full? → evict lowest-score entry
4. Insert key + embedding + score
```

```python
# corelib/dynamicemb/dynamicemb/dynamicemb_config.py
DynamicEmbTableOptions(
    embedding_dtype=torch.bfloat16,
    dim=128,                       # embedding dimension
    max_capacity=10_000_000,       # max rows per GPU shard
    bucket_capacity=128,           # hash bucket width
    evict_strategy=DynamicEmbEvictStrategy.KLru,  # LRU eviction
    max_load_factor=0.5,           # rehash threshold
)
```

> **DE 관점**: Redis의 LRU eviction + hash table을 GPU HBM에서 CUDA 커널로 구현한 것. `bucket_capacity=128`은 Redis의 `maxmemory-samples`와 유사한 역할.

---

[← 2장](../part1/ch02_meta_vs_nvidia.md) | [목차](../README.md) | [4장 →](ch04_dynamicemb_ops.md)
''')

# Ch04
wf('part2/ch04_dynamicemb_ops.md', f'''# 4장. DynamicEmb 연산

---

## 4.1 Eviction 전략

| Strategy | Score 계산 | 퇴출 기준 |
|----------|-----------|----------|
| **LRU** | GPU nanosecond timer | 가장 오래 전 접근 |
| **LFU** | Access count per bucket | 가장 적게 접근 |
| **Epoch-LRU** | Step counter | Epoch 기반 |
| **No Eviction** | - | 자동 확장 (OOM까지) |

## 4.2 Row-wise Sharding

```
GPU 0: items 0~2.5M    (shard 0)
GPU 1: items 2.5M~5M   (shard 1)
GPU 2: items 5M~7.5M   (shard 2)
GPU 3: items 7.5M~10M  (shard 3)

Lookup: bucketize(key) → target_gpu → All2All → lookup → All2All back
```

## 4.3 Optimizer Integration

```python
# Embedding + optimizer state를 같은 버퍼에 저장
# SGD: [embedding] (dim only)
# Adam: [embedding | m_state | v_state] (dim * 3)
# AdaGrad: [embedding | accumulator] (dim * 2)

# corelib/dynamicemb/dynamicemb/optimizer.py
def get_optimizer_state_dim(optimizer_type, dim, dtype):
    if optimizer_type == EmbOptimType.SGD: return 0
    if optimizer_type == EmbOptimType.ADAM: return 2 * dim
    if optimizer_type == EmbOptimType.EXACT_ADAGRAD: return dim
```

## 4.4 Checkpoint

```python
# 파일 구조: {{table_name}}_emb_{{item}}.rank_{{rank}}.world_size_{{ws}}
# items: keys, values, scores, opt_values
model.dump(root_path="/checkpoints/epoch_10", rank=rank, world_size=world_size)
model.load(root_path="/checkpoints/epoch_10", rank=rank, world_size=world_size)
```

> **실무 적용**: 유저 Ontology의 유저/POI 임베딩을 DynamicEmb로 관리. 새 POI가 추가되면 실시간 insert, 오래 방문하지 않은 POI는 LRU로 자동 eviction. HDFS 체크포인트와 연동.

---

[← 3장](ch03_dynamicemb_arch.md) | [목차](../README.md) | [5장 →](../part3/ch05_async_kvcache.md)
''')

# Ch05
wf('part3/ch05_async_kvcache.md', f'''# 5장. Async KV Cache

> 추론 latency의 핵심 -- H2D 전송과 HSTU 연산을 겹쳐서 지연 숨기기

---

## 5.1 비동기 파이프라인

![Async KV Cache]({F}/ch05_async_kvcache.png)

*[그림 5-1] H2D 전송과 HSTU 연산이 동시에 실행. 전송 시간이 "숨겨짐".*

## 5.2 Paged KV Cache 구조

```python
# KV Cache 테이블 shape:
# [num_layers, num_cache_pages, 2, page_size, num_heads, head_dim]
#                               ^ K and V

# Page-based allocation:
# pages_per_seq = ceil(max_seq_len / page_size)
# LRU eviction when pages exhausted
```

## 5.3 3단계 파이프라인

```
1. prepare_kvcache_async()
   → 별도 CUDA stream에서 Host → GPU 전송 시작
   → 동시에 CPU에서 메타데이터 계산

2. prepare_kvcache_wait()
   → H2D 완료 대기 + 페이지 할당 확정

3. HSTU forward + append_kvcache()
   → HSTU 연산 수행
   → 새로 생성된 KV를 캐시에 추가

4. offload_kvcache() (background)
   → 완료된 KV를 GPU → Host로 비동기 전송
```

> **HSTU 스터디 연결**: Meta의 `STULayer.cached_forward()`는 KV Cache의 기본 개념. NVIDIA는 이를 **Paged + Async**로 확장하여 실제 서빙에서의 latency를 3-8x 줄임.

---

[← 4장](../part2/ch04_dynamicemb_ops.md) | [목차](../README.md) | [6장 →](ch06_inference_serving.md)
''')

# Ch06
wf('part3/ch06_inference_serving.md', f'''# 6장. 추론 서빙 최적화

---

## 6.1 Inference Speedup

![Inference Speedup]({F}/ch06_inference_speedup.png)

*[그림 6-1] KV Cache + CUDA Graph으로 최대 8x 속도 향상 (L20 GPU 기준)*

## 6.2 최적화 기법 3가지

### A. CUDA Graph
```python
# 커널 실행 순서를 한 번 기록 → 반복 재생
# 커널 launch overhead 제거 (특히 small batch에서 효과적)
# 제약: 입력 shape이 동일해야 함 → padding 필요
```

### B. Kernel Fusion
```python
# 여러 연산을 하나의 CUDA 커널로 합침
# 예: LayerNorm + Dropout → triton_layer_norm_dropout
#     AddMM + SiLU → triton_addmm_silu_fwd
# 효과: GPU 메모리 접근 횟수 감소 → bandwidth 병목 해소
```

### C. Triton Server + AOTInductor
```
# 1. 모델 export
torch.export(model) → torch._inductor.aoti_compile_and_package → model.pt2

# 2. Triton Server 배포
Sparse backend (Python): 임베딩 룩업 (GPU당 공유)
Dense backend (C++): HSTU + MLP (GPU당 1개 인스턴스)

# 3. Request flow
Client → Triton → Sparse lookup → Dense inference → Response
```

## 6.3 성능 수치

| Config | Without KV | With KV+CUDA | Speedup |
|--------|-----------|-------------|---------|
| Batch 1, 8 layers | baseline | 1.3x | 30% faster |
| Batch 8, 8 layers | baseline | 2.6x | 160% faster |
| HSTU block only, batch 8 | baseline | **8.0x** | 700% faster |

---

[← 5장](ch05_async_kvcache.md) | [목차](../README.md) | [7장 →](../part4/ch07_distributed.md)
''')

# Ch07
wf('part4/ch07_distributed.md', f'''# 7장. 분산 학습 아키텍처

---

## 7.1 Sparse + Dense 분리

![Distributed]({F}/ch07_distributed.png)

*[그림 7-1] Sparse(임베딩): TorchRec row-wise sharding / Dense(HSTU+MLP): Megatron TP+DP*

## 7.2 병렬화 전략

| Component | Library | Parallelism | Communication |
|-----------|---------|------------|---------------|
| DynamicEmb | TorchRec | Row-wise sharding | All-to-All |
| HSTU Linear | Megatron | **Tensor Parallel** | AllReduce |
| HSTU Attention | Megatron | **Sequence Parallel** | AllGather |
| MLP | Megatron | **Data Parallel** | AllReduce |

## 7.3 Gradient 동기화

```
1. DynamicEmb gradients → scale by TP_size × DP_size
2. AllGather across TP group
3. DDP AllReduce across DP group
4. Optimizer step (per-shard)
```

> **HSTU 스터디 연결**: Meta의 `DDP(model, device_ids=[rank])`는 Data Parallel만 지원. NVIDIA는 **Tensor Parallel**을 추가하여 단일 모델을 여러 GPU에 분할. 이는 모델이 단일 GPU 메모리에 안 들어갈 때 필수.

---

[← 6장](../part3/ch06_inference_serving.md) | [목차](../README.md) | [8장 →](ch08_training_pipeline.md)
''')

# Ch08
wf('part4/ch08_training_pipeline.md', f'''# 8장. 학습 파이프라인

---

## 8.1 지원 데이터셋

| Dataset | Users | Items | Seq Len |
|---------|-------|-------|---------|
| MovieLens 1M | - | - | Standard |
| MovieLens 20M | - | - | Larger |
| **KuaiRand-Pure** | 27K | 7.5K | 1~910 |
| **KuaiRand-1K** | 1K | 4.4M | 10~49K |
| **KuaiRand-27K** | 27K | 32M | 100~228K |

## 8.2 Gin Config

```python
# examples/hstu/training/configs/movielen_ranking.gin
TrainerArgs.train_batch_size = 128
NetworkArgs.num_layers = 1
NetworkArgs.num_attention_heads = 4
NetworkArgs.hidden_size = 128
NetworkArgs.attention_head_size = 32
RankingArgs.prediction_head_arch = [512, 10]
OptimizerArgs.optimizer_str = 'adam'
OptimizerArgs.learning_rate = 1e-3
```

## 8.3 학습 실행

```bash
# 단일 GPU 랭킹 학습
python3 examples/hstu/training/pretrain_gr_ranking.py \\
  --gin_config examples/hstu/training/configs/movielen_ranking.gin

# 멀티 GPU (torchrun)
torchrun --nproc_per_node=4 \\
  examples/hstu/training/pretrain_gr_ranking.py \\
  --gin_config examples/hstu/training/configs/kuairand_ranking.gin
```

## 8.4 SID-GR (Semantic ID 기반 생성형 검색)

```
기존 Retrieval: user_embedding @ item_embedding → Top-K (ANN search)

SID-GR: item을 semantic ID tuple로 tokenize
  → item_id 12345 → (cluster_7, sub_23, leaf_156)
  → Beam search로 ID tuple "생성"
  → 수백~수천 beam width (LLM의 <10 대비)

장점: 임베딩 테이블 대신 작은 codebook 사용 → 메모리 절약
```

---

[← 7장](ch07_distributed.md) | [목차](../README.md) | [9장 →](../part5/ch09_application.md)
''')

# Ch09
wf('part5/ch09_application.md', f'''# 9장. 실무 적용 전략

---

## 9.1 유저 Ontology 구축

### DynamicEmb 활용

```
현재: 유저/POI 임베딩을 Feature Store에 정적 저장
      → 새 POI 추가 시 재학습 필요, 업데이트 주기 느림

개선: DynamicEmb로 GPU에서 실시간 관리
      → 새 POI: insert (해시테이블에 즉시 추가)
      → 비활성 POI: LRU eviction (자동 메모리 해제)
      → 체크포인트: dump/load로 HDFS 연동
```

| 설정 | 값 | 이유 |
|------|---|------|
| `evict_strategy` | LRU | 최근 방문 POI 우선 유지 |
| `max_capacity` | 10M per GPU | 전체 POI 커버 |
| `bucket_capacity` | 128 | 충돌 최소화 |
| `embedding_dtype` | BF16 | 메모리 절약 + 정밀도 유지 |

## 9.2 장소추천 서빙 최적화

### Latency Budget

```
Target: < 50ms per request

현재 bottleneck:
  Embedding lookup: ~10ms
  HSTU forward:     ~30ms (8 layers, seq_len 4096)
  MLP scoring:      ~5ms
  Total:            ~45ms

최적화 적용:
  Async KV Cache:   HSTU → ~15ms (2x faster)
  CUDA Graph:       커널 launch → ~3ms 절약
  Kernel Fusion:    LN+Dropout → ~2ms 절약
  Total:            ~25ms (45% 절감)
```

## 9.3 3개 스터디 통합 적용

```
┌─────────────────────────────────────────────┐
│  Meta HSTU (1순위)                          │
│  → 모델 아키텍처 설계                       │
│  → STU Layer, Target-Aware Attention         │
└──────────────┬──────────────────────────────┘
               ↓
┌──────────────┴──────────────────────────────┐
│  NVIDIA recsys-examples (3순위)             │
│  → 프로덕션 인프라                          │
│  → DynamicEmb, Async KV Cache, Triton       │
└──────────────┬──────────────────────────────┘
               ↓
┌──────────────┴──────────────────────────────┐
│  MS Recommenders (2순위)                    │
│  → 평가 프레임워크                          │
│  → Offline metrics, ABT framework            │
└──────────────┬──────────────────────────────┘
               ↓
┌──────────────┴──────────────────────────────┐
│  서비스 2026 과제                            │
│  ✅ 콘텐츠피드 초개인화 (HSTU 모델)            │
│  ✅ 장소추천 랭커 (DynamicEmb + 서빙)     │
│  ✅ 추천 시뮬레이터 (Recommenders 메트릭)    │
│  ✅ 유저 Ontology (DynamicEmb 동적 관리)    │
└─────────────────────────────────────────────┘
```

---

[← 8장](../part4/ch08_training_pipeline.md) | [목차](../README.md)
''')

print("All NVIDIA markdown done!")
