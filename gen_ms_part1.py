#!/usr/bin/env python3
"""MS Recommenders Part 1 (ch1-3) figures + markdown"""
import numpy as np, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BASE = '/home1/irteam/work/hstu-study-guide/ms-recommenders'
FIG = f'{BASE}/figures'
for d in [FIG, f'{BASE}/part1', f'{BASE}/part2', f'{BASE}/part3', f'{BASE}/part4', f'{BASE}/part5']:
    os.makedirs(d, exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','axes.unicode_minus':False,'figure.dpi':150})

def sf(fig,n):
    p=f'{FIG}/{n}.png'; fig.savefig(p,bbox_inches='tight',facecolor='white',edgecolor='none'); plt.close(fig)

# ============ Ch01 Figures ============

# Fig: 5 Core Tasks
fig,ax=plt.subplots(figsize=(13,4)); ax.axis('off')
tasks=[
    (1.5, 'Prepare\nData','#E3F2FD','#1565C0','datasets/\nsplitters'),
    (4, 'Model\n(35+ algos)','#E8F5E9','#2E7D32','models/\n18 families'),
    (6.5, 'Evaluate\n(20+ metrics)','#FFF3E0','#E65100','evaluation/\npython & spark'),
    (9, 'Optimize\n(HPO)','#F3E5F5','#6A1B9A','tuning/\nNNI, Hyperopt'),
    (11.5, 'Operationalize\n(Deploy)','#FFEBEE','#C62828','AKS, CosmosDB\nDatabricks'),
]
for x,t,fc,ec,sub in tasks:
    rect=patches.FancyBboxPatch((x-1.1,0.8),2.2,2,boxstyle="round,pad=0.12",facecolor=fc,edgecolor=ec,lw=2.5)
    ax.add_patch(rect); ax.text(x,2.1,t,ha='center',va='center',fontsize=11,fontweight='bold',color=ec)
    ax.text(x,1.15,sub,ha='center',va='center',fontsize=8,color='#777')
for i in range(len(tasks)-1):
    ax.annotate('',xy=(tasks[i+1][0]-1.2,1.8),xytext=(tasks[i][0]+1.2,1.8),
                arrowprops=dict(arrowstyle='->',lw=2.5,color='#666'))
ax.set_xlim(-0.3,13); ax.set_ylim(0.3,3.3)
sf(fig,'ch01_five_tasks')

# Fig: Library architecture
fig,ax=plt.subplots(figsize=(12,6)); ax.axis('off')
mods=[
    (6,5, 'recommenders/', '#F5F5F5','#333', 10, 0.6),
    (2,3.8, 'models/\n18 families\n(SAR, ALS, NCF,\nSASRec, LightGCN...)', '#E3F2FD','#1565C0', 3, 1.6),
    (5.5,3.8, 'datasets/\nMovieLens, Amazon\nMIND, Criteo', '#E8F5E9','#2E7D32', 2.8, 1.2),
    (8.5,3.8, 'evaluation/\n20+ metrics\n(Python + Spark)', '#FFF3E0','#E65100', 2.8, 1.2),
    (11.5,3.8, 'tuning/\nNNI, Hyperopt\nparameter sweep', '#F3E5F5','#6A1B9A', 2.5, 1.0),
    (6,1.5, 'examples/ (56+ notebooks)\n00_quick_start → 06_benchmarks', '#FFFDE7','#F57F17', 8, 0.6),
    (6,0.5, 'scenarios/ (7 domains)\nads, news, retail, travel, gaming...', '#FFEBEE','#C62828', 8, 0.6),
]
for x,y,t,fc,ec,w,h in mods:
    rect=patches.FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle="round,pad=0.1",facecolor=fc,edgecolor=ec,lw=2)
    ax.add_patch(rect); ax.text(x,y,t,ha='center',va='center',fontsize=8,fontweight='bold',color=ec)
for i in [1,2,3,4]:
    ax.annotate('',xy=(mods[i][0],mods[i][1]+mods[i][6]/2),xytext=(6,5-0.3),
                arrowprops=dict(arrowstyle='->',lw=1.5,color='#999'))
ax.set_xlim(-0.5,14); ax.set_ylim(-0.2,5.8)
sf(fig,'ch01_architecture')

# ============ Ch02 Figures ============

# Fig: Algorithm map
fig,ax=plt.subplots(figsize=(14,8)); ax.axis('off')
cats=[
    (1.5, 7, 'Collaborative Filtering', '#1565C0', [
        'SAR (item-item similarity)',
        'ALS (Spark, matrix factorization)',
        'SVD (Surprise library)',
        'BPR (Bayesian Personalized Ranking)',
        'BiVAE (Variational AutoEncoder)',
        'LightFM (hybrid MF)',
        'FM / FFM (Factorization Machines)',
    ]),
    (6, 7, 'Deep Learning', '#2E7D32', [
        'NCF (Neural CF)',
        'Wide & Deep (PyTorch)',
        'RBM (Boltzmann Machine)',
        'LightGCN (Graph Neural Net)',
        'xDeepFM (Feature Interaction)',
        'Multinomial VAE',
        'EmbeddingDotBias',
    ]),
    (10.5, 7, 'Sequential', '#E65100', [
        'SASRec (Self-Attention)',
        'SSEPT (Personalized Transformer)',
        'GRU (Recurrent)',
        'Caser (CNN Sequence)',
        'NextItNet (Dilated CNN)',
        'SLi-Rec (Short/Long-term)',
        'SUM (Multi-Interest)',
    ]),
    (3.5, 2, 'Content-Based / News', '#6A1B9A', [
        'TF-IDF (text similarity)',
        'DKN (Knowledge Graph)',
        'NRMS (Multi-Head Attention)',
        'LSTUR (Long/Short-term)',
        'NPA (Personalized Attention)',
        'NAML (Multi-View)',
        'LightGBM (ranking)',
    ]),
    (9, 2, 'Advanced', '#C62828', [
        'GeoIMC (Geometric MF)',
        'RLRMC (Riemannian MF)',
        'Cornac (library wrapper)',
        'VowpalWabbit (online)',
        'SARplus (distributed)',
        '',
        '',
    ]),
]
for x,y,title,color,items in cats:
    rect=patches.FancyBboxPatch((x-2,y-4.2),4,4.7,boxstyle="round,pad=0.1",
         facecolor='white',edgecolor=color,lw=2)
    ax.add_patch(rect)
    ax.text(x,y+0.2,title,ha='center',va='center',fontsize=11,fontweight='bold',color=color)
    for i,item in enumerate(items):
        if item:
            ax.text(x,y-0.7-i*0.5,item,ha='center',va='center',fontsize=8,color='#444')
ax.text(7,8.5,'35+ Algorithms in 5 Categories',ha='center',fontsize=14,fontweight='bold',color='#333')
ax.set_xlim(-1.5,14); ax.set_ylim(-3,9)
sf(fig,'ch02_algorithm_map')

# Fig: Framework dependency
fig,ax=plt.subplots(figsize=(10,4))
frameworks=['PyTorch','TensorFlow\n1.x','TensorFlow\n2.x/Keras','PySpark','Cornac/\nSurprise','LightGBM/\nScikit']
counts=[3,8,4,1,3,2]
colors=['#E65100','#1565C0','#42A5F5','#2E7D32','#6A1B9A','#C62828']
bars=ax.bar(range(len(frameworks)),counts,color=colors,edgecolor='white')
ax.set_xticks(range(len(frameworks))); ax.set_xticklabels(frameworks,fontsize=9)
ax.set_ylabel('Number of algorithms',fontsize=10)
ax.set_title('Algorithms by Framework',fontsize=12,fontweight='bold')
for b,c in zip(bars,counts):
    ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.2,str(c),ha='center',fontsize=11,fontweight='bold')
ax.grid(True,alpha=0.3,axis='y')
sf(fig,'ch02_frameworks')

# ============ Ch03 Figures ============

# Fig: Benchmark results
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,5.5))

algos=['BiVAE','BPR','SAR','LightGCN','NCF','EmbDotBias','SVD','ALS']
prec=[0.412,0.388,0.331,0.380,0.347,0.104,0.091,0.048]
ndcg=[0.475,0.442,0.382,0.420,0.396,0.118,0.094,0.044]
colors_b=['#1565C0','#1976D2','#1E88E5','#2196F3','#42A5F5','#90CAF9','#BBDEFB','#E3F2FD']

y=np.arange(len(algos))
ax1.barh(y,prec,color=colors_b,edgecolor='white',height=0.6)
for i,v in enumerate(prec):
    ax1.text(v+0.005,i,f'{v:.3f}',va='center',fontsize=9,fontweight='bold')
ax1.set_yticks(y); ax1.set_yticklabels(algos,fontsize=10)
ax1.set_xlabel('Precision@10',fontsize=10)
ax1.set_title('Precision@10 (MovieLens 100k)',fontsize=11,fontweight='bold')
ax1.set_xlim(0,0.5); ax1.grid(True,alpha=0.3,axis='x')

# Speed vs accuracy
train_time=[22.7,4.97,0.23,23.1,113.3,81.8,0.93,12.4]
ax2.scatter(train_time,prec,s=[200]*8,c=colors_b,edgecolor='#333',lw=1.5,zorder=3)
for i,a in enumerate(algos):
    ax2.annotate(a,(train_time[i],prec[i]),fontsize=8,fontweight='bold',
                xytext=(8,5),textcoords='offset points',color='#333')
ax2.set_xlabel('Training Time (seconds)',fontsize=10)
ax2.set_ylabel('Precision@10',fontsize=10)
ax2.set_title('Speed vs Accuracy Trade-off',fontsize=11,fontweight='bold')
ax2.set_xscale('log'); ax2.grid(True,alpha=0.3)
ax2.axhline(y=0.33,color='#E65100',linestyle='--',lw=1,alpha=0.5)
ax2.text(0.5,0.34,'SAR baseline',fontsize=8,color='#E65100')
plt.tight_layout()
sf(fig,'ch03_benchmark')

# Fig: HSTU vs Recommenders comparison
fig,ax=plt.subplots(figsize=(12,5)); ax.axis('off')
rows=[
    ('Aspect','HSTU Study (1st)','Recommenders Study (2nd)'),
    ('Focus','Single SOTA model deep-dive','35+ algorithm library + evaluation'),
    ('Code Level','GPU kernels, Triton, CUDA','Algorithm logic, metric implementations'),
    ('Key Insight','SiLU gating, Jagged Tensor,\nTarget-Aware Attention','Offline evaluation methodology,\nDiversity/Novelty metrics'),
    ('Apply To','Model architecture\n(next-gen ranker)','Evaluation framework\n(simulator + ABT)'),
    ('Baseline','SASRec','SAR, BPR, BiVAE, NCF'),
]
colors_row=[('#F5F5F5','#333'),('#E3F2FD','#1565C0'),('#E3F2FD','#1565C0'),
            ('#FFF3E0','#E65100'),('#E8F5E9','#2E7D32'),('#F3E5F5','#6A1B9A')]
for i,(a,b,c) in enumerate(rows):
    y=4.5-i*0.85
    for j,(text,x,w) in enumerate([(a,1.2,2),(b,4.5,3.5),(c,9,3.5)]):
        fc,ec=colors_row[i] if i>0 else ('#333','white')
        if i==0:
            ax.text(x,y,text,ha='center',va='center',fontsize=10,fontweight='bold',color='white',
                    bbox=dict(boxstyle='round',facecolor='#333',edgecolor='#333'))
        else:
            rect=patches.FancyBboxPatch((x-w/2,y-0.35),w,0.7,boxstyle="round,pad=0.06",
                 facecolor=fc,edgecolor=ec,lw=1)
            ax.add_patch(rect)
            ax.text(x,y,text,ha='center',va='center',fontsize=8,color=ec)
ax.set_xlim(-0.5,12); ax.set_ylim(-0.5,5.5)
sf(fig,'ch03_hstu_vs_recommenders')

print("Part 1 figures done!")

# ============ MARKDOWN ============
F='../figures'

def w(path,content):
    with open(f'{BASE}/{path}','w') as f: f.write(content)
    print(f'  {path}')

w('part1/ch01_overview.md', f'''# 1장. 라이브러리 아키텍처

> recommenders-team/recommenders -- 35+ 알고리즘, 20+ 메트릭, 56+ 노트북

---

## 1.1 5가지 Core Task

![5 Tasks]({F}/ch01_five_tasks.png)

*[그림 1-1] Recommenders 라이브러리의 5단계 워크플로우*

| Task | 디렉토리 | 핵심 기능 |
|------|---------|----------|
| **Prepare Data** | `datasets/`, `examples/01_*` | MovieLens/Amazon/MIND 로딩, train/test 분할 |
| **Model** | `models/` (18 families) | 35+ 알고리즘 학습 |
| **Evaluate** | `evaluation/` | 20+ 메트릭 (rating, ranking, diversity) |
| **Optimize** | `tuning/` | NNI, Hyperopt, AzureML Hyperdrive |
| **Operationalize** | `examples/05_*` | AKS, Cosmos DB, Databricks 배포 |

---

## 1.2 라이브러리 구조

![Architecture]({F}/ch01_architecture.png)

*[그림 1-2] recommenders/ 핵심 모듈 구조*

## 1.3 설치

```bash
# 빠른 설치 (uv 권장)
uv pip install recommenders

# GPU 모델 포함
uv pip install "recommenders[gpu]"

# Spark 알고리즘 포함
uv pip install "recommenders[spark]"

# 전체 설치
uv pip install "recommenders[all]"
```

| Extra | 포함 | 대표 알고리즘 |
|-------|------|-------------|
| core | NumPy, Pandas, scikit-learn | SAR, TF-IDF |
| `[gpu]` | TensorFlow, PyTorch | NCF, SASRec, LightGCN |
| `[spark]` | PySpark 3.3+ | ALS |
| `[experimental]` | LightFM, Surprise, VW | SVD, BPR |

---

[목차](../README.md) | [2장 →](ch02_algorithms_map.md)
''')

w('part1/ch02_algorithms_map.md', f'''# 2장. 35+ 알고리즘 지도

---

## 2.1 카테고리별 알고리즘

![Algorithm Map]({F}/ch02_algorithm_map.png)

*[그림 2-1] 5개 카테고리, 35+ 알고리즘. 각 카테고리의 대표 알고리즘을 표시.*

---

## 2.2 프레임워크별 분포

![Frameworks]({F}/ch02_frameworks.png)

*[그림 2-2] TensorFlow 1.x 기반이 가장 많고, PyTorch (SASRec, Wide&Deep 등)가 성장 중*

---

## 2.3 알고리즘 선택 가이드

| 상황 | 추천 알고리즘 | 이유 |
|------|-------------|------|
| **빠른 베이스라인** | SAR | 0.23초 학습, implicit feedback, 높은 정확도 |
| **정확도 최우선** | BiVAE, BPR | Precision@10 최고 (0.41, 0.39) |
| **시퀀스 패턴** | SASRec | Transformer 기반, HSTU의 baseline |
| **그래프 관계** | LightGCN | User-Item 그래프 구조 활용 |
| **콘텐츠 기반** | NRMS, DKN | 뉴스/기사 추천 (텍스트 활용) |
| **대규모 분산** | ALS (Spark) | PySpark 기반 수억 레코드 처리 |
| **CTR 예측** | xDeepFM, LightGBM | Feature interaction, 랭킹 최적화 |

> **HSTU 스터디 연결**
> - SASRec이 이 라이브러리에도 구현되어 있음 → HSTU의 baseline
> - 이 라이브러리의 SASRec (`models/sasrec/`) vs HSTU의 SASRec (`research/modeling/sequential/sasrec.py`) 비교 가능
> - HSTU 논문의 "SASRec 대비 +56.7% HR@10" 결과를 이 라이브러리로 재현/검증 가능

---

## 2.4 알고리즘 × 데이터 유형 매트릭스

| Algorithm | Explicit Rating | Implicit Feedback | Sequential | Content Features |
|-----------|:-:|:-:|:-:|:-:|
| SAR | | O | | |
| ALS | O | O | | |
| SVD | O | | | |
| BPR | | O | | |
| NCF | | O | | |
| Wide&Deep | | O | | O |
| SASRec | | O | O | |
| LightGCN | | O | | |
| NRMS | | O | | O |
| LightGBM | | O | | O |

---

[← 1장](ch01_overview.md) | [목차](../README.md) | [3장 →](ch03_quick_benchmark.md)
''')

w('part1/ch03_quick_benchmark.md', f'''# 3장. 벤치마크 결과 해석

---

## 3.1 MovieLens 100k 벤치마크

![Benchmark]({F}/ch03_benchmark.png)

*[그림 3-1] 왼쪽: Precision@10 순위 / 오른쪽: 학습시간 vs 정확도 트레이드오프*

### 결과 요약

| Algorithm | Precision@10 | NDCG@10 | MAP | Train Time |
|-----------|:-----------:|:-------:|:---:|:----------:|
| **BiVAE** | **0.412** | **0.475** | **0.146** | 22.7s |
| BPR | 0.388 | 0.442 | 0.132 | 5.0s |
| LightGCN | 0.380 | 0.420 | 0.089 | 23.1s |
| NCF | 0.347 | 0.396 | 0.108 | 113.3s |
| **SAR** | 0.331 | 0.382 | 0.111 | **0.23s** |
| SVD | 0.091 | 0.094 | 0.013 | 0.93s |
| ALS | 0.048 | 0.044 | 0.005 | 12.4s |

> **Key Insights**
> - **BiVAE** = 정확도 1위, but SAR 대비 100배 느림
> - **SAR** = 최고의 가성비 (0.23초에 Precision 0.33)
> - **NCF** = 가장 느림 (113초), 정확도는 중간
> - **ALS** = Spark용이라 소규모 데이터에서는 오히려 불리

---

## 3.2 HSTU 스터디와의 비교

![HSTU vs Recommenders]({F}/ch03_hstu_vs_recommenders.png)

*[그림 3-2] 두 스터디의 초점과 적용 영역 비교*

### 같은 알고리즘, 다른 관점

```
SASRec in Recommenders Library:
  - PyTorch 기반 깔끔한 구현
  - MovieLens/Amazon 데이터 노트북 제공
  - 다른 알고리즘과 동일 조건 비교 가능

SASRec in HSTU Repo:
  - HSTU의 baseline으로만 사용
  - Gin config로 하이퍼파라미터 관리
  - HSTU-large 대비 -56.7% HR@10 (열등)

→ Recommenders의 SASRec으로 베이스라인 구축
→ HSTU 아키텍처로 개선 효과 측정
→ ABT framework에서 이 비교를 자동화
```

---

## 3.3 벤치마크 재현 방법

```python
# examples/06_benchmarks/movielens.ipynb 기반
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_stratified_split
from recommenders.evaluation.python_evaluation import (
    map_at_k, ndcg_at_k, precision_at_k, recall_at_k
)

# 1. 데이터 로드 + 분할
data = movielens.load_pandas_df(size="100k")
train, test = python_stratified_split(data, ratio=0.75)

# 2. 알고리즘 학습 (예: SAR)
from recommenders.models.sar import SARSingleNode
model = SARSingleNode(similarity_type="jaccard", time_decay_coefficient=30)
model.fit(train)

# 3. 예측 + 평가
top_k = model.recommend_k_items(test, top_k=10)
eval_map = map_at_k(test, top_k, k=10)
eval_ndcg = ndcg_at_k(test, top_k, k=10)
eval_prec = precision_at_k(test, top_k, k=10)
```

---

[← 2장](ch02_algorithms_map.md) | [목차](../README.md) | [4장 →](../part2/ch04_rating_metrics.md)
''')

print("Part 1 markdown done!")
