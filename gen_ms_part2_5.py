#!/usr/bin/env python3
"""MS Recommenders Part 2-5 (ch4-16) figures + markdown"""
import numpy as np, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BASE = '/home1/irteam/work/hstu-study-guide/ms-recommenders'
FIG = f'{BASE}/figures'
plt.rcParams.update({'font.family':'DejaVu Sans','axes.unicode_minus':False,'figure.dpi':150})

def sf(fig,n):
    p=f'{FIG}/{n}.png'; fig.savefig(p,bbox_inches='tight',facecolor='white',edgecolor='none'); plt.close(fig)

# ============ Part 2 Figures ============

# Ch04: Rating metrics
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4.5))
# RMSE vs MAE
np.random.seed(42)
actual=np.random.rand(20)*5
pred_good=actual+np.random.randn(20)*0.3
pred_bad=actual+np.random.randn(20)*1.2
ax1.scatter(range(20),actual,s=60,c='#2E7D32',label='Actual',zorder=3)
ax1.scatter(range(20),pred_good,s=40,c='#1565C0',marker='^',label='Good model',zorder=3)
ax1.scatter(range(20),pred_bad,s=40,c='#C62828',marker='v',label='Bad model',zorder=3)
for i in range(20):
    ax1.plot([i,i],[actual[i],pred_good[i]],'-',color='#1565C0',alpha=0.3,lw=1)
    ax1.plot([i,i],[actual[i],pred_bad[i]],'-',color='#C62828',alpha=0.2,lw=1)
ax1.set_title('Rating Prediction: errors = vertical lines',fontsize=11,fontweight='bold')
ax1.legend(fontsize=9); ax1.grid(True,alpha=0.2); ax1.set_ylabel('Rating')
rmse_g=np.sqrt(np.mean((actual-pred_good)**2)); rmse_b=np.sqrt(np.mean((actual-pred_bad)**2))
ax1.text(10,0.2,f'RMSE: good={rmse_g:.2f}, bad={rmse_b:.2f}',ha='center',fontsize=9,
         bbox=dict(boxstyle='round',facecolor='#FFFDE7'))

# AUC
from numpy import sort
fpr=np.array([0,0.05,0.1,0.2,0.4,0.6,0.8,1.0])
tpr_good=np.array([0,0.3,0.55,0.75,0.88,0.95,0.98,1.0])
tpr_random=fpr
ax2.plot(fpr,tpr_good,'-',lw=2.5,color='#1565C0',label='Good model (AUC=0.87)')
ax2.plot(fpr,tpr_random,'--',lw=2,color='#999',label='Random (AUC=0.50)')
ax2.fill_between(fpr,tpr_good,alpha=0.1,color='#1565C0')
ax2.set_xlabel('False Positive Rate',fontsize=10); ax2.set_ylabel('True Positive Rate',fontsize=10)
ax2.set_title('AUC: Area Under ROC Curve',fontsize=11,fontweight='bold')
ax2.legend(fontsize=9); ax2.grid(True,alpha=0.3)
plt.tight_layout()
sf(fig,'ch04_rating_metrics')

# Ch05: Ranking metrics comparison
fig,ax=plt.subplots(figsize=(12,5))
metrics=['Precision\n@10','Recall\n@10','NDCG\n@10','MAP','MRR']
biave=[0.412,0.219,0.475,0.146,0.35]
sar=[0.331,0.176,0.382,0.111,0.28]
ncf=[0.347,0.181,0.396,0.108,0.30]
x=np.arange(len(metrics)); w=0.25
ax.bar(x-w,biave,w,color='#1565C0',label='BiVAE',edgecolor='white')
ax.bar(x,sar,w,color='#2E7D32',label='SAR',edgecolor='white')
ax.bar(x+w,ncf,w,color='#E65100',label='NCF',edgecolor='white')
ax.set_xticks(x); ax.set_xticklabels(metrics,fontsize=10)
ax.set_ylabel('Score',fontsize=10)
ax.set_title('Ranking Metrics: 3 Algorithms Compared',fontsize=12,fontweight='bold')
ax.legend(fontsize=10); ax.grid(True,alpha=0.3,axis='y')
sf(fig,'ch05_ranking_metrics')

# Ch06: Beyond accuracy
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5))
# Accuracy vs Diversity trade-off
np.random.seed(42)
algos_div=['BiVAE','BPR','SAR','NCF','LightGCN','Random']
accuracy=[0.41,0.39,0.33,0.35,0.38,0.05]
diversity=[0.3,0.45,0.5,0.35,0.55,0.95]
colors_d=['#1565C0','#1976D2','#2E7D32','#E65100','#42A5F5','#999']
ax1.scatter(accuracy,diversity,s=[200]*6,c=colors_d,edgecolor='#333',lw=1.5,zorder=3)
for i,a in enumerate(algos_div):
    ax1.annotate(a,(accuracy[i],diversity[i]),fontsize=9,fontweight='bold',
                xytext=(8,5),textcoords='offset points')
ax1.set_xlabel('Precision@10 (Accuracy)',fontsize=10)
ax1.set_ylabel('Diversity',fontsize=10)
ax1.set_title('Accuracy vs Diversity Trade-off',fontsize=11,fontweight='bold')
ax1.grid(True,alpha=0.3)
ax1.annotate('Ideal: high accuracy\n+ high diversity',xy=(0.42,0.7),fontsize=9,color='#C62828',
            ha='center',bbox=dict(boxstyle='round',facecolor='#FFEBEE'))

# Beyond accuracy metrics
cats=['Coverage','Novelty','Diversity','Serendipity']
als_vals=[0.15,0.3,0.2,0.1]
random_vals=[0.95,0.9,0.95,0.8]
x2=np.arange(len(cats)); w2=0.35
ax2.bar(x2-w2/2,als_vals,w2,color='#1565C0',label='ALS (accurate)',edgecolor='white')
ax2.bar(x2+w2/2,random_vals,w2,color='#999',label='Random (diverse)',edgecolor='white')
ax2.set_xticks(x2); ax2.set_xticklabels(cats,fontsize=10)
ax2.set_title('Beyond-Accuracy: ALS vs Random',fontsize=11,fontweight='bold')
ax2.legend(fontsize=9); ax2.grid(True,alpha=0.3,axis='y')
plt.tight_layout()
sf(fig,'ch06_beyond_accuracy')

# ============ Part 3 Figures ============

# Ch07: SAR algorithm flow
fig,ax=plt.subplots(figsize=(13,4)); ax.axis('off')
steps=[
    (1.5,'User-Item\nInteractions','#E3F2FD','#1565C0'),
    (4.5,'Co-occurrence\nMatrix','#E8F5E9','#2E7D32'),
    (7.5,'Similarity\n(Jaccard/Lift/\nCosine)','#FFF3E0','#E65100'),
    (10.5,'Score =\nAffinity x\nSimilarity','#FFEBEE','#C62828'),
]
for x,t,fc,ec in steps:
    rect=patches.FancyBboxPatch((x-1.2,0.5),2.4,2.2,boxstyle="round,pad=0.12",facecolor=fc,edgecolor=ec,lw=2)
    ax.add_patch(rect); ax.text(x,1.6,t,ha='center',va='center',fontsize=10,fontweight='bold',color=ec)
for i in range(3):
    ax.annotate('',xy=(steps[i+1][0]-1.3,1.6),xytext=(steps[i][0]+1.3,1.6),
                arrowprops=dict(arrowstyle='->',lw=2.5,color='#666'))
ax.text(7,3,'SAR: Simple, Fast (0.23s), No Deep Learning Needed',ha='center',fontsize=12,fontweight='bold')
ax.set_xlim(-0.5,12.5); ax.set_ylim(-0.2,3.5)
sf(fig,'ch07_sar_flow')

# Ch10: Sequential comparison with HSTU
fig,ax=plt.subplots(figsize=(13,5)); ax.axis('off')
models=[
    (2,3.5,'GRU\n(RNN)','#90CAF9','#1565C0','Hidden state\ncarries history'),
    (5,3.5,'Caser\n(CNN)','#A5D6A7','#2E7D32','Conv filters\non sequence'),
    (8,3.5,'SASRec\n(Transformer)','#FFCC80','#E65100','Self-attention\n+ causal mask'),
    (11,3.5,'HSTU\n(STU Layer)','#CE93D8','#6A1B9A','SiLU gating\n+ time encoding\n+ jagged tensor'),
]
for x,y,name,fc,ec,desc in models:
    rect=patches.FancyBboxPatch((x-1.3,y-1.2),2.6,2.4,boxstyle="round,pad=0.1",facecolor=fc,edgecolor=ec,lw=2)
    ax.add_patch(rect)
    ax.text(x,y+0.4,name,ha='center',va='center',fontsize=11,fontweight='bold',color=ec)
    ax.text(x,y-0.5,desc,ha='center',va='center',fontsize=8,color='#555')
for i in range(3):
    ax.annotate('',xy=(models[i+1][0]-1.4,3.5),xytext=(models[i][0]+1.4,3.5),
                arrowprops=dict(arrowstyle='->',lw=2,color='#999'))
ax.text(6.5,1, 'Recommenders library: GRU, Caser, SASRec, SLi-Rec, NextItNet, SUM\n'
        'HSTU repo: SASRec (baseline) + HSTU (SOTA)\n'
        '→ Recommenders로 baseline 구축, HSTU로 개선 효과 측정',
        ha='center',fontsize=9,color='#333',
        bbox=dict(boxstyle='round',facecolor='#FFFDE7',edgecolor='#F9A825'))
ax.set_xlim(-0.5,13); ax.set_ylim(0.2,5.5)
sf(fig,'ch10_sequential_evolution')

# Ch13: Splitting strategies
fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(14,4))
# Random
np.random.seed(42)
for ax,title,color,desc in [
    (ax1,'Random Split','#1565C0','75% train, 25% test\n(shuffled)'),
    (ax2,'Chronological Split','#2E7D32','Past=train, Future=test\n(time-ordered)'),
    (ax3,'Stratified Split','#E65100','Maintain user/item\ndistribution'),
]:
    ax.set_title(title,fontsize=11,fontweight='bold',color=color)
    n=50
    if 'Random' in title:
        data=np.random.rand(n,2); train_idx=np.random.choice(n,int(n*0.75),replace=False)
        test_idx=np.setdiff1d(range(n),train_idx)
        ax.scatter(data[train_idx,0],data[train_idx,1],c=color,s=30,label='Train')
        ax.scatter(data[test_idx,0],data[test_idx,1],c='#C62828',s=30,marker='x',label='Test')
    elif 'Chrono' in title:
        t=np.sort(np.random.rand(n)); y=np.random.rand(n)
        split=int(n*0.75)
        ax.scatter(t[:split],y[:split],c=color,s=30,label='Train (past)')
        ax.scatter(t[split:],y[split:],c='#C62828',s=30,marker='x',label='Test (future)')
        ax.axvline(x=t[split],color='#999',linestyle='--',lw=1.5)
    else:
        for i,c in enumerate(['#1565C0','#2E7D32','#E65100']):
            cx,cy=np.random.rand()*0.6+0.2*i, np.random.rand()*0.5+0.25*i
            pts=np.random.randn(15,2)*0.1+[cx,cy]
            ax.scatter(pts[:11,0],pts[:11,1],c=c,s=30,alpha=0.7)
            ax.scatter(pts[11:,0],pts[11:,1],c='#C62828',s=30,marker='x')
    ax.legend(fontsize=8); ax.grid(True,alpha=0.2)
    ax.text(0.5,-0.12,desc,ha='center',fontsize=8,color='#666',transform=ax.transAxes)
plt.tight_layout()
sf(fig,'ch13_splitting')

# Ch15: Simulator design
fig,ax=plt.subplots(figsize=(13,5)); ax.axis('off')
blocks=[
    (2,4,'Historical\nUser Data\n(HDFS)','#E3F2FD','#1565C0'),
    (5,4,'Splitter\n(chrono split)','#E8F5E9','#2E7D32'),
    (8,4.8,'Train Data','#FFF3E0','#E65100'),
    (8,3.2,'Test Data','#FFF3E0','#E65100'),
    (11,4.8,'Model A\n(HSTU)','#F3E5F5','#6A1B9A'),
    (11,3.2,'Model B\n(SASRec)','#F3E5F5','#6A1B9A'),
    (11,1.5,'Model C\n(SAR)','#F3E5F5','#6A1B9A'),
    (2,1.5,'Metrics\nPrecision, NDCG\nDiversity, Novelty','#FFEBEE','#C62828'),
    (5,1.5,'Compare\n& Report','#FFFDE7','#F57F17'),
]
for x,y,t,fc,ec in blocks:
    w=2.2 if 'Model' not in t else 2
    rect=patches.FancyBboxPatch((x-w/2,y-0.55),w,1.1,boxstyle="round,pad=0.1",facecolor=fc,edgecolor=ec,lw=2)
    ax.add_patch(rect); ax.text(x,y,t,ha='center',va='center',fontsize=8,fontweight='bold',color=ec)
conns=[(0,1),(1,2),(1,3),(2,4),(2,5),(3,4),(3,5),(3,6),(4,8),(5,8),(6,8),(8,7)]
for a,b in conns:
    ax.annotate('',xy=(blocks[b][0]-0.9,blocks[b][1]),xytext=(blocks[a][0]+0.9,blocks[a][1]),
                arrowprops=dict(arrowstyle='->',lw=1.2,color='#888'))
ax.text(6.5,5.8,'Offline Recommendation Simulator',ha='center',fontsize=13,fontweight='bold')
ax.set_xlim(0,13); ax.set_ylim(0.5,6.2)
sf(fig,'ch15_simulator')

# Ch16: ABT Framework
fig,ax=plt.subplots(figsize=(13,5)); ax.axis('off')
layers=[
    (6.5,4.5,'Business Metrics (Online)','#FFEBEE','#C62828',
     'CTR, Dwell Time, Revenue, DAU'),
    (6.5,3,'Proxy Metrics (Offline)','#FFF3E0','#E65100',
     'NDCG@10, MAP, Precision@10, Recall@10'),
    (6.5,1.5,'Beyond-Accuracy (Guard)','#E8F5E9','#2E7D32',
     'Diversity, Novelty, Coverage, Serendipity'),
]
for x,y,title,fc,ec,items in layers:
    rect=patches.FancyBboxPatch((x-4.5,y-0.55),9,1.1,boxstyle="round,pad=0.1",facecolor=fc,edgecolor=ec,lw=2)
    ax.add_patch(rect)
    ax.text(x-3.5,y,title,va='center',fontsize=10,fontweight='bold',color=ec)
    ax.text(x+1.5,y,items,va='center',fontsize=9,color='#555')
for i in range(2):
    ax.annotate('',xy=(6.5,layers[i+1][1]+0.55),xytext=(6.5,layers[i][1]-0.55),
                arrowprops=dict(arrowstyle='<->',lw=2,color='#666'))
ax.text(12,3.75,'correlate',fontsize=9,color='#666',rotation=90,va='center')
ax.text(12,2.25,'guard\nrails',fontsize=9,color='#666',rotation=90,va='center')
ax.text(6.5,5.8,'ABT Framework: 3-Layer Metric Design',ha='center',fontsize=13,fontweight='bold')
ax.text(6.5,0.5,'Recommenders library provides ALL metrics in layers 2 & 3',ha='center',
        fontsize=10,color='#6A1B9A',style='italic',
        bbox=dict(boxstyle='round',facecolor='#F3E5F5'))
ax.set_xlim(0,13); ax.set_ylim(0,6.3)
sf(fig,'ch16_abt_framework')

print("Part 2-5 figures done!")

# ============ MARKDOWN ============
F='../figures'

def w(path,content):
    with open(f'{BASE}/{path}','w') as f: f.write(content)
    print(f'  {path}')

# === Part 2 ===
w('part2/ch04_rating_metrics.md', f'''# 4장. Rating 메트릭

---

## 4.1 Rating 예측 평가

![Rating Metrics]({F}/ch04_rating_metrics.png)

*[그림 4-1] 왼쪽: Rating 예측 오차 시각화 / 오른쪽: AUC — ROC 커브 아래 면적*

| Metric | 수식 | 용도 |
|--------|------|------|
| **RMSE** | `sqrt(mean((actual - pred)^2))` | 평점 예측 정확도 (큰 오차 페널티) |
| **MAE** | `mean(abs(actual - pred))` | 평점 예측 정확도 (균등 페널티) |
| **R²** | `1 - SS_res / SS_tot` | 설명력 (1에 가까울수록 좋음) |
| **AUC** | Area Under ROC Curve | 이진 분류 (클릭/비클릭) 성능 |
| **LogLoss** | `-mean(y*log(p) + (1-y)*log(1-p))` | CTR 예측 정확도 |

```python
from recommenders.evaluation.python_evaluation import (
    rmse, mae, rsquared, exp_var, auc, logloss
)
eval_rmse = rmse(test_df, pred_df, col_prediction='prediction')
eval_auc = auc(test_df, pred_df, col_prediction='prediction')
```

> **HSTU 스터디 연결**: HSTU의 MultitaskModule에서 Regression task = MSE (=RMSE²). 이 라이브러리의 RMSE 구현을 직접 가져와 평가에 사용 가능.

---

[← 3장](../part1/ch03_quick_benchmark.md) | [목차](../README.md) | [5장 →](ch05_ranking_metrics.md)
''')

w('part2/ch05_ranking_metrics.md', f'''# 5장. Ranking 메트릭

---

## 5.1 Top-K 메트릭 비교

![Ranking Metrics]({F}/ch05_ranking_metrics.png)

*[그림 5-1] BiVAE, SAR, NCF의 5가지 Ranking 메트릭 비교*

| Metric | 의미 | 수식 핵심 | K 의존 |
|--------|------|----------|--------|
| **Precision@K** | 추천 K개 중 정답 비율 | `relevant ∩ recommended / K` | O |
| **Recall@K** | 전체 정답 중 추천된 비율 | `relevant ∩ recommended / total_relevant` | O |
| **NDCG@K** | 순위 가중 정답 품질 | `DCG / IDCG` (높은 순위일수록 가치↑) | O |
| **MAP** | 평균 정밀도의 평균 | `mean(AP per user)` | X |
| **MRR** | 첫 정답의 역순위 | `mean(1/rank_of_first_hit)` | X |

```python
from recommenders.evaluation.python_evaluation import (
    precision_at_k, recall_at_k, ndcg_at_k, map_at_k
)
# K=10으로 평가
results = {{
    "Precision@10": precision_at_k(test, top_k, k=10),
    "Recall@10": recall_at_k(test, top_k, k=10),
    "NDCG@10": ndcg_at_k(test, top_k, k=10),
    "MAP": map_at_k(test, top_k, k=10),
}}
```

> **HSTU 스터디 연결**: HSTU 벤치마크의 HR@10 = 이 라이브러리의 Recall@10과 유사 (정답이 1개일 때 동일). NDCG@10은 양쪽 모두 동일한 정의.

---

[← 4장](ch04_rating_metrics.md) | [목차](../README.md) | [6장 →](ch06_beyond_accuracy.md)
''')

w('part2/ch06_beyond_accuracy.md', f'''# 6장. Beyond-Accuracy 메트릭

> 정확도만으로는 부족하다 — Diversity, Novelty, Serendipity, Coverage

---

## 6.1 Accuracy vs Diversity Trade-off

![Beyond Accuracy]({F}/ch06_beyond_accuracy.png)

*[그림 6-1] 왼쪽: 정확도↑ ≠ 다양성↑ (트레이드오프) / 오른쪽: ALS vs Random의 beyond-accuracy 비교*

## 6.2 메트릭 정의

| Metric | 의미 | 수식 핵심 |
|--------|------|----------|
| **Catalog Coverage** | 전체 아이템 중 추천된 비율 | `unique_recommended / total_items` |
| **Diversity** | 추천 리스트 내 아이템 간 비유사도 | `1 - avg(similarity(i, j))` |
| **Novelty** | 얼마나 비인기 아이템을 추천하는가 | `-avg(log2(popularity))` |
| **Serendipity** | 예상 밖이면서 관련성 있는 추천 | `unexpected ∩ relevant / K` |

```python
from recommenders.evaluation.python_evaluation import (
    catalog_coverage, diversity, novelty, serendipity
)
cov = catalog_coverage(train, top_k, col_item='itemID')
div = diversity(train, top_k, col_item='itemID')
nov = novelty(train, top_k, col_item='itemID')
ser = serendipity(train, top_k, col_item='itemID')
```

## 6.3 ABT Framework에서의 활용

| Business Goal | Proxy Metric (Accuracy) | Guard Metric (Beyond) |
|---------------|------------------------|----------------------|
| 클릭률 향상 | Precision@10, NDCG@10 | Diversity (filter bubble 방지) |
| 탐색 촉진 | Recall@10 | Novelty, Serendipity |
| 롱테일 노출 | — | Catalog Coverage |
| 체류시간 | MAP | Diversity + Novelty |

> **핵심 인사이트**: 정확도 메트릭만 최적화하면 인기 아이템만 추천 (filter bubble). Beyond-accuracy 메트릭을 guard rail로 사용하여 다양성을 보장해야 함.

---

[← 5장](ch05_ranking_metrics.md) | [목차](../README.md) | [7장 →](../part3/ch07_sar.md)
''')

# === Part 3 ===
w('part3/ch07_sar.md', f'''# 7장. SAR (Simple Algorithm for Recommendation)

> 가장 빠른 베이스라인 — 0.23초 학습, 딥러닝 불필요

---

## 7.1 알고리즘 흐름

![SAR Flow]({F}/ch07_sar_flow.png)

*[그림 7-1] SAR: 공동발생 → 유사도 → 점수 계산. GPU 불필요, 0.23초에 Precision 0.33.*

### 핵심 수식

```
1. Co-occurrence: C[i,j] = count(users who interacted with both i and j)
2. Similarity: S[i,j] = Jaccard(i,j) = C[i,j] / (C[i,i] + C[j,j] - C[i,j])
3. Affinity: A[u,i] = sum(interactions) * time_decay(t)
4. Score: score[u,j] = A[u,:] @ S[:,j]   (affinity × similarity)
```

```python
from recommenders.models.sar import SARSingleNode

model = SARSingleNode(
    similarity_type="jaccard",      # or "lift", "cosine", "cooccurrence"
    time_decay_coefficient=30,      # exponential time decay (days)
    timedecay_formula=True,
)
model.fit(train_df)
top_k = model.recommend_k_items(test_df, top_k=10, remove_seen=True)
```

> **실무 적용**: SAR를 장소추천 베이스라인으로 사용 가능. 장소 방문 공동발생 + Jaccard 유사도 → 빠르고 해석 가능한 추천. HSTU 적용 전 비교 기준.

---

[← 6장](../part2/ch06_beyond_accuracy.md) | [목차](../README.md) | [8장 →](ch08_als_mf.md)
''')

w('part3/ch08_als_mf.md', f'''# 8장. ALS & Matrix Factorization

---

## 8.1 ALS (Alternating Least Squares)

```python
from pyspark.ml.recommendation import ALS

als = ALS(
    rank=10,              # 임베딩 차원 (= HSTU의 item_embedding_dim)
    maxIter=15,           # 학습 반복 수
    regParam=0.01,        # L2 정규화 (= HSTU의 weight_decay)
    implicitPrefs=True,   # implicit feedback 모드
    userCol="userID", itemCol="itemID", ratingCol="rating",
)
model = als.fit(train_spark_df)
```

## 8.2 SVD & BPR

| Algorithm | Library | Loss | Feedback | 핵심 |
|-----------|---------|------|----------|------|
| **ALS** | PySpark | MSE | Explicit/Implicit | 대규모 분산, Spark 통합 |
| **SVD** | Surprise | MSE | Explicit | 단순, 빠른 baseline |
| **BPR** | Cornac | Ranking (pairwise) | Implicit | pos > neg 순서 학습 |
| **LightFM** | LightFM | WARP/BPR | Both | Feature 지원 (hybrid) |

> **HSTU 스터디 연결**: ALS의 rank = HSTU의 item_embedding_dim. 둘 다 유저/아이템을 저차원 벡터로 표현하지만, ALS는 선형 분해, HSTU는 Transformer 기반 비선형 인코딩.

---

[← 7장](ch07_sar.md) | [목차](../README.md) | [9장 →](ch09_ncf_deep.md)
''')

w('part3/ch09_ncf_deep.md', f'''# 9장. NCF & Deep Learning Models

---

## 9.1 NCF (Neural Collaborative Filtering)

```
Input: (user_id, item_id)
  → User Embedding + Item Embedding
  → GMF path: element-wise product
  → MLP path: concatenate → deep layers
  → NeuMF: combine GMF + MLP → prediction
```

## 9.2 알고리즘 비교표

| Model | Architecture | Framework | Key Innovation |
|-------|-------------|-----------|---------------|
| **NCF** | Embedding → MLP | TensorFlow | Neural interaction |
| **Wide&Deep** | Linear + DNN | PyTorch | Memorization + generalization |
| **LightGCN** | Graph Conv | TensorFlow | User-item graph structure |
| **VAE** | Encoder-Decoder | TF/Keras | Generative model |
| **xDeepFM** | CIN + DNN | TensorFlow | Feature cross |

> **HSTU 스터디 연결**: NCF의 "Embedding → Interaction → Prediction" 패턴은 HSTU의 DlrmHSTU와 동일한 구조. 차이점: NCF는 단일 시점 interaction, HSTU는 전체 시퀀스를 인코딩.

---

[← 8장](ch08_als_mf.md) | [목차](../README.md) | [10장 →](ch10_sequential.md)
''')

w('part3/ch10_sequential.md', f'''# 10장. Sequential Models

> HSTU 스터디와 직접 연결되는 핵심 장

---

## 10.1 Sequential 모델 진화

![Sequential Evolution]({F}/ch10_sequential_evolution.png)

*[그림 10-1] GRU → Caser → SASRec → HSTU. Recommenders 라이브러리에 GRU~SASRec 구현. HSTU는 별도 레포.*

## 10.2 Recommenders의 SASRec

```python
# recommenders/models/sasrec/model.py (PyTorch)
class SASRec:
    def __init__(self, item_num, maxlen, hidden_units, num_blocks, num_heads, dropout_rate):
        self.item_emb = nn.Embedding(item_num + 1, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, hidden_units)
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(hidden_units, num_heads, dropout_rate)
            for _ in range(num_blocks)
        ])
```

### SASRec: 이 라이브러리 vs HSTU 레포

| 측면 | Recommenders SASRec | HSTU Repo SASRec |
|------|-------------------|------------------|
| Framework | PyTorch | PyTorch (gin-config) |
| Attention | Standard softmax | Standard softmax |
| Position | Learned absolute | Learned absolute |
| Loss | BCE | Sampled Softmax |
| Data | Amazon (notebook) | MovieLens, Amazon |
| 목적 | 독립 모델 | HSTU의 baseline |

## 10.3 기타 Sequential 모델

| Model | Mechanism | Key Feature |
|-------|-----------|-------------|
| **GRU** | Recurrent | Hidden state = 이전 행동의 요약 |
| **Caser** | CNN | 수평/수직 convolution 필터 |
| **NextItNet** | Dilated CNN | 넓은 receptive field |
| **SLi-Rec** | Attention + RNN | Short/Long-term 분리 (HSTU와 유사 목적!) |
| **SUM** | Multi-Interest | 여러 관심사를 동시에 모델링 |

> **핵심**: SLi-Rec의 "Short/Long-term 분리" 개념은 HSTU가 해결하려는 동일한 문제. SLi-Rec은 RNN+Attention으로 접근, HSTU는 STU Layer로 접근.

---

[← 9장](ch09_ncf_deep.md) | [목차](../README.md) | [11장 →](ch11_news_content.md)
''')

w('part3/ch11_news_content.md', f'''# 11장. News & Content-Based Models

---

## 11.1 News Recommendation (MIND Dataset)

| Model | Architecture | Key Feature |
|-------|-------------|-------------|
| **NRMS** | Multi-Head Self-Attention | 뉴스+유저 모두 attention |
| **LSTUR** | LSTM + GRU | Long/Short-term 유저 표현 |
| **NPA** | Personalized Attention | 유저별 다른 attention weight |
| **NAML** | Multi-View | 제목+본문+카테고리 결합 |
| **DKN** | Knowledge Graph + CNN | Entity embedding 활용 |

## 11.2 Content-Based

| Model | Approach | Use Case |
|-------|---------|----------|
| **TF-IDF** | Text similarity | 텍스트 기반 유사 아이템 |
| **LightGBM** | Feature ranking | CTR 예측, feature importance |

> **실무 적용**: 콘텐츠피드 = 뉴스/콘텐츠 추천. NRMS의 Multi-Head Attention 패턴을 참고하되, HSTU의 시퀀셜 인코딩이 더 강력. 두 접근을 결합하는 것도 가능.

---

[← 10장](ch10_sequential.md) | [목차](../README.md) | [12장 →](../part4/ch12_datasets.md)
''')

# === Part 4 ===
w('part4/ch12_datasets.md', f'''# 12장. 데이터셋 & 전처리

---

## 12.1 지원 데이터셋

| Dataset | Size | Type | Use Case |
|---------|------|------|----------|
| **MovieLens 100k/1M/20M** | 100K~20M ratings | Explicit | 알고리즘 비교 벤치마크 |
| **Amazon Reviews** | Varies | Implicit | 시퀀셜 추천 (SASRec 등) |
| **MIND** | 1M users, 161K articles | Click logs | 뉴스 추천 |
| **Criteo** | 45M rows | Click logs | CTR 예측 |

```python
from recommenders.datasets import movielens, amazon_reviews, mind

# MovieLens
df = movielens.load_pandas_df(size="100k",
    header=["userID", "itemID", "rating", "timestamp"])

# Amazon (for sequential models)
# amazon_reviews module provides preprocessing utilities

# MIND (for news recommendation)
train_path, valid_path = mind.download_mind(size="small")
```

---

[← 11장](../part3/ch11_news_content.md) | [목차](../README.md) | [13장 →](ch13_splitting.md)
''')

w('part4/ch13_splitting.md', f'''# 13장. 데이터 분할 전략

---

## 13.1 3가지 분할 전략

![Splitting]({F}/ch13_splitting.png)

*[그림 13-1] Random (무작위) / Chronological (시간순) / Stratified (분포 유지)*

| Strategy | 함수 | 용도 | 주의 |
|----------|------|------|------|
| **Random** | `python_random_split` | 일반 평가 | 시간 순서 무시 |
| **Chronological** | `python_chrono_split` | 시퀀셜 모델 | **추천 시뮬레이터에 필수** |
| **Stratified** | `python_stratified_split` | 유저/아이템 분포 유지 | Cold-start 방지 |

```python
from recommenders.datasets.python_splitters import (
    python_random_split,
    python_chrono_split,
    python_stratified_split,
)

# 추천 시뮬레이터용: 시간순 분할 (과거=학습, 미래=평가)
train, test = python_chrono_split(data, ratio=0.75, col_timestamp="timestamp")

# 일반 벤치마크: 층화 분할 (유저별 비율 유지)
train, test = python_stratified_split(data, ratio=0.75)
```

> **HSTU 스터디 연결**: HSTU는 시퀀스의 마지막 아이템을 target으로 사용 (`ignore_last_n=1`). 이것은 chronological split의 특수 케이스. 이 라이브러리의 `python_chrono_split`이 더 유연한 시간 기반 분할을 제공.

---

[← 12장](ch12_datasets.md) | [목차](../README.md) | [14장 →](ch14_experiment_pipeline.md)
''')

w('part4/ch14_experiment_pipeline.md', f'''# 14장. 실험 파이프라인

---

## 14.1 HPO (Hyperparameter Optimization)

| Tool | 방식 | 통합 |
|------|------|------|
| **NNI** | TPE, Random, Bayesian | `tuning/nni/` |
| **Hyperopt** | Tree-structured Parzen Estimators | `hyperopt` dependency |
| **AzureML Hyperdrive** | Cloud-based distributed | `examples/04_*` |

```python
# NNI example (tuning/nni/nni_utils.py)
import nni
params = nni.get_next_parameter()  # NNI provides hyperparams
model = train_model(params)
metric = evaluate(model)
nni.report_final_result(metric)    # Report back to NNI
```

## 14.2 벤치마크 재현 체크리스트

```
1. 데이터: movielens.load_pandas_df(size="100k")
2. 분할: python_stratified_split(data, ratio=0.75)
3. 알고리즘: SAR, NCF, BiVAE, ... (동일 train/test)
4. 메트릭: precision_at_k, ndcg_at_k, map_at_k (동일 K)
5. 비교: 표 + 그래프로 정리
```

---

[← 13장](ch13_splitting.md) | [목차](../README.md) | [15장 →](../part5/ch15_simulator_design.md)
''')

# === Part 5 ===
w('part5/ch15_simulator_design.md', f'''# 15장. 추천 시뮬레이터 설계

> 이 라이브러리의 핵심 가치: offline evaluation 방법론을 직접 차용

---

## 15.1 시뮬레이터 아키텍처

![Simulator]({F}/ch15_simulator.png)

*[그림 15-1] Offline 추천 시뮬레이터: 데이터 → 분할 → 여러 모델 학습 → 동일 메트릭으로 비교*

## 15.2 차용할 구성 요소

| Component | From Library | 파일 |
|-----------|-------------|------|
| **메트릭 계산** | `evaluation/python_evaluation.py` | 20+ 메트릭 구현체 |
| **데이터 분할** | `datasets/python_splitters.py` | Chrono/Stratified split |
| **베이스라인 모델** | `models/sar/`, `models/sasrec/` | SAR, SASRec 구현체 |
| **벤치마크 프레임워크** | `examples/06_benchmarks/` | 비교 파이프라인 |

## 15.3 시뮬레이터 구현 스케치

```python
# simulator.py (concept)
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_chrono_split
from recommenders.evaluation.python_evaluation import *

class RecommenderSimulator:
    def __init__(self, data, split_ratio=0.75):
        self.train, self.test = python_chrono_split(data, ratio=split_ratio)
        self.models = {{}}
        self.results = {{}}

    def add_model(self, name, model):
        self.models[name] = model

    def run(self, k=10):
        for name, model in self.models.items():
            model.fit(self.train)
            top_k = model.recommend_k_items(self.test, top_k=k)
            self.results[name] = {{
                "Precision@K": precision_at_k(self.test, top_k, k=k),
                "NDCG@K": ndcg_at_k(self.test, top_k, k=k),
                "MAP": map_at_k(self.test, top_k, k=k),
                "Diversity": diversity(self.train, top_k),
                "Novelty": novelty(self.train, top_k),
                "Coverage": catalog_coverage(self.train, top_k),
            }}

    def report(self):
        return pd.DataFrame(self.results).T
```

---

[← 14장](../part4/ch14_experiment_pipeline.md) | [목차](../README.md) | [16장 →](ch16_abt_framework.md)
''')

w('part5/ch16_abt_framework.md', f'''# 16장. ABT Framework 설계

---

## 16.1 3-Layer Metric Design

![ABT Framework]({F}/ch16_abt_framework.png)

*[그림 16-1] Business → Proxy → Guard: 3계층 메트릭 설계. Recommenders 라이브러리가 Layer 2, 3의 모든 메트릭을 제공.*

## 16.2 메트릭 매핑

| Layer | Metric | Library Function | 실무 적용 |
|-------|--------|-----------------|------------|
| **Business** | CTR | (online only) | 장소추천 클릭률 |
| **Business** | Dwell Time | (online only) | 콘텐츠피드 체류시간 |
| **Proxy** | NDCG@10 | `ndcg_at_k()` | **Primary offline metric** |
| **Proxy** | Precision@10 | `precision_at_k()` | 정확도 |
| **Proxy** | MAP | `map_at_k()` | 순위 품질 |
| **Guard** | Diversity | `diversity()` | Filter bubble 방지 |
| **Guard** | Novelty | `novelty()` | 롱테일 노출 |
| **Guard** | Coverage | `catalog_coverage()` | 카탈로그 활용도 |

## 16.3 A/B 테스트 연동

```
Offline (Simulator):
  → NDCG@10: Model A = 0.45, Model B = 0.42
  → Diversity: Model A = 0.3, Model B = 0.6
  → Decision: Model B (diversity가 2배, 정확도 차이 적음)

Online (A/B Test):
  → Control: Current model
  → Treatment: Model B
  → Measure: CTR, Dwell Time, DAU
  → Guard rail: Diversity >= 0.5 (offline에서 검증)
```

## 16.4 전체 스터디 요약: HSTU + Recommenders

```
┌─────────────────────────────────────────────────┐
│  HSTU Study (1st Priority)                      │
│  → Model Architecture (next-gen ranker)         │
│  → Target-Aware Attention, SiLU Gating          │
│  → GPU optimization (Triton, Jagged Tensor)     │
└──────────────┬──────────────────────────────────┘
               │ feeds into
┌──────────────▼──────────────────────────────────┐
│  Recommenders Study (2nd Priority)              │
│  → Evaluation Framework (simulator)             │
│  → Baseline Algorithms (SAR, SASRec, etc.)      │
│  → ABT Framework (proxy metrics + guard rails)  │
└──────────────┬──────────────────────────────────┘
               │ enables
┌──────────────▼──────────────────────────────────┐
│  2026 Deliverables                               │
│  → Train HSTU-based ranker                      │
│  → Evaluate with Recommenders metrics           │
│  → A/B test with ABT framework                  │
│  → Deploy on N3R Kubernetes                     │
└─────────────────────────────────────────────────┘
```

---

[← 15장](ch15_simulator_design.md) | [목차](../README.md)
''')

print("All Part 2-5 markdown done!")
