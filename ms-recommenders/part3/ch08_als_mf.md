# 8장. ALS & Matrix Factorization

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
