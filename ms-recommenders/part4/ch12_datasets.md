# 12장. 데이터셋 & 전처리

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
