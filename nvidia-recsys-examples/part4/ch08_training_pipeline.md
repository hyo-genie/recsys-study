# 8장. 학습 파이프라인

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
python3 examples/hstu/training/pretrain_gr_ranking.py \
  --gin_config examples/hstu/training/configs/movielen_ranking.gin

# 멀티 GPU (torchrun)
torchrun --nproc_per_node=4 \
  examples/hstu/training/pretrain_gr_ranking.py \
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
