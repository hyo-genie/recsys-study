# 4장. DynamicEmb 연산

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
# 파일 구조: {table_name}_emb_{item}.rank_{rank}.world_size_{ws}
# items: keys, values, scores, opt_values
model.dump(root_path="/checkpoints/epoch_10", rank=rank, world_size=world_size)
model.load(root_path="/checkpoints/epoch_10", rank=rank, world_size=world_size)
```

> **실무 적용**: 유저 Ontology의 유저/POI 임베딩을 DynamicEmb로 관리. 새 POI가 추가되면 실시간 insert, 오래 방문하지 않은 POI는 LRU로 자동 eviction. HDFS 체크포인트와 연동.

---

[← 3장](ch03_dynamicemb_arch.md) | [목차](../README.md) | [5장 →](../part3/ch05_async_kvcache.md)
