# 14장. 실험 파이프라인

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
