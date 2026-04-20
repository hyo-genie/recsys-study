# 4장. Online Decoding: sparse_transition

---

## 4.1 Constrained Beam Search 흐름

```
Step 0 (초기):
  LLM logprobs → start_mask 적용 → top_k → beam 초기화

Step 1 ~ d_dense-1 (Dense):
  LLM logprobs → dense_mask[state, :] 적용 → top_k → beam 갱신

Step d_dense ~ L-1 (CSR):
  LLM logprobs → generate_and_apply_logprobs_mask() → top_k → beam 갱신
```

---

## 4.2 Dense 경로 (Level 0 ~ d-1)

| 연산 | 코드 | 복잡도 |
|------|------|--------|
| 유효성 확인 | `mask = dense_mask[state]` | O(1) lookup |
| 다음 상태 | `next = dense_states[state, token]` | O(1) lookup |
| 마스킹 | `logprobs[~mask] = -inf` | O(V) 벡터 연산 |

```python
# 의사 코드 (JAX)
valid_mask = dense_mask[current_states]     # (batch, beam, V)
masked_logprobs = jnp.where(valid_mask, logprobs, -jnp.inf)
top_tokens = jax.lax.top_k(masked_logprobs, beam_size)
next_states = dense_states[current_states, top_tokens]
```

---

## 4.3 CSR 경로 (Level d ~ L-1)

`generate_and_apply_logprobs_mask()` 핵심 로직:

```
1. indptr에서 현재 상태의 전이 범위 조회
   start = indptr[state]
   end   = indptr[state + 1]

2. packed_csr[start:end]에서 (token, next_state) 쌍 burst-read
   → 최대 max_branch_factor개만 읽으면 됨

3. token으로 logprobs 인덱싱
   → safe_logprobs = logprobs[token_ids]

4. 유효하지 않은 위치는 -inf 마스킹
   → validity = (next_state != 0)  # 0 = invalid

5. (safe_logprobs, token_ids, next_states) 반환
```

| 단계 | 연산 | 복잡도 |
|------|------|--------|
| indptr 조회 | 2회 메모리 접근 | O(1) |
| burst-read | 연속 메모리 읽기 | O(K) where K = 분기 수 |
| logprobs gather | 인덱싱 | O(K) |
| **총합** | | **O(K)** — N에 독립적 |

---

## 4.4 Beam Management

| 연산 | JAX | PyTorch |
|------|-----|---------|
| top-k 선택 | `jax.lax.top_k` | `torch.topk` |
| beam gather | einsum one-hot | `torch.gather` |
| 점수 누적 | `prev_scores + logprobs` | `prev_scores + logprobs` |
| 상태 전이 | one-hot contraction | advanced indexing |

### JAX `_gather_beams()` — TPU 최적화

```python
# one-hot contraction (TPU에서 gather보다 빠름)
one_hot = jax.nn.one_hot(beam_indices, old_beam_size)
gathered = jnp.einsum('bm,bo...->bm...', one_hot, source)
```

> **Data Engineer 관점**: `gather` 대신 `einsum`을 쓰는 이유는 TPU가 행렬 연산에 최적화되어 있기 때문.
> GPU에서는 `torch.gather`가 더 효율적 → PyTorch 버전은 gather 사용.

---

## 4.5 JAX vs PyTorch 차이

| 측면 | JAX (decoding_jax.py) | PyTorch (decoding_pt.py) |
|------|----------------------|--------------------------|
| **JIT** | `@jax.jit` + static_argnums | `@torch.inference_mode()` |
| **디바이스** | 암시적 (jax.devices) | 명시적 `device=` 파라미터 |
| **beam gather** | one-hot einsum | torch.gather |
| **softmax** | jax.nn.log_softmax | F.log_softmax |
| **난수** | PRNG key 전달 | 없음 (inference only) |
| **타겟 HW** | TPU | GPU (CUDA) |
| **루프** | jax.lax.fori_loop | Python for loop |

---

[← 3장](ch03_offline_indexing.md) | [목차](../README.md) | [5장 →](ch05_code_walkthrough.md)
