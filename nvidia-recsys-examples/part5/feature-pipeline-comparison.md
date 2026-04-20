# 부록. batch-feature-pipeline vs NVIDIA DynamicEmb 비교

> 현재 피처 파이프라인과 NVIDIA 프로덕션 임베딩 관리의 Gap 분석

---

## 아키텍처 비교

| 측면 | **batch-feature-pipeline** (현재) | **NVIDIA DynamicEmb** |
|------|---------------------|----------------------|
| **아키텍처** | Batch pipeline (Spark → HDFS → Hive) | GPU 해시테이블 (CUDA 커널) |
| **업데이트 주기** | **7~21일** (Argo CronWorkflow) | **실시간** (insert per request) |
| **새 아이템** | 다음 배치까지 대기 | 즉시 insert (zero-collision hash) |
| **비활성 아이템** | 수동 cleanup (retention period) | **자동 LRU/LFU eviction** |
| **임베딩 모델** | Qwen3, RaptorBERT, Word2Vec, Larva | 학습 가능 임베딩 (backprop) |
| **임베딩 차원** | 128~2560 (PCA 축소) | 128~512 (학습 최적화) |
| **저장소** | HDFS + JFS + Hive + pkl cache | GPU HBM + Host memory |
| **서빙** | pkl cache → ComponentRetriever | 직접 GPU lookup (μs 단위) |
| **스케일** | HDFS 기반 무제한 | GPU 메모리 제한 (eviction으로 관리) |
| **파이프라인** | Argo Workflow (K8s) | 학습 루프 내장 (forward/backward) |
| **Orchestration** | 83개 Argo YAML | gin-config + torchrun |

---

## batch-feature-pipeline 현재 데이터 흐름

```
POI 데이터 (HDFS)
  → Spark batch job (7일 주기)
  → Qwen3/RaptorBERT 임베딩 생성 (2560-dim → PCA → 128-dim)
  → Parquet → JFS 캐시
  → Hive/Iceberg 테이블 등록 (feature_hive.py)
  → pkl cache (ComponentRetriever)
  → 서빙

문제:
  - 새 POI 추가 시 최대 7일 대기
  - 비활성 POI도 계속 저장 (수동 cleanup)
  - pkl cache 로딩 시간이 서빙 latency에 포함
```

## DynamicEmb 적용 시 데이터 흐름

```
실시간 요청
  → GPU hash table lookup (μs)
  → 없으면? → insert + 초기 임베딩 (warm-start from batch-feature-pipeline)
  → 있으면? → return embedding + update LRU score
  → 비활성 POI → 자동 LRU eviction (메모리 자동 관리)
  → 서빙 latency: μs 수준

이점:
  - 새 POI 즉시 반영
  - 메모리 자동 관리 (LRU/LFU)
  - 서빙 latency 10x↓
```

---

## 현실적 적용 방안

batch-feature-pipeline를 DynamicEmb로 **전면 대체**하는 것은 비현실적입니다. HDFS/Hive 기반 배치 파이프라인이 팀 전체 시스템에 연결되어 있기 때문입니다.

### 방안 1: 하이브리드 (권장)

```
batch-feature-pipeline (배치, 7일)                NVIDIA DynamicEmb (실시간)
  ├─ Qwen3 임베딩 생성               ├─ batch-feature-pipeline 결과를 warm-start로 로드
  ├─ HDFS/Hive 저장                  ├─ 실시간 업데이트 (새 POI, 행동 반영)
  └─ 기존 downstream 유지             └─ HSTU ranker 서빙 시 직접 lookup

연결점: DynamicEmb.load() ← batch-feature-pipeline의 Parquet/pkl 출력
```

### 방안 2: 서빙 레이어만 교체

```
현재: pkl cache → ComponentRetriever → 서빙 (ms 단위)
변경: DynamicEmb GPU lookup → 서빙 (μs 단위)

batch-feature-pipeline 배치 파이프라인은 그대로, 서빙 시점에서만 DynamicEmb 사용
```

### 방안 3: 신규 모델에만 적용

```
기존 모델: batch-feature-pipeline 파이프라인 유지
신규 HSTU ranker: DynamicEmb + Async KV Cache + Triton 서빙

점진적 마이그레이션: 성과 검증 후 확대
```

---

## batch-feature-pipeline에서 바로 참고할 NVIDIA 패턴

| NVIDIA 패턴 | batch-feature-pipeline 적용 | 난이도 |
|-------------|----------------|--------|
| **LRU eviction** | Hive 테이블에 last_access 컬럼 추가 → 자동 cleanup | 낮음 |
| **Frequency-based admission** | 인기 POI만 캐시에 올리는 로직 (이미 trending 로직 있음) | 낮음 |
| **Checkpoint dump/load** | DynamicEmb와 batch-feature-pipeline 간 임베딩 교환 포맷 정의 | 중간 |
| **Row-wise sharding** | 멀티 GPU 서빙 시 POI를 GPU별로 분산 | 높음 |
| **Async H2D** | 서빙 시 임베딩 프리페치 (현재는 전체 pkl 로드) | 높음 |

---

[← 9장](ch09_application.md) | [목차](../README.md)
