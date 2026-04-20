# 9장. 실무 적용 전략

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
