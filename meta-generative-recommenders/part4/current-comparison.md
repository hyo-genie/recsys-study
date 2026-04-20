# 부록. HSTU vs 현재 아키텍처 비교

> 현재 장소추천/콘텐츠피드 시스템과 HSTU 도입 시 변화 분석

---

## 서빙 아키텍처 비교

| 측면 | **현재** | **HSTU 도입 시** |
|------|-----------------|----------------|
| **Ranking Model** | ListNet (3-layer MLP) | HSTU (STU Layer × N) |
| **Loss** | Top-1 ListNet softmax | Sampled Softmax + Multi-task |
| **입력** | Item features (정적 피처) | **User action sequence** (동적) |
| **시간 정보** | 미사용 | **Timestamp encoding** (log-bucket) |
| **행동 유형** | 미구분 (클릭만) | **ActionEncoder** (click/visit/save/search 구분) |
| **후보 인지** | 독립 scoring | **Target-Aware Attention** (후보가 이력 참조) |
| **Multi-task** | 단일 score | 클릭 + 체류시간 + 저장 동시 학습 |
| **Serving** | Go API → KServe (MLP) | Go API → KServe (HSTU + KV Cache) |

---

## 검색/랭킹 파이프라인 비교

### 현재: Two-Stage (Retrieval → ListNet Ranking)

```
Go API (discover/local-context/local-persona/poi2poi)
  ├─ Retriever 1: Popular POI (static index)
  ├─ Retriever 2: User History (OpenSearch)
  ├─ Retriever 3: Demographic (OpenSearch)
  ├─ Retriever 4: Blog/Embedding (Milvus 128-dim)
  ├─ Retriever 5: KNN Vector (OpenSearch 300-dim)
  → Candidate Pool (~100 items)
  → ListNet Ranker (KServe, 3-layer MLP)
  → KL-divergence calibration (retriever source 비율 유지)
  → Top-K Response
```

### HSTU 적용 시: Two-Stage (Retrieval → HSTU Ranking)

```
Go API (동일)
  ├─ 기존 Retriever (OpenSearch, Milvus) → 그대로 유지
  → Candidate Pool (~100 items)
  → HSTU Ranker (KServe, STU Layer × 8)
    ├─ Input: user action sequence + candidate items
    ├─ ActionEncoder: click=2, visit=4, save=8
    ├─ TimestampEncoder: 방문 시간 패턴
    ├─ Target-Aware Attention: 후보별 이력 가중
    └─ Multi-task: P(click) + E(dwell_time) + P(save)
  → Weighted Score = α·P(click) + β·E(dwell) + γ·P(save)
  → Top-K Response
```

---

## 피처 비교

| 피처 | **현재 (ListNet)** | **HSTU** |
|------|-----------------|---------|
| POI 메타데이터 | O (category, location, quality) | O (item embedding) |
| 유저 프로필 | △ (demographic) | O (**Contextual Features**로 시퀀스에 prepend) |
| 행동 시퀀스 | X | **O (핵심 입력)** |
| 행동 유형 구분 | X | **O (ActionEncoder 비트마스크)** |
| 시간 패턴 | X | **O (TimestampEncoder, 시간대/요일)** |
| 후보 정보 | △ (score만) | **O (Target-Aware Attention)** |

---

## 모델 학습 비교

| 측면 | **ListNet (현재)** | **HSTU** |
|------|-----------------|---------|
| 학습 데이터 | Feature 테이블 (Hive) | User action sequence (시간순) |
| 데이터 전처리 | median imputation + one-hot | Embedding lookup + Action encoding |
| GPU | KServe 추론만 | **학습 + 추론 모두 GPU** |
| 분산 학습 | 없음 (단일 GPU) | DDP (또는 Megatron TP+DP) |
| 학습 주기 | 수동 재학습 | 정기 재학습 (Argo Workflow) |
| 파라미터 수 | ~수만 (3-layer MLP) | **수백만~수억** (STU × 8, 임베딩 테이블) |

---

## 도입 로드맵 제안

### Phase 1: 오프라인 검증 (1~2개월)
```
- MovieLens/KuaiRand 데이터로 HSTU 학습 재현
- 장소추천 데이터로 오프라인 벤치마크
  - 비교: ListNet vs SASRec vs HSTU
  - 메트릭: NDCG@10, Precision@10, HR@10 (MS Recommenders 활용)
```

### Phase 2: 온라인 파일럿 (2~3개월)
```
- KServe에 HSTU 모델 배포
- 기존 Go API는 그대로, Ranker만 교체 (A/B test)
- A: ListNet (현재) vs B: HSTU
- 메트릭: CTR, 체류시간, 저장률
```

### Phase 3: 프로덕션 확대 (3~6개월)
```
- DynamicEmb 도입 (NVIDIA 스터디 참고)
- Async KV Cache 적용 (서빙 latency 최적화)
- Multi-task scoring 활용한 최종 랭킹 수식 최적화
```

---

## 유지할 것 vs 바꿀 것

| Component | 유지 | 바꿀 것 |
|-----------|------|---------|
| Go API Gateway | O | |
| Multi-source Retrieval | O | |
| OpenSearch/Milvus | O | |
| KServe Inference | O | |
| Argo Workflow | O | |
| **ListNet Ranker** | | → **HSTU Ranker** |
| **정적 피처 입력** | | → **유저 시퀀스 입력** |
| **단일 score** | | → **Multi-task score** |
| **CTR만 평가** | | → **NDCG + Diversity 평가** |

---

[← 18장](ch18_practical_application.md) | [목차](../../README.md)
