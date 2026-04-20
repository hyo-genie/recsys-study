# 부록. Recommenders 평가 프레임워크 vs 현재 메트릭

> 현재 CTR/CVR 기반 평가를 어떻게 고도화할 수 있는가

---

## 평가 체계 비교

| 측면 | **현재** | **MS Recommenders** |
|------|-----------------|-------------------|
| **메트릭 종류** | CTR, CVR (2가지) | **20+ 메트릭** |
| **Ranking 메트릭** | 없음 | Precision@K, Recall@K, NDCG@K, MAP, MRR |
| **Diversity 메트릭** | 없음 | Catalog Coverage, Diversity, Novelty, Serendipity |
| **평가 방식** | SQL 집계 → Hive 테이블 | Python/Spark 함수로 즉시 계산 |
| **데이터 분할** | 없음 (online만) | Random, Chronological, Stratified |
| **모델 비교** | A/B 테스트 (수동) | **동일 조건 오프라인 비교 프레임워크** |
| **통계 검정** | 없음 | 지원 가능 (scipy 연동) |

---

## 현재 메트릭 상세

### 온라인 메트릭 (Hive 집계)

```sql
-- 현재 방식: 화면/템플릿/카테고리별 CTR 집계
SELECT screen_name, card_template, category_name,
       COUNT(impression) as imp_count,
       COUNT(click) as click_count,
       click_count / imp_count as ctr
FROM mapdiscover_metrics
WHERE datestr = '20260416'
GROUP BY screen_name, card_template, category_name
```

### 추적하는 액션

| Action Code | 의미 |
|-------------|------|
| `CK_poi-card` | POI 카드 클릭 |
| `CK_save-bmk-place` | 장소 저장 |
| `CK_coupon` | 쿠폰 클릭 |
| `CK_reservation_icon` | 예약 클릭 |
| `CK_review` | 리뷰 클릭 |
| `CK_clip` | 클립 클릭 |

### 문제점

```
1. CTR/CVR만으로는 "좋은 추천"을 판단할 수 없음
   → 인기 아이템만 추천해도 CTR은 높음 (filter bubble)

2. 오프라인 비교 불가
   → 새 모델을 A/B 테스트 없이 사전 검증할 수 없음
   → A/B 테스트 비용이 높음 (트래픽 분할, 기간 소요)

3. Retriever 소스별 비교만 가능
   → "어떤 알고리즘이 더 좋은가"는 알 수 없음
   → KL-divergence 보정이 소스 비율 유지에만 사용됨
```

---

## MS Recommenders로 추가 가능한 메트릭

### Layer 1: Ranking Quality (오프라인 proxy)

```python
from recommenders.evaluation.python_evaluation import (
    precision_at_k, recall_at_k, ndcg_at_k, map_at_k
)

# 현재 시스템에 없는 핵심 메트릭
results = {
    "NDCG@10": ndcg_at_k(test, top_k, k=10),    # 순위 품질
    "Precision@10": precision_at_k(test, top_k, k=10),  # 정확도
    "MAP": map_at_k(test, top_k, k=10),           # 평균 정밀도
    "Recall@10": recall_at_k(test, top_k, k=10),  # 커버리지
}
```

### Layer 2: Beyond-Accuracy (guard rail)

```python
from recommenders.evaluation.python_evaluation import (
    diversity, novelty, catalog_coverage, serendipity
)

# Filter bubble 방지를 위한 guard rail 메트릭
guards = {
    "Diversity": diversity(train, top_k),           # 추천 다양성
    "Novelty": novelty(train, top_k),               # 비인기 아이템 노출
    "Coverage": catalog_coverage(train, top_k),     # 카탈로그 활용
    "Serendipity": serendipity(train, top_k),       # 의외성+관련성
}
```

### Layer 3: Online↔Offline 상관관계

| Offline Metric | Online Metric | 기대 상관 |
|----------------|---------------|----------|
| NDCG@10 | CTR | 높음 (순위 좋으면 클릭↑) |
| Precision@10 | 저장률 | 중간 (정확하면 저장↑) |
| Diversity | 체류시간 | 높음 (다양하면 탐색↑) |
| Novelty | 신규 POI 방문 | 높음 (새 장소 발견) |
| Coverage | 롱테일 매출 | 중간 (카탈로그 활용↑) |

---

## 데이터 분할 전략 도입

### 현재: 분할 없음 (온라인 CTR만)

### 도입: Chronological Split (추천 시뮬레이터의 기반)

```python
from recommenders.datasets.python_splitters import python_chrono_split

# 장소추천 유저 행동 데이터
# 과거 4주 = 학습, 최근 1주 = 평가
train, test = python_chrono_split(
    user_actions_df,
    ratio=0.8,
    col_timestamp="action_time"
)

# 동일 train/test로 여러 모델 비교
for model in [ListNet, SASRec, HSTU, SAR]:
    model.fit(train)
    predictions = model.predict(test)
    ndcg = ndcg_at_k(test, predictions, k=10)
    print(f"{model.__class__.__name__}: NDCG@10 = {ndcg:.4f}")
```

---

## ABT Framework 적용 제안

```
현재 A/B 테스트 흐름:
  Go API → experimentKey/experimentValue → 트래픽 분할 → CTR 비교 (수주 소요)

ABT Framework 도입 후:
  1. Offline Simulator (수분)
     → Chrono split → 모델 A/B 학습 → NDCG + Diversity 비교
     → "HSTU가 ListNet 대비 NDCG +15%, Diversity +30%"

  2. 소규모 Online A/B (1주)
     → Offline에서 유의미한 모델만 A/B 진행
     → CTR + 체류시간 확인

  3. 전체 배포
     → Guard rail 확인: Diversity >= 0.5, Coverage >= 0.3
     → 통과 시 전체 트래픽 적용

효과: A/B 테스트 비용 70%↓ (사전 필터링), 의사결정 속도 3x↑
```

---

## 즉시 적용 가능한 Quick Win

| Action | 난이도 | 효과 | 코드 출처 |
|--------|--------|------|----------|
| NDCG@10 오프라인 메트릭 추가 | 낮음 | 모델 품질 정량화 | `evaluation/python_evaluation.py` |
| Chrono split 도입 | 낮음 | 오프라인 비교 가능 | `datasets/python_splitters.py` |
| Diversity 메트릭 추가 | 낮음 | Filter bubble 감지 | `evaluation/python_evaluation.py` |
| SAR 베이스라인 추가 | 중간 | 딥러닝 모델의 개선 효과 측정 | `models/sar/sar_singlenode.py` |
| 오프라인 시뮬레이터 프로토타입 | 중간 | A/B 사전 필터링 | `examples/06_benchmarks/` |

---

[← 16장](ch16_abt_framework.md) | [목차](../README.md)
