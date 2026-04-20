# Recommenders 라이브러리 완벽 가이드

**Linux Foundation AI / recommenders-team/recommenders (20K+ stars)**

> 30+ 추천 알고리즘, 20+ 평가 메트릭, 56+ 노트북을 가진 종합 레퍼런스 라이브러리

---

## 스터디 목적 (2026 과제 연결)

| 과제 | 이 레포에서 얻을 것 |
|------|------------------|
| 추천 시뮬레이터 개발 (4Q) | Offline evaluation 방법론, 20+ 메트릭 구현체 직접 차용 |
| ABT framework 구축 (1Q~) | Proxy 지표(NDCG, MAP, Diversity, Serendipity) 정의 참고 |
| 장소추천/콘텐츠피드 베이스라인 | 동일 조건에서 다양한 알고리즘 성능 비교 |

---

## 목차

### Part 1. 라이브러리 개요 (1~3장)

| 장 | 제목 | 핵심 내용 |
|---|------|----------|
| [1장](part1/ch01_overview.md) | 라이브러리 아키텍처 | 디렉토리 구조, 5가지 Core Task, 의존성, 설치 |
| [2장](part1/ch02_algorithms_map.md) | 35+ 알고리즘 지도 | CF/CB/Sequential/Graph/News 카테고리별 분류, 프레임워크별 정리 |
| [3장](part1/ch03_quick_benchmark.md) | 벤치마크 결과 해석 | MovieLens 100k 8개 알고리즘 비교, 속도/정확도 트레이드오프 |

### Part 2. 평가 메트릭 심층 분석 (4~6장)

| 장 | 제목 | 핵심 내용 |
|---|------|----------|
| [4장](part2/ch04_rating_metrics.md) | Rating 메트릭 | RMSE, MAE, R², Explained Variance, AUC, LogLoss |
| [5장](part2/ch05_ranking_metrics.md) | Ranking 메트릭 | Precision@K, Recall@K, NDCG@K, MAP, MRR |
| [6장](part2/ch06_beyond_accuracy.md) | Beyond-Accuracy 메트릭 | Diversity, Novelty, Serendipity, Catalog Coverage |

### Part 3. 핵심 알고리즘 코드 분석 (7~11장)

| 장 | 제목 | 핵심 내용 |
|---|------|----------|
| [7장](part3/ch07_sar.md) | SAR (Simple Algorithm) | Item-item 유사도, 시간 감쇠, 가장 빠른 베이스라인 |
| [8장](part3/ch08_als_mf.md) | ALS & Matrix Factorization | PySpark ALS, SVD, BPR, 분산 학습 |
| [9장](part3/ch09_ncf_deep.md) | NCF & Deep Learning | Neural CF, Wide&Deep, VAE, LightGCN |
| [10장](part3/ch10_sequential.md) | Sequential Models | SASRec, GRU, Caser, SLi-Rec (HSTU 스터디와 연결) |
| [11장](part3/ch11_news_content.md) | News & Content-Based | NRMS, DKN, TF-IDF, MIND 데이터셋 |

### Part 4. 데이터 & 실험 파이프라인 (12~14장)

| 장 | 제목 | 핵심 내용 |
|---|------|----------|
| [12장](part4/ch12_datasets.md) | 데이터셋 & 전처리 | MovieLens, Amazon, MIND, Criteo 로딩/변환 |
| [13장](part4/ch13_splitting.md) | 데이터 분할 전략 | Random, Chronological, Stratified, Cold-Start 처리 |
| [14장](part4/ch14_experiment_pipeline.md) | 실험 파이프라인 | HPO (NNI, Hyperopt), 벤치마크 프레임워크, 재현 가이드 |

### Part 5. 실무 적용 (15~16장)

| 장 | 제목 | 핵심 내용 |
|---|------|----------|
| [15장](part5/ch15_simulator_design.md) | 추천 시뮬레이터 설계 | Offline evaluation 방법론 차용, 메트릭 선택 가이드 |
| [16장](part5/ch16_abt_framework.md) | ABT Framework 설계 | Proxy 지표 매핑, A/B 테스트 연동, 실무 적용 전략 |

---

## 레포 구조 요약

```
recommenders/
├── recommenders/           # Core library
│   ├── models/             # 18개 알고리즘 패밀리
│   ├── datasets/           # 6개 데이터셋 로더
│   ├── evaluation/         # 20+ 메트릭 (Python + Spark)
│   ├── tuning/             # HPO (NNI, parameter sweep)
│   └── utils/              # 유틸리티
├── examples/               # 56+ Jupyter 노트북
│   ├── 00_quick_start/     # 알고리즘별 빠른 시작
│   ├── 02_model_*/         # 알고리즘 Deep Dive
│   ├── 03_evaluate/        # 평가 메트릭 가이드
│   └── 06_benchmarks/      # 알고리즘 비교
└── scenarios/              # 7개 비즈니스 도메인
```

---

## HSTU 스터디와의 관계

| 관점 | HSTU 스터디 (1순위) | Recommenders 스터디 (2순위) |
|------|-------------------|---------------------------|
| 초점 | 단일 SOTA 모델 심층 분석 | 다양한 알고리즘 + 평가 방법론 |
| 기초 지식 | Part 1에서 선형대수~추천시스템 커버 | **이미 커버됨** → 바로 실무 내용 |
| 코드 수준 | GPU 커널, Triton, Jagged Tensor | 알고리즘 로직, 메트릭 구현 |
| 적용 | 콘텐츠피드/장소추천 모델링 | 추천 시뮬레이터 / ABT / 베이스라인 |
