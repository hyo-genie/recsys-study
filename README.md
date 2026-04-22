# 빅테크 추천시스템 스터디

> Meta, Microsoft, NVIDIA, YouTube의 추천 시스템 코드를 직접 뜯어보고, 2026 과제에 적용하기 위한 학습서

---

## 대상 독자

- ML 모델링/임베딩 경험이 없는 **Data Engineer**
- HDFS, Spark, Kubernetes, Argo Workflow 배경
- 추천 시스템 아키텍처를 이해하고 실무에 적용하려는 엔지니어

## 2026 과제 연결

| 과제 | 1순위 Meta HSTU | 2순위 MS Recommenders | 3순위 NVIDIA |
|------|---------------|---------------------|-------------|
| 콘텐츠피드 초개인화 | STU Layer, Multi-task | SASRec 베이스라인 | - |
| 장소추천 ranker | Target-Aware Attention | 오프라인 비교 | 서빙 최적화 |
| 추천 시뮬레이터 | - | **20+ 메트릭, Split 전략** | - |
| ABT framework | - | **Proxy/Guard 메트릭** | - |
| 유저 Ontology | 시퀀스 인코딩 | - | **DynamicEmb** |

---

## 목차

### [1. Meta Generative Recommenders](meta-generative-recommenders/) (18장)

HSTU 아키텍처 — 추천을 생성 모델로 재정의한 ICML'24 논문 구현체

| Part | 장 | 핵심 |
|------|---|------|
| Part 1. 기초 지식 | [1장](meta-generative-recommenders/part1/ch01_linear_algebra.md)~[6장](meta-generative-recommenders/part1/ch06_recsys.md) | 선형대수 → ML → 딥러닝 → 임베딩 → 어텐션 → 추천시스템 |
| Part 2. HSTU 아키텍처 | [7장](meta-generative-recommenders/part2/ch07_paper_overview.md)~[9장](meta-generative-recommenders/part2/ch09_jagged_tensor.md) | 논문 개요, STU Layer, Jagged Tensor |
| Part 3. 코드 워크스루 | [10장](meta-generative-recommenders/part3/ch10_repo_structure.md)~[15장](meta-generative-recommenders/part3/ch15_dlrmv3_production.md) | 레포 구조, 데이터, 모듈, 연산, Research, DLRMv3 |
| Part 4. 실습/적용 | [16장](meta-generative-recommenders/part4/ch16_environment.md)~[18장](meta-generative-recommenders/part4/ch18_practical_application.md) | 환경, 하이퍼파라미터, 실무 적용 |
| **부록** | [현재 시스템 비교](meta-generative-recommenders/part4/current-comparison.md) | **HSTU vs ListNet(현재 ranker) Gap 분석, 도입 로드맵** |
| **[6주 팀 스터디](meta-generative-recommenders/team-study/)** | ML + MLOps 합동 | 논문 연결 + 역할별 발표 (week1~6) |

### [2. MS Recommenders](ms-recommenders/) (16장)

30+ 알고리즘, 20+ 평가 메트릭 종합 라이브러리

| Part | 장 | 핵심 |
|------|---|------|
| Part 1. 라이브러리 개요 | [1장](ms-recommenders/part1/ch01_overview.md)~[3장](ms-recommenders/part1/ch03_quick_benchmark.md) | 아키텍처, 알고리즘 지도, 벤치마크 |
| Part 2. 평가 메트릭 | [4장](ms-recommenders/part2/ch04_rating_metrics.md)~[6장](ms-recommenders/part2/ch06_beyond_accuracy.md) | Rating, Ranking, Beyond-Accuracy |
| Part 3. 알고리즘 코드 | [7장](ms-recommenders/part3/ch07_sar.md)~[11장](ms-recommenders/part3/ch11_news_content.md) | SAR, ALS, NCF, Sequential, News |
| Part 4. 실험 파이프라인 | [12장](ms-recommenders/part4/ch12_datasets.md)~[14장](ms-recommenders/part4/ch14_experiment_pipeline.md) | 데이터, Split, HPO |
| Part 5. 실무 적용 | [15장](ms-recommenders/part5/ch15_simulator_design.md)~[16장](ms-recommenders/part5/ch16_abt_framework.md) | 시뮬레이터, ABT Framework |
| **부록** | [현재 시스템 비교](ms-recommenders/part5/current-comparison.md) | **현재 CTR/CVR → NDCG+Diversity 고도화 방안** |

### [3. NVIDIA recsys-examples](nvidia-recsys-examples/) (9장)

HSTU의 프로덕션급 확장 — DynamicEmb, Async KV Cache, Megatron

| Part | 장 | 핵심 |
|------|---|------|
| Part 1. 개요 | [1장](nvidia-recsys-examples/part1/ch01_overview.md)~[2장](nvidia-recsys-examples/part1/ch02_meta_vs_nvidia.md) | 3개 레포 관계, Meta vs NVIDIA |
| Part 2. DynamicEmb | [3장](nvidia-recsys-examples/part2/ch03_dynamicemb_arch.md)~[4장](nvidia-recsys-examples/part2/ch04_dynamicemb_ops.md) | GPU 해시테이블, Eviction, Sharding |
| Part 3. 추론 최적화 | [5장](nvidia-recsys-examples/part3/ch05_async_kvcache.md)~[6장](nvidia-recsys-examples/part3/ch06_inference_serving.md) | Async KV Cache, CUDA Graph, Triton |
| Part 4. 분산 학습 | [7장](nvidia-recsys-examples/part4/ch07_distributed.md)~[8장](nvidia-recsys-examples/part4/ch08_training_pipeline.md) | TorchRec + Megatron, SID-GR |
| Part 5. 적용 | [9장](nvidia-recsys-examples/part5/ch09_application.md) | 유저 Ontology, 서빙 최적화 |
| **부록** | [batch-feature-pipeline 비교](nvidia-recsys-examples/part5/batch-feature-pipeline-comparison.md) | **DynamicEmb vs batch-feature-pipeline 배치 파이프라인 Gap** |

### [4. YouTube STATIC Constrained Decoding](youtube-static-constraint/) (7장)

LLM 기반 Generative Retrieval의 Constrained Decoding — 948x 속도 향상

| Part | 장 | 핵심 |
|------|---|------|
| Part 1. 배경 지식 | [1장](youtube-static-constraint/part1/ch01_background.md)~[2장](youtube-static-constraint/part1/ch02_trie_and_csr.md) | Generative Retrieval, Semantic ID, Trie→CSR |
| Part 2. 코드 워크스루 | [3장](youtube-static-constraint/part2/ch03_offline_indexing.md)~[5장](youtube-static-constraint/part2/ch05_code_walkthrough.md) | Offline Indexing, Online Decoding, 실습 |
| Part 3. 성능 분석 | [6장](youtube-static-constraint/part3/ch06_benchmarks.md) | CPU Trie/Hash/PPV vs STATIC, 948x 속도 |
| Part 4. 적용 | [7장](youtube-static-constraint/part4/ch07_application.md) | Generative Retrieval 전체 그림, HSTU 연결 |

---

## 현재 시스템 vs 빅테크 비교 요약

| 컴포넌트 | **현재** | **빅테크 (학습 후 적용)** |
|----------|-----------------|----------------------|
| Ranking Model | ListNet (3-layer MLP) | **HSTU** (STU Layer × 8) |
| Evaluation | CTR/CVR (Hive SQL) | **NDCG + Diversity + Coverage** (20+ 메트릭) |
| Embedding Mgmt | batch-feature-pipeline (7일 배치) | **DynamicEmb** (실시간 GPU 해시) |
| Serving | Go + KServe (MLP) | **Async KV Cache + CUDA Graph** (3-8x↑) |
| A/B Testing | experimentKey 수동 | **Offline Simulator → Online ABT** |

> 각 스터디의 **부록**에 상세 비교 + 도입 로드맵이 포함되어 있습니다.

---

## 레포지토리 구조

```
recsys-study/
├── README.md
├── meta-generative-recommenders/     # 1순위 (18장 + 부록)
│   └── team-study/                   #   └─ 6주 팀 스터디 (ML + MLOps)
├── ms-recommenders/                  # 2순위 (16장 + 부록)
├── nvidia-recsys-examples/           # 3순위 (9장 + 부록)
└── youtube-static-constraint/        # 4순위 (7장 + 13 그래프)
```

**총 50장 + 부록 3편, 80개 그래프/다이어그램**

---

## 참고 레포지토리

- [meta-recsys/generative-recommenders](https://github.com/meta-recsys/generative-recommenders) — HSTU (ICML'24)
- [recommenders-team/recommenders](https://github.com/recommenders-team/recommenders) — 30+ 추천 알고리즘
- [NVIDIA/recsys-examples](https://github.com/NVIDIA/recsys-examples) — 프로덕션급 서빙
- [youtube/static-constraint-decoding](https://github.com/youtube/static-constraint-decoding) — LLM 구조화 출력

## 참고 논문

- [Actions Speak Louder than Words](https://arxiv.org/abs/2402.17152) (ICML'24)
- [Self-Attentive Sequential Recommendation (SASRec)](https://arxiv.org/abs/1808.09781) (ICDM'18)
- [Vectorizing the Trie: Efficient Constrained Decoding](https://arxiv.org/abs/2602.22647) (arXiv'26)
