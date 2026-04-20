# NVIDIA recsys-examples 스터디 가이드

**NVIDIA/recsys-examples -- HSTU의 프로덕션급 확장**

> Meta의 Generative Recommenders를 대규모 프로덕션 환경으로 확장: DynamicEmb + Async KV Cache + Megatron-Core

---

## 스터디 목적 (2026 과제 연결)

| 과제 | 이 레포에서 얻을 것 |
|------|------------------|
| 유저 Ontology 구축 | DynamicEmb: 대규모 유저/POI 임베딩 관리, 동적 업데이트 패턴 |
| 멀티모달 기반 개인화 | 멀티모달 벡터 저장, 실시간 ANN 검색 아키텍처 |
| 장소추천 ranker 서빙 | Async KV Cache, CUDA Graph, Triton: 추론 latency 감소 |

---

## 목차

### Part 1. 개요 (1~2장)

| 장 | 제목 | 핵심 |
|---|------|------|
| [1장](part1/ch01_overview.md) | 프로젝트 개요 | 3개 레포 관계, 레포 구조, 핵심 컴포넌트 |
| [2장](part1/ch02_meta_vs_nvidia.md) | Meta GR vs NVIDIA | 7가지 차이점, NVIDIA가 추가한 것 |

### Part 2. DynamicEmb (3~4장)

| 장 | 제목 | 핵심 |
|---|------|------|
| [3장](part2/ch03_dynamicemb_arch.md) | DynamicEmb 아키텍처 | GPU 해시테이블, 메모리 레이아웃, 루프업 파이프라인 |
| [4장](part2/ch04_dynamicemb_ops.md) | DynamicEmb 연산 | Eviction, Sharding, Optimizer, Checkpoint |

### Part 3. 추론 최적화 (5~6장)

| 장 | 제목 | 핵심 |
|---|------|------|
| [5장](part3/ch05_async_kvcache.md) | Async KV Cache | Paged KV, H2D/D2H overlap, 지연 시간 숨기기 |
| [6장](part3/ch06_inference_serving.md) | 추론 서빙 | CUDA Graph, Triton, AOTInductor C++ export |

### Part 4. 분산 학습 (7~8장)

| 장 | 제목 | 핵심 |
|---|------|------|
| [7장](part4/ch07_distributed.md) | 분산 학습 아키텍처 | TorchRec (sparse) + Megatron-Core (dense) |
| [8장](part4/ch08_training_pipeline.md) | 학습 파이프라인 | Config, Dataset, Benchmark, SID-GR |

### Part 5. 적용 (9장)

| 장 | 제목 | 핵심 |
|---|------|------|
| [9장](part5/ch09_application.md) | 실무 적용 전략 | 유저 Ontology, 서빙 최적화, 인프라 설계 |
| [부록](part5/batch-feature-pipeline-comparison.md) | batch-feature-pipeline vs DynamicEmb | 현재 파이프라인과의 Gap 분석, 하이브리드 적용 방안 |
