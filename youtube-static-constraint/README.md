# YouTube STATIC Constrained Decoding 스터디 가이드

**youtube/static-constraint-decoding -- LLM 기반 추천의 Constrained Decoding**

> Generative Retrieval에서 LLM 출력을 유효한 아이템으로 제한하는 고성능 알고리즘

---

## 스터디 목적

| 주제 | 이 레포에서 얻을 것 |
|------|------------------|
| Generative Retrieval 이해 | 기존 Retrieval-Ranking 파이프라인 → LLM 기반 생성형 검색으로의 패러다임 전환 |
| Constrained Decoding | Trie → CSR 변환, Dense/Sparse 하이브리드 마스킹 |
| 가속기 최적화 | JAX/TPU + PyTorch/GPU에서 벡터화된 희소 연산 |
| 프로덕션 배포 | YouTube 대규모 추천 시스템에 실제 적용된 사례 |

---

## 목차

### Part 1. 배경 지식 (1~2장)

| 장 | 제목 | 핵심 |
|---|------|------|
| [1장](part1/ch01_background.md) | Generative Retrieval & Semantic ID | 기존 추천 vs 생성형 추천, Semantic ID, Constrained Decoding 필요성 |
| [2장](part1/ch02_trie_and_csr.md) | Trie 구조와 CSR 포맷 | Prefix Trie, CSR 희소 행렬, STATIC 하이브리드 설계 |

### Part 2. 코드 워크스루 (3~5장)

| 장 | 제목 | 핵심 |
|---|------|------|
| [3장](part2/ch03_offline_indexing.md) | Offline Indexing | build_static_index() 8단계, 입출력 텐서 해부 |
| [4장](part2/ch04_online_decoding.md) | Online Decoding | sparse_transition 흐름, Dense/CSR 분기, beam search |
| [5장](part2/ch05_code_walkthrough.md) | 코드 구조 & 실습 | 레포 구조, JAX vs PyTorch 차이, example.ipynb 재현 |

### Part 3. 성능 분석 (6장)

| 장 | 제목 | 핵심 |
|---|------|------|
| [6장](part3/ch06_benchmarks.md) | 벤치마크 & 베이스라인 | CPU Trie / Hash Bitmap / PPV vs STATIC, 948x 속도 향상 |

### Part 4. 적용 (7장)

| 장 | 제목 | 핵심 |
|---|------|------|
| [7장](part4/ch07_application.md) | 실무 적용 & 확장 | Generative Retrieval 전체 그림, HSTU 연결, 적용 시나리오 |

---

## 핵심 수치

| 항목 | 값 |
|------|-----|
| CPU Trie 대비 속도 | **948x** |
| 가속기 대비 속도 (PPV) | **47-1,033x** |
| 스텝당 마스킹 지연 | **0.033ms** |
| 추론 시간 대비 오버헤드 | **0.25%** |
| 코드 규모 | ~800 LOC (Python) |
| 외부 의존성 | NumPy, JAX or PyTorch |

---

## 참고

- [youtube/static-constraint-decoding](https://github.com/youtube/static-constraint-decoding)
- [Vectorizing the Trie (arXiv:2602.22647)](https://arxiv.org/abs/2602.22647)
- [Colab Example](https://colab.research.google.com/github/youtube/static-constraint-decoding/blob/main/example.ipynb)
