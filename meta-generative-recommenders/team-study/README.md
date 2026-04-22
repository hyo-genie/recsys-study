# Meta Generative Recommenders — 6주 팀 스터디

> [meta-recsys/generative-recommenders](https://github.com/meta-recsys/generative-recommenders) 코드를 ML 엔지니어 + MLOps 엔지니어가 함께 읽는 6주 스터디

---

## 스터디 구성

- **참여 역할:** ML 엔지니어 2명 (A, B) + MLOps 엔지니어 2명 (A, B)
- **방식:** 주차별 발표 (개인 2주 + 합동 1주 × 2 사이클)
- **목표:** HSTU 아키텍처와 Meta 추천 시스템 스택을 ML/MLOps 양쪽 시각에서 이해

## 주차별 일정

| 주차 | 발표자 | 주제 | 핵심 논문 |
|------|--------|------|-----------|
| [1주차](week1-ml/) | ML 엔지니어 A | Meta는 왜 추천을 generative/sequential 문제로 다시 보나 | HSTU, DLRM |
| [2주차](week2-mlops/) | MLOps 엔지니어 A | 레포 구조와 실행 스택으로 보는 Meta식 RecSys 시스템 구성 | TorchRec, FBGEMM |
| [3주차](week3-ml/) | ML 엔지니어 B | 행동 시퀀스와 콘텐츠 신호를 모델 내부에서 어떻게 묶는가 | HSTU, SASRec |
| [4주차](week4-joint/) | ML A + MLOps A | 공개 실험 파이프라인과 baseline 해석 | SASRec, Revisiting Neural Retrieval, Turning Dross Into Gold Loss |
| [5주차](week5-mlops/) | MLOps 엔지니어 B | 추천 시스템에서 왜 별도 ops 계층이 필요한가 | FlashAttention-3, HSTU (M-FALCON) |
| [6주차](week6-joint/) | ML B + MLOps B | 연구 코드가 production benchmark로 넘어가는 방식 | DLRM, MLPerf Inference, HSTU |

## 핵심 축

| 관점 | 논문 흐름 |
|------|-----------|
| **ML** | Actions Speak Louder than Words → DLRM → SASRec |
| **MLOps** | TorchRec → FBGEMM → FlashAttention-3 → MLPerf Inference |
| **공동** | SASRec + Revisiting Neural Retrieval (4주차), DLRM + MLPerf + HSTU (6주차) |

---

## 참고 논문 전체 목록

| 약칭 | 논문 | 사용 주차 |
|------|------|-----------|
| **HSTU** | [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations](https://arxiv.org/abs/2402.17152) (ICML'24) | 1, 3, 5, 6 |
| **DLRM** | [Deep Learning Recommendation Model for Personalization and Recommendation Systems](https://arxiv.org/abs/1906.00091) | 1, 6 |
| **SASRec** | [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781) (ICDM'18) | 3, 4 |
| **TorchRec** | [TorchRec: a PyTorch Domain Library for Recommendation Systems](https://arxiv.org/abs/2104.07958) | 2 |
| **FBGEMM** | [Enabling High-Performance Low-Precision Deep Learning Inference with FBGEMM](https://arxiv.org/abs/2101.05615) | 2 |
| **Revisiting NR** | [Revisiting Neural Retrieval on Accelerators](https://arxiv.org/abs/2306.04039) | 4 |
| **Dross→Gold** | [Turning Dross Into Gold Loss: is BERT4Rec really better than SASRec?](https://arxiv.org/abs/2309.07602) | 4 |
| **FlashAttn3** | [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08691) | 5 |
| **MLPerf** | [MLPerf Inference Benchmark](https://arxiv.org/abs/1911.02549) | 6 |

## 기존 학습서 연결

이 스터디의 배경지식은 [기존 학습서 (18장)](../) 를 참고하세요.
