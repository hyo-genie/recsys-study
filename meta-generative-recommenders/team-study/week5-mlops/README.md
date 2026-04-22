# 5주차 — MLOps 엔지니어 B

## 주제: 추천 시스템에서 왜 별도 ops 계층이 필요한가

---

## 직접 연결 논문

1. **FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision**
   - [arxiv.org/abs/2407.08691](https://arxiv.org/abs/2407.08691)
   - `ops/cpp/hstu_attention`이 FlashAttention V3 기반 attention 구현

2. **Actions Speak Louder than Words** (ICML'24) — 특히 HSTU efficiency, M-FALCON 부분
   - [arxiv.org/abs/2402.17152](https://arxiv.org/abs/2402.17152)
   - 긴 시퀀스에서의 학습/추론 효율을 강조

## 왜 이 논문인가

README는:
- `ops/triton` — 효율성 실험용 triton kernel
- `ops/cpp` — efficient CUDA kernel
- 특히 `ops/cpp/hstu_attention` — FlashAttention V3 기반 attention 구현

이라고 설명한다. 동시에 메인 논문은 HSTU와 M-FALCON을 통해 긴 시퀀스에서의 효율을 강조한다.

## 발표 포인트

이 회차는 **"GPU 최적화 일반론"이 아니다.** 추천 시스템이 만나는 고유한 병목을 설명하는 시간이다.

> 추천 시스템이 길고 sparse한 sequence, jagged tensor, attention 병목을 만나면 **왜 별도 커널 계층이 필요한지** 설명하는 시간

### 다뤄야 할 질문

- [ ] 추천 시스템의 attention 병목은 NLP/Vision과 어떻게 다른가?
- [ ] Jagged tensor가 왜 표준 CUDA kernel으로 처리하기 어려운가?
- [ ] FlashAttention-3의 핵심 아이디어 (asynchrony, low-precision)는?
- [ ] M-FALCON이 해결하는 구체적 문제는?
- [ ] Triton kernel vs CUDA kernel — 언제 어떤 것을 쓰나?

### 추천 레포 참고 파일

- `ops/triton/` — Triton 커널 디렉토리
- `ops/cpp/` — CUDA 커널 디렉토리
- `ops/cpp/hstu_attention/` — FlashAttention V3 기반 구현

## 참고 자료

- [기존 학습서 13장](../../part3/ch13_ops_kernels.md) — 연산 커널
- [기존 학습서 9장](../../part2/ch09_jagged_tensor.md) — Jagged Tensor

## 발표 자료

> 발표 슬라이드, 노트 등을 이 폴더에 추가하세요.
