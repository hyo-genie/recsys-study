# 2주차 — MLOps 엔지니어 A

## 주제: 레포 구조와 실행 스택으로 보는 Meta식 RecSys 시스템 구성

---

## 직접 연결 논문

1. **TorchRec: a PyTorch Domain Library for Recommendation Systems**
   - [arxiv.org/abs/2104.07958](https://arxiv.org/abs/2104.07958)
   - 대규모 추천에서 필요한 sparse·parallelism primitive를 제공하는 라이브러리

2. **Enabling High-Performance Low-Precision Deep Learning Inference with FBGEMM**
   - [arxiv.org/abs/2101.05615](https://arxiv.org/abs/2101.05615)
   - 저정밀 고성능 커널 계층. Meta 추천 모델의 런타임 기반

## 왜 이 논문인가

이 레포는 설치 의존성으로 `torchrec`과 `fbgemm_gpu`를 직접 요구한다. 2주차는 단순히 "폴더 구조를 읽는 시간"이 아니라, **Meta가 추천 시스템을 어떤 런타임/프리미티브 위에 올려 두는지** 보는 시간이다.

즉 2주차의 직접 연결 논문은 **모델 논문보다 시스템 기반 논문**이 더 맞다.

## 발표 포인트

이 주차의 핵심 질문은 하나:

> "Meta는 추천 모델을 그냥 PyTorch 모델로 두는가, 아니면 **sparse/embedding/parallelism 전용 스택** 위에 올리는가?"

답은 후자. 이걸 팀에 공유하면 충분하다.

### 다뤄야 할 질문

- [ ] 레포의 폴더 구조 (`generative_recommenders/`, `ops/`, `configs/`) 는 어떤 관심사를 분리하고 있나?
- [ ] `torchrec`이 제공하는 핵심 abstraction은? (EmbeddingBagCollection, ShardedModule 등)
- [ ] `fbgemm_gpu`가 담당하는 계산 계층은?
- [ ] 일반 PyTorch vs TorchRec 기반 추천 모델의 차이점은?

### 추천 레포 참고 파일

- `setup.py` / `pyproject.toml` — 의존성 목록
- `generative_recommenders/` — 패키지 구조
- `ops/` — triton/cpp 커널 디렉토리
- `configs/` — 실험 설정

## 참고 자료

- [기존 학습서 10장](../../part3/ch10_repo_structure.md) — 레포 구조
- [기존 학습서 9장](../../part2/ch09_jagged_tensor.md) — Jagged Tensor

## 발표 자료

> 발표 슬라이드, 노트 등을 이 폴더에 추가하세요.
