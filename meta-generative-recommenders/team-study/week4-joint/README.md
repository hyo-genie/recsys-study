# 4주차 — 합동 세션 1

## ML 엔지니어 A + MLOps 엔지니어 A

## 주제: 공개 실험 파이프라인과 baseline 해석

---

## 직접 연결 논문

1. **Self-Attentive Sequential Recommendation (SASRec)** (ICDM'18)
   - [arxiv.org/abs/1808.09781](https://arxiv.org/abs/1808.09781)
   - README 결과표의 SASRec baseline

2. **Revisiting Neural Retrieval on Accelerators**
   - [arxiv.org/abs/2306.04039](https://arxiv.org/abs/2306.04039)
   - SASRec row에서 BCE 대신 사용한 sampled softmax loss의 출처

3. **Turning Dross Into Gold Loss: is BERT4Rec really better than SASRec?**
   - [arxiv.org/abs/2309.07602](https://arxiv.org/abs/2309.07602)
   - BERT4Rec·GRU4Rec 비교값의 출처

## 왜 이 논문인가

README가 거의 답을 준다. README는 public experiment 결과표를 제시하면서:
- SASRec row는 원 논문 기반이되 BCE 대신 *Revisiting Neural Retrieval on Accelerators*의 sampled softmax loss를 사용했다고 밝힘
- BERT4Rec·GRU4Rec 비교값은 *Turning Dross Into Gold Loss*에서 가져왔다고 명시

## 역할 분리

### ML 엔지니어 A — Baseline 의미와 Metric 비교

- [ ] HSTU vs SASRec vs BERT4Rec vs GRU4Rec 결과표 해석
- [ ] 각 baseline의 loss function 차이 (BCE vs sampled softmax)
- [ ] Metric 비교의 공정성: 동일 조건에서 비교되고 있는가?
- [ ] HSTU의 성능 우위가 어디서 오는지 해석

### MLOps 엔지니어 A — 실험 실행 파이프라인

- [ ] `preprocess_public_data.py` — 데이터 전처리 흐름
- [ ] `main.py` — 학습 진입점과 실행 옵션
- [ ] `configs/` (gin config) — 실험 설정 관리 방식
- [ ] `exps/` → TensorBoard — 결과 추적 흐름

### 공통 논의

- [ ] 같은 실험을 ML과 MLOps 관점에서 각각 어떻게 해석하나?
- [ ] 재현 실험을 돌려보려면 어떤 순서로 진행해야 하나?

### 추천 레포 참고 파일

- `README.md` — 실험 결과표, baseline 설명
- `preprocess_public_data.py` — 데이터 전처리
- `main.py` — 학습 진입점
- `configs/` — gin 설정 파일들

## 참고 자료

- [기존 학습서 11장](../../part3/ch11_data_pipeline.md) — 데이터 파이프라인
- [기존 학습서 14장](../../part3/ch14_research_experiments.md) — 실험 관련

## 발표 자료

> 발표 슬라이드, 노트 등을 이 폴더에 추가하세요.
