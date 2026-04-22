# 6주차 — 합동 세션 2

## ML 엔지니어 B + MLOps 엔지니어 B

## 주제: 연구 코드가 production benchmark로 넘어가는 방식

---

## 직접 연결 논문

1. **Deep Learning Recommendation Model for Personalization and Recommendation Systems (DLRM)**
   - [arxiv.org/abs/1906.00091](https://arxiv.org/abs/1906.00091)
   - DLRM 원형. HSTU가 다시 DLRM-v3 문맥으로 감싸진 이유를 이해하는 데 필요

2. **MLPerf Inference Benchmark**
   - [arxiv.org/abs/1911.02549](https://arxiv.org/abs/1911.02549)
   - DLRMv3가 MLCommons에서 2026년 MLPerf Inference v6.0의 첫 sequential recommendation benchmark로 소개

3. **Actions Speak Louder than Words** (ICML'24)
   - [arxiv.org/abs/2402.17152](https://arxiv.org/abs/2402.17152)
   - HSTU 기반 아키텍처가 DLRMv3의 핵심

## 왜 이 논문인가

README는 `generative_recommenders/dlrm_v3`를 HSTU를 사용한 DLRM 모델과 training/inference benchmark라고 설명한다. DLRMv3는 MLCommons에서 MLPerf Inference v6.0의 첫 sequential recommendation benchmark로 소개되며 HSTU 기반이다.

마지막 주차에 DLRM 원형, MLPerf inference benchmark, HSTU 메인 논문을 같이 묶는 것이 가장 자연스럽다.

## 역할 분리

### ML 엔지니어 B — 왜 HSTU가 다시 DLRM-v3로 감싸졌는가

- [ ] DLRM → DLRMv2 → DLRMv3의 진화 흐름
- [ ] DLRMv3에서 HSTU가 담당하는 역할
- [ ] 연구 모델(HSTU)이 표준 benchmark 모델(DLRMv3)로 전환되는 과정
- [ ] MLPerf에서 sequential recommendation이 새 카테고리로 추가된 의미

### MLOps 엔지니어 B — 왜 production benchmark는 별도 실행 경로가 필요한가

- [ ] `generative_recommenders/dlrm_v3/` 디렉토리 구조
- [ ] Research code vs Benchmark code의 차이점
- [ ] Training benchmark와 Inference benchmark의 분리
- [ ] MLPerf submission 요구사항이 코드 구조에 미치는 영향

### 공통 논의 (6주 전체 회고)

- [ ] 6주간 읽은 내용을 종합했을 때, HSTU의 핵심 기여는?
- [ ] ML 관점에서 가장 인상적이었던 설계 결정은?
- [ ] MLOps 관점에서 가장 인상적이었던 시스템 구성은?
- [ ] 우리 팀에 적용할 수 있는 것은?

### 추천 레포 참고 파일

- `generative_recommenders/dlrm_v3/` — DLRMv3 디렉토리
- `README.md` — DLRMv3, MLPerf 관련 설명

## 참고 자료

- [기존 학습서 15장](../../part3/ch15_dlrmv3_production.md) — DLRMv3 Production
- [기존 학습서 18장](../../part4/ch18_practical_application.md) — 실무 적용

## 발표 자료

> 발표 슬라이드, 노트 등을 이 폴더에 추가하세요.
