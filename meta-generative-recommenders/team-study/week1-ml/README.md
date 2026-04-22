# 1주차 — ML 엔지니어 A

## 주제: Meta는 왜 추천을 generative / sequential 문제로 다시 보나

---

## 직접 연결 논문

1. **Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations** (ICML'24)
   - [arxiv.org/abs/2402.17152](https://arxiv.org/abs/2402.17152)
   - 이 레포의 메인 논문. HSTU 아키텍처를 제안하며, 추천을 generative transduction 문제로 재정의

2. **Deep Learning Recommendation Model for Personalization and Recommendation Systems (DLRM)**
   - [arxiv.org/abs/1906.00091](https://arxiv.org/abs/1906.00091)
   - Meta 추천 시스템의 전통적 설계 기준선. "무엇을 버리고 무엇을 확장했는지"를 설명할 때 가장 직접적

## 왜 이 논문인가

1주차는 레포 전체의 문제의식을 여는 회차. README도 이 레포를 *Actions Speak Louder than Words*의 구현으로 소개하면서, 고전적 DLRM 패러다임을 generative recommender로 재구성한다고 설명한다.

## 발표 포인트

이 주차에서는 **모델 구현보다 문제 재정의**를 설명하면 된다.

> "Meta는 추천을 fixed ranking 블록이 아니라, 긴 사용자 행동 시퀀스를 다루는 **generative transduction** 문제로 보고 있다"

### 다뤄야 할 질문

- [ ] DLRM의 구조적 한계는 무엇이었나? (sparse + dense → interaction → prediction)
- [ ] "Generative Recommendation"이란 정확히 무엇을 의미하나?
- [ ] Sequential transducer가 기존 ranking model과 어떻게 다른가?
- [ ] Trillion-parameter scale에서 DLRM이 왜 한계에 부딪히나?

### 추천 레포 참고 파일

- `README.md` — 레포 소개, DLRM과의 관계
- `generative_recommenders/` — 전체 패키지 구조 개요

## 참고 자료

- [기존 학습서 Part 1](../../part1/) — 기초 지식 (선형대수 ~ 추천시스템)
- [기존 학습서 7장](../../part2/ch07_paper_overview.md) — 논문 개요

## 발표 자료

> 발표 슬라이드, 노트 등을 이 폴더에 추가하세요.
