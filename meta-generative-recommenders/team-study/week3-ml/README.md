# 3주차 — ML 엔지니어 B

## 주제: 행동 시퀀스와 콘텐츠 신호를 모델 내부에서 어떻게 묶는가

---

## 직접 연결 논문

1. **Actions Speak Louder than Words** (ICML'24)
   - [arxiv.org/abs/2402.17152](https://arxiv.org/abs/2402.17152)
   - HSTU 내부 구조: action encoder, content encoder, sequence transducer

2. **Self-Attentive Sequential Recommendation (SASRec)** (ICDM'18)
   - [arxiv.org/abs/1808.09781](https://arxiv.org/abs/1808.09781)
   - README에서 공개 실험의 주요 baseline으로 명시. HSTU의 출발점

## 왜 이 논문인가

`modules/`에는 `action_encoder`, `content_encoder`, `preprocessors`, `stu`, `dynamic_stu`, `hstu_transducer` 같은 파일이 있다. HSTU를 "SASRec보다 조금 복잡한 모델"로 보는 게 아니라, **Meta가 sequence abstraction을 어떻게 더 넓혔는지**로 읽어야 한다.

## 발표 포인트

이 주차에서 설명할 것은 수식 전체가 아니다. 아래 5가지가 **어떻게 분리되어 있는지**만 잡아주면 충분하다.

### 핵심 모듈 분리

```
행동(action) ──┐
콘텐츠(content) ──┤
위치/문맥 ──┤──→ Sequence Transducer ──→ Multi-task Head
             │
시퀀스 구조 ──┘
```

### 다뤄야 할 질문

- [ ] Action encoder vs Content encoder의 역할 구분은?
- [ ] STU Layer에서 attention이 SASRec과 어떻게 다른가?
- [ ] Dynamic STU는 왜 필요한가?
- [ ] Multi-task head가 지원하는 task 유형은?
- [ ] SASRec → HSTU로의 핵심 확장 포인트는?

### 추천 레포 참고 파일

- `generative_recommenders/modules/action_encoder.py`
- `generative_recommenders/modules/content_encoder.py`
- `generative_recommenders/modules/preprocessors.py`
- `generative_recommenders/modules/stu.py`
- `generative_recommenders/modules/dynamic_stu.py`
- `generative_recommenders/modules/hstu_transducer.py`

## 참고 자료

- [기존 학습서 8장](../../part2/ch08_stu_layer.md) — STU Layer
- [기존 학습서 12장](../../part3/ch12_modules.md) — 모듈 상세

## 발표 자료

> 발표 슬라이드, 노트 등을 이 폴더에 추가하세요.
