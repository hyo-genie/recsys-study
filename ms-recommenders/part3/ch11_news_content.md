# 11장. News & Content-Based Models

---

## 11.1 News Recommendation (MIND Dataset)

| Model | Architecture | Key Feature |
|-------|-------------|-------------|
| **NRMS** | Multi-Head Self-Attention | 뉴스+유저 모두 attention |
| **LSTUR** | LSTM + GRU | Long/Short-term 유저 표현 |
| **NPA** | Personalized Attention | 유저별 다른 attention weight |
| **NAML** | Multi-View | 제목+본문+카테고리 결합 |
| **DKN** | Knowledge Graph + CNN | Entity embedding 활용 |

## 11.2 Content-Based

| Model | Approach | Use Case |
|-------|---------|----------|
| **TF-IDF** | Text similarity | 텍스트 기반 유사 아이템 |
| **LightGBM** | Feature ranking | CTR 예측, feature importance |

> **실무 적용**: 콘텐츠피드 = 뉴스/콘텐츠 추천. NRMS의 Multi-Head Attention 패턴을 참고하되, HSTU의 시퀀셜 인코딩이 더 강력. 두 접근을 결합하는 것도 가능.

---

[← 10장](ch10_sequential.md) | [목차](../README.md) | [12장 →](../part4/ch12_datasets.md)
