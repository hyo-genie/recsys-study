# 9장. NCF & Deep Learning Models

---

## 9.1 NCF (Neural Collaborative Filtering)

```
Input: (user_id, item_id)
  → User Embedding + Item Embedding
  → GMF path: element-wise product
  → MLP path: concatenate → deep layers
  → NeuMF: combine GMF + MLP → prediction
```

## 9.2 알고리즘 비교표

| Model | Architecture | Framework | Key Innovation |
|-------|-------------|-----------|---------------|
| **NCF** | Embedding → MLP | TensorFlow | Neural interaction |
| **Wide&Deep** | Linear + DNN | PyTorch | Memorization + generalization |
| **LightGCN** | Graph Conv | TensorFlow | User-item graph structure |
| **VAE** | Encoder-Decoder | TF/Keras | Generative model |
| **xDeepFM** | CIN + DNN | TensorFlow | Feature cross |

> **HSTU 스터디 연결**: NCF의 "Embedding → Interaction → Prediction" 패턴은 HSTU의 DlrmHSTU와 동일한 구조. 차이점: NCF는 단일 시점 interaction, HSTU는 전체 시퀀스를 인코딩.

---

[← 8장](ch08_als_mf.md) | [목차](../README.md) | [10장 →](ch10_sequential.md)
