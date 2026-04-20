# 16장. 환경 구축과 실행

---

## 16.1 시스템 요구사항

| 항목 | 최소 | 권장 |
|------|------|------|
| GPU | 24GB HBM (A100) | H100 80GB |
| CUDA | 12.1 | 12.4 |
| Python | 3.10 | 3.10 |
| OS | Ubuntu 20.04 | Ubuntu 22.04 |

## 16.2 설치

```bash
# 1. 레포 클론
git clone https://github.com/meta-recsys/generative-recommenders.git
cd generative-recommenders

# 2. 의존성 설치
pip3 install -r requirements.txt
# torch>=2.6.0, fbgemm_gpu>=1.1.0, torchrec>=1.1.0, gin_config>=0.5.0

# 3. 데이터 전처리
mkdir -p tmp/
python3 preprocess_public_data.py
```

## 16.3 Research 실험 재현

```bash
# MovieLens-1M + HSTU-large (single GPU)
CUDA_VISIBLE_DEVICES=0 python3 main.py \
  --gin_config_file=configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin \
  --master_port=12345

# TensorBoard 모니터링
tensorboard --logdir exps/ml-1m-l200/ --port 24001
```

### 예상 결과 (ML-1M, HSTU-large)

| Metric | Expected | Epochs |
|--------|----------|--------|
| HR@10 | ~0.33 | ~80 |
| NDCG@10 | ~0.185 | ~80 |
| Training time | ~30min | 101 (single A100) |

## 16.4 DLRMv3 실행

```bash
# 4-GPU debug 학습
LOCAL_WORLD_SIZE=4 WORLD_SIZE=4 python3 \
  generative_recommenders/dlrm_v3/train/train_ranker.py \
  --dataset debug --mode train

# 4-GPU 추론 벤치마크
LOCAL_WORLD_SIZE=4 WORLD_SIZE=4 python3 \
  generative_recommenders/dlrm_v3/inference/main.py \
  --dataset debug
```

---

[← 15장](../part3/ch15_dlrmv3_production.md) | [목차](../../README.md) | [17장 →](ch17_hyperparameters.md)
