# CIFAR-10: Sign-STE vs Dropout — Interpolation Trajectory 실험

## 1. 목적

같은 CNN 아키텍처에서 **Sign activation(STE)** 을 쓴 모델과 **Dropout**을 쓴 모델이,
두 클래스(cat → dog) 이미지 사이를 선형 보간(interpolation)했을 때
logit 공간에서 얼마나 다르게/비슷하게 움직이는지를 비교.

seed 간 variance도 함께 봐서, 두 정규화(regularization) 방식이
**결정 경계 근방에서의 안정성/매끄러움**에 어떤 영향을 주는지 확인하는 실험.

---

## 2. 모델 구조 (공통 backbone)

- VGG-style CNN: `64 → 128 → 256 → 512` conv 채널, 각 stage 뒤 `MaxPool2d`
- 마지막 `AdaptiveAvgPool2d(1,1)` → `Linear(512, 10)`
- **Sign activation은 두 모델 모두 정확히 한 번**, 128채널 conv 직후 (2번째 MaxPool 직전)에 위치
  - `SignSTE`: forward는 `sign(x)`, backward는 straight-through (gradient 그대로 통과)

| | SignCNN | DropoutCNN |
|---|---|---|
| Sign 위치 | 동일 | 동일 |
| 추가 regularization | 없음 | `Dropout2d(0.3)` (거의 매 conv block 뒤) + 마지막 FC 앞 `Dropout(0.3)` |

---

## 3. 데이터 & 학습 설정

- Dataset: HuggingFace `uoft-cs/cifar10` (torchvision CIFAR10과 label 순서 동일)
- Train transform: RandomCrop(32, pad=4) + RandomHorizontalFlip + Normalize
- Eval/interpolation transform: augmentation 없이 Normalize만 (재현성 확보)
- Optimizer: Adam, lr=1e-3, weight_decay=5e-4
- Scheduler: CosineAnnealingLR
- Batch size 256, Epoch 100
- **Seed 3개 (0, 1, 2) × 모델 2종(Sign / Dropout) = 총 6개 모델 학습**

---

## 4. Interpolation 실험

1. train set에서 `cat` 클래스 1장(x0), `dog` 클래스 1장(x1) 선택
2. `x0 → x1`을 alpha=0~1 사이 **100 point 선형보간**
   - `interp = (1-α)·x0 + α·x1`
3. 학습된 6개 모델 각각에 100개 보간 이미지를 통과시켜 logit 추출
4. 결과 저장: `[100, 6, 10]` (interp step × model × class) → `.pt` 파일

모델 순서: `sign_seed0, dropout_seed0, sign_seed1, dropout_seed1, sign_seed2, dropout_seed2`

---

## 5. 분석 스크립트 (2번째 코드)

Sign 계열 3개 seed, Dropout 계열 3개 seed로 분리 후 각각에 대해 7종 분석/플롯 생성 (총 14 PNG):

| # | 분석 | 설명 |
|---|---|---|
| 1 | 3 seeds adjacent KL | `KL(logit[i] \|\| logit[i+1])`, seed별 |
| 2 | mean adjacent KL | 3 seed 평균 logit 기준 adjacent KL |
| 3 | 4 trajectories adjacent KL | seed0/1/2 + mean 겹쳐 비교 |
| 4 | 3 seeds from-start KL | `KL(logit[0] \|\| logit[i])`, seed별 |
| 5 | mean from-start KL | 평균 logit 기준 from-start KL |
| 6 | 4 trajectories from-start KL | seed0/1/2 + mean 겹쳐 비교 |
| 7 | PCA trajectory | 4 trajectory(3 seed+mean)를 합쳐 fit한 2D PCA 상에서 궤적 시각화 (시작=○, 끝=✕) |

- **Adjacent KL**: 보간 스텝 사이의 "국소적 변화량" → 급격한 클래스 전환(decision boundary 근접) 지점을 포착
- **From-start KL**: 시작점(cat) 대비 누적 변화량 → 전체 궤적의 단조성/비단조성 확인
- 최종 결과는 `analysis_results.pt`에 sign/dropout 각각의 7개 텐서 + PCA explained variance로 저장

---

## 6. 기대되는 관찰 포인트

- Sign(STE) 모델은 activation이 이산적(±1)이라 **logit이 계단식으로 급변**할 가능성 → adjacent KL에 뾰족한 peak
- Dropout 모델은 상대적으로 **매끄러운 전이** 예상 (다만 inference 시 dropout은 꺼져 있으므로 차이는 순수히 학습된 weight 분포 차이에서 옴)
- seed 간 variance가 크면 → 해당 정규화 방식이 결정 경계 위치를 불안정하게 만든다는 신호
- PCA 궤적에서 seed 3개가 유사한 경로를 그리는지, mean과의 이격이 얼마나 되는지로 "일관성" 판단 가능
