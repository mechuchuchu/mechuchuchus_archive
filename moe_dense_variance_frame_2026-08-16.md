# MoE vs Dense — Variance/Bias 통일 프레임

> 2026-08-16 사고실험 정리

---

## 0. 배경 실험 (SP interpolation, OLMoE-1B-7B vs Olmo-3-1025-7B)

Soft prompt 두 지점 사이 continuous interpolation (100 step)을 하면서 logit 궤적을 관찰:

- **Cumulative KL from initial logits**: 초반(~step 50)엔 MoE가 dense보다 빠르게 divergence 뜸. 근데 step ~65부터 dense가 역전해서 끝까지 앞서감.
- **Step-to-step KL**: MoE는 expert routing이 바뀌는 지점(거의 매 스텝)마다 spike. Dense는 100 step 내내 거의 0 — 완전히 smooth.
- **PCA trajectory**: Dense는 깔끔한 단일 arc, PC1 기준 순방향으로 계속 진행(~1500까지 누적 이동). MoE는 X마크(routing 변경) 뒤덮인 지그재그, 특히 self-crossing/loop 존재 — 국소적으로 진동하지만 net displacement는 작음(~250 근처에서 맴돎).

**해석**: MoE의 hard routing은 SP처럼 natural language manifold 제약이 없는 연속 입력 공간에서는 거의 매 스텝 routing boundary를 가로지름 → 국소적으로 격렬히 튀지만(high local variance), self-crossing 때문에 전역 탐색 거리는 오히려 dense보다 짧음. "Discontinuity = 더 풍부한 함수 표현"이라는 순진한 가설에 대한 반례.

---

## 1. Params 기준 vs FLOPs 기준

- Same total param 기준: dense가 항상 sparse(MoE)를 이김 — 이미 알려진 사실 (Switch Transformer, Clark et al. 2022 "Unified Scaling Laws for Routed LMs" 등). Sparse는 매 step 파라미터 서브셋만 gradient 받으니까 파라미터당 학습 효율이 구조적으로 낮음.
- Sparse의 존재 이유는 애초에 params 축이 아니라 **FLOPs 축**: same FLOPs 기준으로는 특정 scale 이상부터 sparse가 dense를 이김. Quail MoE 플랜(30B total/3B active)도 이 베팅.

---

## 2. 함수공간 차이: coarse-partition + local-fine vs global-fine

- **Dense**: 하나의 전역 함수가 전체 input space에 대해 극도로 매끄럽고 정교한 매핑 — capacity가 전부 하나의 연속 함수에 집중.
- **MoE**: 1차로 거칠게(coarse) input space를 routing으로 파티션 → 각 파티션 내부에서 expert가 2차로 정교하게(fine) 매핑. 정교함이 계층화됨 (coarse cut + local fine cut).
- 이 프레임에서 "함수공간이 다르다"는 수학적으로 참(MoE는 불연속함수)이지만, 실질적 이점 여부는 **input distribution이 자연스럽게 coarse cluster로 나뉘는가**에 달림.
  - Natural language: topic/domain별로 실제 클러스터링 → coarse-then-fine 유리
  - SP interpolation처럼 연속적이고 클러스터 구조 없는 입력 → coarse cut 자체가 방해(oscillation)로 작용 (오늘 관찰한 현상의 근본 원인)

---

## 3. Ensemble vs MoE: 완전히 다른 메커니즘

- **Ensemble** (같은 문제, 서로 다른 init/서브넷의 N-param 모델 k개, 전체 input space를 각자 다르게 근사): 다양성의 원천 = "같은 문제를 다른 방식으로 풂" → variance가 상쇄되며 거대모델(Nk param)의 보간에 근접. Overparam 영역에서 개별 모델이 이미 training point를 interpolate(bias≈0)하니까, ensemble이 줄이는 건 거의 순수 variance.
- **MoE** (input space를 routing으로 쪼갠 뒤 각 조각을 expert 하나가 전담): 다양성이 아니라 **분업**. 각 expert는 자기 담당 조각만 봄 — redundant coverage가 구조적으로 없음(애초에 routing이 그걸 없애려고 존재). 그래서 ensemble의 variance-reduction 효과가 발생하지 않음.
- 결론: Ensemble = "같은 문제, 다른 풀이 → 평균으로 정답에 수렴" (variance reduction). MoE = "다른 문제, 각자 풀이 → 합쳐서 커버" (capacity specialization). 처음부터 다른 축에서 이득을 얻는 구조라 같은 잣대(보간 근접성)로 비교하는 건 category error.

---

## 3.5 Joint training의 k-scaling: subnet collapse는 일어나는가

> Random-vector memorization toy (MLP, LBFGS)로 확인. Joint training이 이론적으로는 "capacity 없이 bias를 낮추는" 레버(§6 열린 질문 2번에 대한 답 후보)지만, 이게 실제로 작동하려면 k개 subnet이 서로 다른 residual을 나눠 맡는 division-of-labor가 실제로 유지돼야 함 — 만약 subnet들이 서로 수렴(collapse)해서 다 같은 함수를 배우면, k개를 합쳐봐야 사실상 1개 subnet과 다를 게 없어짐. 이걸 확인하는 실험.

### k=2, symmetry-breaking 여부에 따른 collapse

동일 초기화(noise=0, 완전 대칭)에서 시작한 두 subnet은 loss가 bagging 수준(~0.26~0.29)에서 멈춤 — cosine similarity가 처음부터 끝까지 정확히 1.0으로 고정되어 있어서, 대칭이 원천적으로 깨질 수가 없기 때문(결정론적 gradient라 두 subnet이 매 스텝 완전히 동일하게 움직임). 반면 noise가 아무리 작아도(1e-5 수준) 대칭이 깨지고, cosine similarity가 빠르게 -1 근처(anti-correlated, 즉 서로 반대 방향 residual을 담당)로 수렴하면서 loss가 joint 정상 수준(~0.003~0.04)까지 떨어짐.

**결론**: Collapse는 "완벽한 대칭 초기화"라는 measure-zero 조건에서만 일어나는 병적 edge case. 현실적인 분산학습 시나리오(각 volunteer node가 독립적으로 random init)에서는 이 조건이 사실상 발생하지 않으므로, collapse 자체는 실질적 리스크가 아님.

### k∈{2,5,10,50}로 확장 — 부분적 collapse는 생기는가

k가 커지면 "완전 붕괴는 없어도 일부 pair가 국소적으로 겹치는 부분적 redundancy"가 생길 가능성이 남아있어서, k를 50까지 올려 pairwise cosine 전수 조사:

| k | eff. rank / (k−1) | max pairwise cosine | collapsed pairs (margin 0.5) |
|---|---|---|---|
| 2  | 1.026 | −0.99 | 0 / 1 |
| 5  | 1.009 | −0.12 | 0 / 10 |
| 10 | 0.950 | +0.23 | 0 / 45 |
| 50 | 0.755 | +0.34 | 0 / 1225 |

(independent init / near-identical init 두 조건 모두 거의 동일한 패턴)

**해석**:
- **완전 collapse는 k=50, 1225개 pair를 다 봐도 단 한 건도 없음** — anti-correlation attractor(division-of-labor)가 k가 커져도 robust하게 유지된다는 위 결론이 k=50까지 그대로 확장됨.
- 다만 **effective rank ratio가 k와 함께 꾸준히 감소**(1.03 → 0.76)하고 **max pairwise cosine도 음수에서 양수로 이동**(−0.99 → +0.34) — 이건 "급격한 붕괴"는 아니지만 k가 커질수록 division-of-labor가 점점 덜 완벽해진다(일부 subnet 쌍이 서로 약하게 겹치기 시작한다)는 신호.

### §5(scaling sweep) 결과와의 연결

이 effective-rank 감소는 §5의 joint scaling exponent(α≈0.4, 선형(α=1)에는 못 미침)와 정합적으로 맞물림 — 만약 division-of-labor가 k에 비례해서 완벽하게 유지된다면 joint capacity가 k×H에 선형으로 늘어야 하는데(α→1에 가까워야 함), 실제로는 k가 커질수록 division-of-labor 효율이 조금씩 새면서(eff rank ratio 감소) 그만큼 capacity 증가폭도 sub-linear(α≈0.4)해지는 것으로 이해할 수 있음. 즉 두 독립적인 실험(§3.5의 pairwise cosine, §5의 scaling exponent)이 서로 다른 각도에서 같은 현상 — "joint training은 k가 커져도 파국적으로 실패하진 않지만, division-of-labor의 완벽도는 k에 따라 서서히 감소한다" — 을 가리키고 있음.

---

## 4. Bias-Variance 통일 프레임

Double descent curve (classical bias-variance tradeoff, overparam 영역에서 bias→0 & variance가 지배적) 기준:

- **Underparam**: bias 이슈로 큰 NN을 절대 못 넘음 (correlated bias는 ensemble 평균으로도 안 사라짐)
- **Overparam**: bias≈0, test error ≈ variance. 서로 decorrelate된 net들의 ensemble이 variance를 1/k로 줄임 → 이게 통함

**MoE = variance 크고 bias 작은 구조**:
- Low bias: 각 expert가 담당 조각 안에서 매우 날카롭게(거의 memorization급) fit 가능 — transformer KV cache의 explicit memory slot과 같은 계열 (routing = 사실상 explicit assignment). RWKV의 compressed fixed-size state(강제 일반화 압력, bias 높고 variance 낮음)와 정반대 극.
- High variance: 오늘 관측한 step-to-step KL spike가 이 local sensitivity를 직접 보여준 것.
- (주의) 고전적 bias-variance의 variance는 "재훈련 시 출력 변동성"이고, 오늘 관측한 건 "같은 weight에서 input 미세변화에 대한 output 변동성"(local sharpness) — 엄밀히는 다른 축이지만, 둘 다 같은 구조적 원인(hard partition/flexible patchwork)에서 나오는 증상으로 연결 지을 수 있음.

---

## 5. 급진적 확장: "학습 토큰 수 = ensemble 수"

> **주의**: dropout=bagging 근사, data augmentation=variance reduction 같은 개별 연결은 새 발견이 아니라 Goodfellow/Bengio/Courville *Deep Learning* 7장(Regularization)에서 이미 명시적으로 다뤄진 고전 프레임. 오늘 사고실험의 실제 기여는 (1) 이 프레임을 MoE에 적용해서 "MoE는 variance 큰 쪽" 결론을 낸 것, (2) 토큰 수 자체를 ensemble 축으로 보는 확장까지 밀어붙인 것 — 아래 표는 기존 지식을 한 축으로 재배치한 것이지 개별 항목이 새로 발견된 건 아님.

자연어 자체가 개별 관측(토큰) 단위로는 극도로 noisy/미분불가능한 discrete label. 토큰 수를 늘리는 게 구조적으로 ensemble member를 늘리는 것과 유사한 통계적 효과(variance reduction)를 낸다는 사고실험.

- Scaling law loss ~ N^(-α) 감소 형태가, 개별 관측 variance가 데이터 수 증가에 따라 줄어드는 통계적 평균화 현상과 함수형이 유사.
- Bias는 남고(모델 capacity 구조적 한계는 안 넘음) variance만 준다는 것도 일치.
- 단, 표준 ensemble의 explicit averaging과 SGD의 순차적 누적 학습(implicit averaging)은 메커니즘이 다름 — 그러나 "많은 noisy 개별 신호 → 하나의 stable 추정"이라는 결과는 동일.

### 이 프레임으로 재해석되는 기존 기법들

| 기법 | 어느 축의 variance를 줄이는가 |
|---|---|
| Data augmentation | Input space variance — 같은 label에 대해 여러 perturbed view로 boundary estimate |
| SAM | Weight space variance — flat minima = weight perturbation에 덜 민감 |
| Dropout | Weight-sharing ensemble (subnet들의 암묵적 평균, Srivastava 2014의 geometric mean 근사) |
| Curriculum | 데이터의 실효 variance 관리 — 쉬운 데이터(tolerance 넓음)로 먼저 bias를 낮추고, 이후 어려운 데이터(tolerance 좁음, 같은 model variance가 더 큰 loss variance로 증폭)로 fine-tuning |
| Residual connection | Depth에 따른 activation/gradient variance 누적 폭발 억제 (identity path가 기본, block은 residual만 학습) |
| LayerNorm/BatchNorm | Activation distribution variance explicit 억제 |

**잠정 결론**: 딥러닝 정규화/스케줄링 기법 상당수가 "어느 축(data/weight/architecture/depth)에서 variance를 누르는 장치"로 통일해서 볼 수 있음. 반면 bias를 낮추는 레버는 capacity(width/depth/param 수) 증가가 거의 유일한 걸로 보이는데, 이게 진짜 유일한 레버인지 아니면 capacity 증가 없이 bias만 의도적으로 낮추는 다른 메커니즘이 있는지는 미해결 질문.

---

## 5.5 실증: Sign net(STE) + dropout — mean adjacent KL 비교

> **성격**: 새 발견이 아니라 "실습/확인"에 가까움. 불연속함수라도 충분히 많은 (decorrelated) 샘플을 모으면 분산이 준다는 건 §4/§5의 통일 프레임에서 이미 예측되는 결과 — sign net처럼 극단적인 불연속 케이스에서도 그 법칙이 무효화되지 않는지 확인해본 것. 결과가 예측을 뒤집은 게 아니라 예측대로 나온 사례.

> Sign activation + STE로 학습한 net (dropout 유무만 다름). Interpolation 경로 위에서 mean logits의 step-to-step KL(KL(mean[i] || mean[i+1]))을 비교.

- **Dropout 없음**: max peak ≈ **0.21** (step ~68 근처). 그 외에도 ~0.095(step ~35), ~0.057(step ~52) 등 여러 지점에서 뾰족한 봉우리가 흩어져 나타남.
- **Dropout 있음**: max peak ≈ **0.083** (step ~58 근처) — **약 2.5배 감소**. 나머지 구간도 대부분 0.01 이하로 baseline 자체가 훨씬 조용함.

**해석**: 개별 seed(trajectory)들의 국소적 KL spike를 평균 냈을 때, dropout이 있으면 그 상쇄가 더 강하게 일어나 최종 mean 신호의 진폭이 확연히 줄어듦. §5의 "dropout = weight-sharing sub-ensemble" 주장이 예측하는 그림과 일치 — 불연속(sign net) 자체는 여전히 존재하지만, 그 불연속이 여러 subnet에 걸쳐 결맞지 않게(decorrelated) 흩어져 있을 때 dropout/ensemble 평균이 진폭을 눌러준다는 것.

(참고: 이 결과는 실험 재검증을 거친 값이며, 이전 버전에서 있었던 "seed별 peak 위치가 한 지점으로 수렴한다"는 서술은 과장이었음 — 실제로는 위치보다 진폭(amplitude) 감소가 더 명확한 효과.)

### Per-seed 분해 — 효과가 두 층위로 나뉜다

Mean만 보면 "seed averaging이 다 한 일"처럼 보이지만, 평균화 이전의 개별 seed(3개) 그래프를 보면 그렇지 않음:

- **Sign_ste 개별 seed 최대치**: seed0 ≈ 0.385, seed1 ≈ 0.36, seed2 ≈ 0.52 — 세 seed 모두 개별적으로 이미 0.3~0.5대의 큰 peak를 가짐(위치는 step 30/40/50/65/70 등으로 서로 다름).
- **Dropout 개별 seed 최대치**: seed0 ≈ 0.074, seed1 ≈ 0.082, seed2 ≈ 0.137 — 개별 seed 레벨에서부터 이미 3~4배 낮음.

**즉 mean에서 관측된 ~2.5배 감소는 두 효과가 중첩된 결과**:
1. Dropout이 개별 forward pass 자체를 이미 안정화(implicit sub-ensemble 근사로 개별 net의 variance를 낮춤)
2. 거기에 seed 간 averaging이 한 번 더 얹혀서 mean 단계에서 추가로 상쇄

개별 seed 단계를 먼저 안 봤으면 "seed averaging만으로 효과가 생겼다"고 오독할 뻔했음 — dropout은 averaging 이전, 개별 net 레벨에서부터 이미 variance를 줄이고 있다는 게 이 분해로 명확해짐.

---

## 6. 열린 질문 (다음에 다룰 것들)

1. **MoE scale 의존성**: OLMoE-1B-7B(active 1B)에서 관찰된 "국소 진동으로 인한 net exploration 제한"이 더 큰 active param(7B급 등)에서도 동일하게 나타나는지, 아니면 scale 커질수록 routing이 안정화되면서 self-crossing이 줄어드는지.
2. **Bias를 의도적으로 낮추는 non-capacity 레버가 존재하는가?** (capacity ↑ = bias ↓가 유일한 축인지, 아니면 architecture/inductive bias 설계로 capacity 고정한 채 bias만 낮출 수 있는지)
3. Curriculum 가설의 직접 검증: 순서를 바꿔가며 초반 gradient noise를 실제로 측정 — "어려운 데이터부터 시작하면 초기 high-variance state에서 gradient가 더 destructive"라는 예측이 맞는지.
