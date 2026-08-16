# Gradient Matching Dataset Distillation — 실험 기록 (2026-08-12)

## 0. 배경

`universal_distilled_synthetic_data_exp_idea.md`에서 출발한 질문: 서로 다른
architecture/weight configuration이 공유하는 "architecture-independent한 학습
정보"가 존재하는가? Gradient matching 기반 dataset distillation을 MNIST 스케일
smoke test로 검증하면서 나온 발견들 정리.

---

## 1. Related work — 바퀴 재발명 체크

| 계보 | 핵심 아이디어 | 우리 실험과의 관계 |
|---|---|---|
| **DD** (Wang et al. 2018) | Performance matching, bi-level optimization, 전체 궤적 unroll | 원조. Section 3.3에서 linear case 하한 $M \ge D$ 증명 |
| **DC/gradient matching** (Zhao 2021) | 1-step gradient cosine+MSE matching | 우리가 실제로 쓰는 방식. DD의 unroll 비용을 회피한 근사 |
| **MTT** (Cazenavette 2022) | 실제 학습된 expert trajectory buffer에서 matching | "trained weight 써야 한다"는 문제의 표준 해법. LLM 스케일에선 연산량 문제로 비실용적 판단 (아래 §5) |
| **TESLA** (Cui 2023) | MTT의 backprop 메모리를 constant로 | 메모리만 해결, FLOPs는 그대로 → 연산량 병목엔 무효 |
| **KIP** (Nguyen 2021) | Infinite-width NTK, closed-form matching | 이론적 upper bound 참고용. O(\|S\|²) 비용으로 실전 불가 |
| **GLaD / HaBa / MetaDD** | Latent space parameterization, architecture-invariant feature 분리 | Cross-architecture 문제의 기존 접근. Pixel/gradient space 직접 최적화가 특정 architecture에 overfit되는 문제 지적 |
| **Gist tokens** (Mu 2023) | Task loss로 직접 backprop되는 compression | "autoencoder로 압축 후 LLM이 알아서 학습" 아이디어의 실제 성립 조건 — reconstruction loss 단독으로는 불충분, task-driven이어야 함 |

---

## 2. 실험 파이프라인 (MNIST smoke test)

- Model: 2-layer MLP (784→128→10), fresh kaiming init
- Region: 완전 독립 random init 16개 (train) + 16개 (unseen, held-out)
- Distillation: `synth_x, synth_y_logits`를 gradient cosine+MSE loss로 최적화
- Eval: distilled set으로 fresh random-init probe를 100 step 학습 → MNIST test accuracy

## 3. 핵심 발견

### 3.1 M(synthetic 개수) vs D(=784, pixel dimension) — DD의 이론적 하한 검증

DD 논문 §3.3: linear/quadratic case에서 arbitrary $\theta_0$에 완전히 일반화하려면
$M \ge D$가 필요하다는 하한 증명. Nonlinear MLP + gradient matching(1-step 근사)
세팅에서 실측:

| M | unseen gradient cosine | downstream accuracy (SGD eval) | downstream accuracy (Adam eval) |
|---|---|---|---|
| 10 | 0.166 | — | — |
| 50 | 0.968 | 76.6~79.2% | 76.6% |
| 150 | — | 81.9% | — |
| 300 | — | 85.5% | — |
| 784 (=D) | 0.985 | 86.8~88.1% | 88.1% |
| 1500 | — | 88.9% | — |

**결론**: $M=D$는 급격한 threshold가 아니라, capacity를 늘릴수록 완만하게 개선되는
곡선의 한 지점일 뿐. DD 논문의 hard bound는 "모든 $\theta_0$에 대한 exact-zero-residual
해"라는 훨씬 강한 조건에서 나온 것이라, 우리의 (discrete 16-region, cross-entropy,
gradient-only matching) 세팅에는 threshold 형태로 그대로 적용되지 않음.

참고: 원 논문(Wang 2018) MNIST random-init 결과 79.50%±8.08% (M=100) — 우리 M=50
gradient matching이 76.6~79.2%로 거의 동일 성능을 절반 sample, 훨씬 낮은 variance로 달성.

### 3.2 Gradient cosine의 "head-tail 비대칭" — 제일 중요한 발견

Full-vector cosine similarity는 magnitude가 큰 성분("head")에 의해 지배되는
metric이라, 실제 distillation 품질 변화를 과소평가함.

Real gradient를 `|gradient|` 기준 head(상위 X%)/tail(나머지)로 분리해서 측정:

| Fraction | Head cosine (M=50→784) | Tail cosine (M=50→784) |
|---|---|---|
| Top 1% | 0.9966 → 0.9951 (거의 불변) | 0.9665 → 0.9827 |
| Top 50% (tail=하위 50%) | 0.9800 → 0.9892 | **0.7388 → 0.8307** |

Sign agreement(magnitude 무시, 부호만 비교, random baseline=0.5)로 교차검증:

| Fraction | Head sign | Tail sign |
|---|---|---|
| Top 1% | 1.0000 (M 무관) | 0.9074 → 0.9311 |
| Top 50% (tail) | 0.9872 → 0.9971 | **0.8294 → 0.8666** |

**결론**:
- Head는 M=50에서 이미 포화(sign agreement=1.0), M을 늘려도 거의 안 움직임.
- Tail은 M=50에서도 완전 랜덤(0.5)이 아니라 0.83~0.90대로 의미 있게 방향이
  맞아있고, M을 늘리면 **cosine과 sign agreement 둘 다 뚜렷하게 개선**됨.
- Downstream accuracy 개선(M=50→784, +7~11%p)의 대부분은 tail 쪽 정보 개선에서
  온 것으로 추정됨. Tail sign agreement가 랜덤이 아니라는 게 magnitude 착시가
  아니라 진짜 학습된 방향이라는 근거.
- **실용적 시사점**: distillation 품질을 full-vector cosine 하나로 평가하면
  오도될 수 있음. Tail(magnitude-small) subset의 cosine/sign agreement가 훨씬
  민감한 proxy metric일 가능성.

### 3.3 Eval optimizer 민감도 (SGD vs Adam)

| M | SGD eval | Adam eval |
|---|---|---|
| 50 | 76.6~79.2% | 76.6% |
| 784 | 86.8~88.1% | 88.1% |
| **Gap** | 7.6~8.9%p | **11.5%p** |

Adam의 $m/\sqrt{v}$ normalization이 tail(small-magnitude) 방향을 head와 비슷한
스케일로 끌어올리는 연산이라, tail 정보가 실제로 개선된 M=784에서 그 차이를
SGD보다 더 크게 드러냄 — §3.2의 head-tail 발견과 일관됨.

**주의**: eval optimizer 선택에 따라 distilled set 간 순위가 바뀔 수 있는
confound. DD 원 논문도 이를 인지해 learning rate/epoch 조합 전체에 대해
grid search 후 최고 성능을 report하는 방식으로 통제함.

### 3.4 Cross-architecture transfer (MLP → CNN)

MLP로 distill된 synthetic set을 CNN(LeNet 스타일)에 그대로 학습:

| M | MLP (source arch) | SmallCNN (unseen arch) |
|---|---|---|
| 10 | 71.3% | 69.8% |
| 50 | 74.9% | 73.6% |
| 150 | 81.9% | 83.1% |
| 300 | 85.5% | 88.7% |
| 784 | 88.1% | 90.8% |
| 1500 | 88.9% | 91.8% |

**M≥150부터 CNN이 오히려 MLP를 앞섬.** GLaD 등이 보고한 "pixel-space distillation은
unseen architecture에서 성능이 크게 떨어진다"는 패턴이 여기선 관측되지 않음.

**미해결 confound**: CNN이 MNIST에 대해 원래 MLP보다 좋은 inductive bias를 가짐.
지금 결과가 "정보가 architecture-invariant하게 전달됐다"는 증거인지, 그냥
"CNN이 어떤 이미지 데이터를 줘도 잘 배운다"는 것뿐인지 미구분 상태.
→ **다음 단계**: class-mean/k-means/random-real 등 gradient matching을 거치지
않은 baseline에도 동일한 CNN vs MLP 비교를 돌려서 confound 제거 필요.

또한 MLP↔CNN은 둘 다 static feedforward, 같은 pixel-grid 입력을 받는 얕은
아키텍처 차이라 "cross-architecture"치고는 약한 테스트. RWKV(state recurrence)
vs Transformer(attention) 수준의 이질적 architecture 간 전이는 미검증.

### 3.5 (참고) Local robustness vs basin diversity — 두 가지 다른 질문

세션 초반에 두 가지 다른 실험을 혼동했다가 분리함:

- **Local robustness**: 하나의 (미학습) anchor 근방에 gaussian perturbation(std
  ball)을 준 region들. `num_regions`와 perturbation `std`가 confound돼 있어서
  "region 개수 증가 효과"와 "perturbation 반경 증가 효과"가 분리 안 됨. Cosine
  0.70~0.78 수준.
- **Basin diversity**: 완전히 독립적인 random init 16개 (본 문서의 §2~3.4가 이 버전).
  Cosine 0.95~0.98 수준으로 더 높게 나옴 — 그러나 둘 다 **학습된(converged) weight가
  아니라 fresh init**이라는 공통 한계 있음. DD/DC/DM/MTT 문헌 대비 원래 가설
  ("서로 다른 **학습된** solution이 공유하는 정보")은 아직 검증 안 된 상태.

---

## 4. 아직 안 풀린 질문 (다음 스텝 후보)

1. **Trained checkpoint 버전**: 지금까지 전부 fresh random init 기반. MTT 스타일로
   실제 MNIST에 수렴한 checkpoint들 사이의 gradient matching으로 바꿔서 같은
   head-tail/M-sweep 분석 재현 — 원래 가설("학습된 solution들이 공유하는 정보")의
   진검승부.
2. **Cross-arch baseline 통제**: §3.4의 confound 제거 (class-mean/k-means/random-real
   대비 CNN 우위 폭 측정).
3. **Cross-arch 이질성 스캔**: MLP → CNN → 작은 attention → RWKV 순으로 architecture
   차이를 점진적으로 늘려가며 transfer가 어디서 무너지는지 확인.
4. **Tail cosine을 실제 evaluation metric으로 도입**: M-sweep 데이터에 대해
   full cosine vs tail cosine이 downstream accuracy를 각각 얼마나 잘 예측하는지
   상관계수 비교.
5. **DPO indifference test**: 기존 vision DD 문헌에 없는 evaluation 축. Synthetic
   dataset과 real dataset을 preference optimization으로 경쟁시켜 구별 가능성 측정 —
   지금까지 찾은 문헌 중 novel한 조합으로 보임.
6. **Adversarial min-max region coverage**: $\min_S \max_{\theta \in C} d(G_\theta(S), G_\theta(D))$.
   AdvFunMatch의 "worst-case matching이 전체 norm ball을 간접적으로 커버한다"는
   논리를 weight-space adversary에 적용하는 아이디어. $C$를 trained checkpoint들의
   convex hull로 잡으면 §4.1과 결합 가능.

## 5. 실전 적용 관련 결론 (RWKV-7 scaling sweep 용도)

- **Data recipe 자체를 탐색하는 연구엔 distillation이 원천적으로 안 맞음** — expert
  trajectory/distilled set이 특정 $D$에 묶여있어서 recipe가 바뀌면 재사용 불가.
  더 나쁘게는, **architecture나 scale이 바뀌어도 gradient/parameter space 자체가
  달라져서 재사용이 깨짐** (§3.4 확인 전까지는 미검증 영역이었음).
- Data recipe/architecture/scale을 고정하고 다른 하나만 바꾸는 screening
  용도라면 이론적으로 경제성 있음.
- 지금 당장 RWKV-7 sweep의 실전 compute 절감 목적엔 gradient matching보다
  **DSIR/D4(SemDeDup) 같은 real-data selection 계열**이 훨씬 가볍고 검증된
  대안. LESS(gradient-influence 기반 selection)가 개념적으로는 gradient
  matching과 제일 가까운 사촌이지만 gradient store 비용이 큼.
- MTT류 trajectory matching은 연산량(expert trajectory 생성 = 사실상 N번의
  전체 학습) 자체가 naive training보다 항상 크므로, 재사용 축이 여러 번
  확보되지 않는 한 LLM 스케일에서 비경제적.

---

*작성: 2026-08-12, MNIST MLP smoke test 기반. 코드: `gradient_matching_M_vs_D.py`,
`cross_arch_transfer_test.py`, `head_tail_gradient_analysis.py`*
