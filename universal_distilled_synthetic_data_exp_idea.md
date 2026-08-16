Universal Vector Dataset Distillation via Cross-Model Gradient Matching

1. 연구 아이디어

일반적인 Dataset Distillation은 특정 모델이나 특정 training trajectory를 대상으로 원본 dataset을 작은 synthetic dataset으로 압축한다.

본 연구에서는 조금 다른 질문을 던진다.

«서로 다른 neural network가 동일한 데이터를 학습할 때 공통적으로 필요로 하는 정보만을 매우 적은 수의 dense vector에 압축할 수 있는가?»

원본 dataset을 D, distilled dataset을 S라고 하면 목표는

$$
|S| \ll |D|
$$

이면서 여러 모델 f_\theta에 대해

$$
G_\theta(S) \approx G_\theta(D)
$$

를 만족시키는 것이다.

여기서

$$
G_\theta(D)

\sum_{(x,y)\in D}
\nabla_\theta L_\theta(x,y)
$$

는 dataset 전체가 모델에 주는 training gradient이다.

궁극적으로는 특정 모델에만 유효한 distilled dataset이 아니라,

$$
\boxed{
\text{architecture-independent training information}
}
$$

을 찾는 것을 목표로 한다.

---

2. 핵심 관찰: 50k vocabulary는 실제로 매우 높은 정보 밀도를 가질 수 있다

Vocabulary가 V=50,000이고 원본 sequence가 1,024 token이라면, 각 token은 one-hot vector로

$$
x_t \in \mathbb{R}^{50,000}
$$

로 표현할 수 있다.

Target 역시

$$
y_t \in \mathbb{R}^{50,000}
$$

이다.

원본 데이터가 sequence 10개라면 총 prediction example 수는

$$
10\times1024=10,240
$$

이다.

즉 원본은 개념적으로

$$
10,240
$$

개의 sparse input/target pair로 표현된다.

그러나 distilled dataset에서는 one-hot 제약을 제거한다.

$$
\tilde{x}_t \in \mathbb{R}^{50,000}
$$

$$
\tilde{y}_t \in \Delta^{49,999}
$$

여기서 \Delta^{49,999}는 50,000차원 probability simplex이다.

따라서 하나의 synthetic example이 여러 token의 정보를 동시에 포함할 수 있다.

목표는 예를 들어

$$
10,240 \rightarrow 1,024
$$

의 10× sample compression이다.

중요한 것은 byte-level compression이 아니라 training sample cardinality compression이다.

---

3. Dense synthetic token

일반적인 token i는 one-hot vector

$$
x=e_i
$$

이다.

Embedding matrix를 E라고 하면

$$
E^\top e_i = E_i
$$

가 되어 해당 token의 embedding을 얻는다.

반면 dense synthetic input

$$
\tilde{x}
$$

를 허용하면

$$
E^\top \tilde{x}

\sum_{i=1}^{50,000}
\tilde{x}_i E_i
$$

가 된다.

즉 synthetic input은 실제 vocabulary token이 아니라 embedding space의 연속적인 위치가 될 수 있다.

Target 역시 hard label

$$
y=e_j
$$

대신 soft target

$$
\tilde{y}\in\Delta^{49,999}
$$

를 사용할 수 있다.

Cross-entropy는

$$
L

-\sum_i \tilde{y}_i\log p_i
$$

이며 logit에 대한 gradient는

$$
\frac{\partial L}{\partial z}

p-\tilde{y}
$$

이다.

따라서 synthetic target 자체가 training signal을 상당히 직접적으로 조절할 수 있다.

---

4. Gradient Matching

원본 dataset을

$$
D={(x_i,y_i)}_{i=1}^{N}
$$

이라고 하자.

모델 f_\theta가 dataset에서 받는 전체 gradient를

$$
G_\theta(D)

\sum_{i=1}^{N}
\nabla_\theta L_\theta(x_i,y_i)
$$

로 정의한다.

Distilled dataset S에 대해서는

$$
G_\theta(S)

\sum_{i=1}^{M}
\nabla_\theta L_\theta(\tilde{x}_i,\tilde{y}_i)
$$

이다.

기본적인 목표는

$$
G_\theta(S)
\approx
G_\theta(D)
$$

이다.

예를 들어 objective는

$$
\mathcal{L}_{\mathrm{GM}}

\left|
G_\theta(S)-G_\theta(D)
\right|^2
$$

로 둘 수 있다.

---

5. 단일 모델에서 여러 모델로

한 모델에 대해서만 distillation하면 synthetic data가 특정 weight configuration에 overfit될 가능성이 있다.

이를 방지하기 위해 여러 모델을 동시에 사용한다.

모델 집합을

$$
\mathcal{F}

{f_{\theta_1},f_{\theta_2},\ldots,f_{\theta_K}}
$$

라고 하면

$$
\mathcal{L}_{\mathrm{multi}}

\sum_{k=1}^{K}
\left|
G_{\theta_k}(S)

G_{\theta_k}(D)
\right|^2
$$

를 최소화한다.

중요한 점은 서로 다른 모델의 gradient를 직접 동일하게 만드는 것이 아니다.

각 모델의 parameter space가 다르더라도 각각에 대해

$$
G_{\theta_k}(S)
\approx
G_{\theta_k}(D)
$$

가 성립하면 된다.

---

6. 첫 번째 핵심 실험: Same Architecture, Different Weights

가장 먼저 검증할 가설은 architecture diversity가 아니라 weight diversity만으로도 distilled data가 특정 model instance를 넘어설 수 있는가이다.

동일한 architecture를 갖는 두 모델을

$$
f_{\theta_1},
\qquad
f_{\theta_2}
$$

라고 하자.

architecture는 완전히 동일하지만

$$
\theta_1\neq\theta_2
$$

이다.

Distillation은

$$
\mathcal{L}

\left|
G_{\theta_1}(S)-G_{\theta_1}(D)
\right|^2
+
\left|
G_{\theta_2}(S)-G_{\theta_2}(D)
\right|^2
$$

를 최소화한다.

이 실험의 의미는 다음과 같다.

«하나의 weight configuration에만 존재하는 shortcut을 사용하지 않고, 같은 architecture가 공유하는 데이터의 학습 정보를 압축할 수 있는가?»

---

7. Held-out Weight Experiment

더 강한 검증을 위해 distillation에 사용하지 않은 weight를 별도로 둔다.

예를 들어

$$
\theta_1,\theta_2,\theta_3,\theta_4
$$

를 distillation에 사용하고,

$$
\theta_5,\theta_6,\theta_7,\theta_8
$$

를 held-out model로 남긴다.

Distillation은

$$
\mathcal{F}_{\mathrm{train}}

{f_{\theta_1},\ldots,f_{\theta_4}}
$$

에서 수행한다.

평가는

$$
\mathcal{F}_{\mathrm{test}}

{f_{\theta_5},\ldots,f_{\theta_8}}
$$

에서 수행한다.

만약 held-out weight에서도

$$
G_{\theta_k}(S)
\approx
G_{\theta_k}(D)
$$

가 나타난다면, S가 특정 weight에 overfit된 것이 아니라 architecture가 공유하는 학습 정보를 포착했다는 강한 증거가 된다.

---

8. DPO를 이용한 더 강한 검증

Gradient matching은 중간 목표일 뿐이다.

최종적으로 더 강한 질문은 다음이다.

«원본 D와 distilled dataset S를 preference optimization에서 서로 경쟁시키면, 모델이 둘 사이의 차이를 학습할 수 있는가?»

DPO에서 D를 chosen, S를 rejected로 둔다.

$$
y_w=D
$$

$$
y_l=S
$$

그러면 DPO는 개념적으로

$$
D>S
$$

라는 preference를 학습하도록 유도한다.

그러나 S가 D의 학습 정보를 충분히 보존했다면, 이 preference에는 의미 있는 정보가 거의 없어야 한다.

따라서 이상적인 결과는

$$
\boxed{
\mathrm{DPO}(D>S)
\approx
\mathrm{no\ learning}
}
$$

이다.

---

9. 반드시 양방향으로 테스트한다

한 방향만 실패하는 것은 충분하지 않다.

두 실험을 모두 수행한다.

Experiment A

$$
D_{\mathrm{chosen}},
\quad
S_{\mathrm{rejected}}
$$

Experiment B

$$
S_{\mathrm{chosen}},
\quad
D_{\mathrm{rejected}}
$$

둘 다 유의미한 preference learning을 만들지 못한다면 훨씬 강한 결과다.

즉,

$$
D\ntriangleright S
$$

뿐 아니라

$$
S\ntriangleright D
$$

도 관찰되어야 한다.

이 경우 D와 S 사이의 preference-discriminative information이 매우 작다는 증거가 된다.

---

10. DPO Failure를 여러 weight에서 평가

가장 중요한 실험 중 하나는 다음과 같다.

Model| Used for Distillation| DPO D>S| DPO S>D
\theta_1| Yes| ?| ?
\theta_2| Yes| ?| ?
\theta_3| Yes| ?| ?
\theta_4| Yes| ?| ?
\theta_5| No| ?| ?
\theta_6| No| ?| ?
\theta_7| No| ?| ?
\theta_8| No| ?| ?

특히 중요한 것은 \theta_5\sim\theta_8이다.

Distillation에 사용하지 않은 weight에서도 DPO가 D와 S를 구별하지 못한다면,

$$
\boxed{
\text{weight-independent information compression}
}
$$

이라는 해석이 가능해진다.

---

11. DPO Failure에 대한 주의점

DPO가 실패했다고 해서 곧바로

$$
D\equiv S
$$

라고 결론내려서는 안 된다.

DPO failure에는 여러 원인이 있을 수 있다.

예를 들어:

- optimization hyperparameter 문제
- reference model 문제
- sequence length 차이
- formatting 차이
- synthetic vector의 입력 방식
- DPO objective가 해당 차이를 표현하지 못하는 문제

등이 있다.

따라서 DPO는 다른 metric과 함께 사용한다.

---

12. Multi-Level Evaluation

최종 평가는 최소한 다음 네 단계로 구성한다.

12.1 Gradient similarity

$$
\cos
\left(
G_\theta(D),
G_\theta(S)
\right)
$$

12.2 One-step update similarity

$$
\theta_D'

\theta-\eta G_\theta(D)
$$

$$
\theta_S'

\theta-\eta G_\theta(S)
$$

그리고

$$
|\theta_D'-\theta_S'|
$$

를 비교한다.

12.3 Multi-step training trajectory

$$
\theta_0
\rightarrow
\theta_1
\rightarrow
\cdots
$$

가 D와 S에서 얼마나 유사한지 평가한다.

12.4 DPO discrimination

$$
D>S
$$

와

$$
S>D
$$

를 각각 학습시켰을 때 preference learning이 발생하는지 측정한다.

---

13. Architecture Diversity 확장

Same-architecture/different-weight 실험이 성공하면 architecture를 점진적으로 다르게 만든다.

Level 1

동일 architecture + different weights

$$
\text{weight diversity}
$$

Level 2

동일 family + different scale

$$
\text{width/depth diversity}
$$

Level 3

MHA / GQA / MQA 등

$$
\text{attention diversity}
$$

Level 4

Full Attention / Sliding Window

$$
\text{receptive-field diversity}
$$

Level 5

Attention / Attention + GDN / SSM hybrid

$$
\text{sequence-mixing diversity}
$$

Level 6

서로 다른 Transformer family

$$
\text{architecture-family diversity}
$$

Architecture diversity가 증가할수록 objective는

$$
\mathcal{L}

\sum_{f\in\mathcal{F}}
\left|
G_f(S)-G_f(D)
\right|^2
$$

형태로 확장된다.

---

14. 중요한 Held-out Architecture 실험

Distillation에 사용하지 않은 architecture를 반드시 둔다.

예를 들어

$$
\mathcal{F}_{\mathrm{train}}

{
f_1,f_2,f_3,f_4
}
$$

로 S를 만들고,

$$
f_5
$$

를 완전히 held-out한다.

이후 f_5에서

$$
D
\quad\text{vs}\quad
S
$$

를 비교한다.

특히 DPO에서

$$
D>S
$$

와

$$
S>D
$$

모두 실패한다면 강한 결과가 된다.

이것은 단순한 model-specific distillation이 아니라 unseen architecture에서도 구별하기 어려운 training information을 압축했다는 증거가 될 수 있다.

---

15. Distilled Dataset Size Scaling

Synthetic dataset의 크기를

$$
128,\ 256,\ 512,\ 1024,\ 2048
$$

등으로 변화시킨다.

각각에 대해

$$
\text{gradient fidelity}
$$

와

$$
\text{DPO discrimination}
$$

을 측정한다.

목표는 다음 trade-off를 관찰하는 것이다.

$$
\text{number of synthetic vectors}
\quad\leftrightarrow\quad
\text{information preserved}
$$

특히

$$
|S|=1024
$$

에서

$$
|D|=10240
$$

의 학습 효과가 얼마나 보존되는지가 핵심적인 첫 번째 target이다.

---

16. Model Diversity Scaling

Distillation에 사용되는 모델의 수를 증가시킨다.

$$
K=1,2,4,8,16,\ldots
$$

그리고 distilled dataset 크기는 고정한다.

$$
|S|=1024
$$

질문은 다음과 같다.

«더 다양한 model을 동시에 만족시키도록 만들수록 unseen model에서 S의 fidelity가 증가하는가?»

가설:

$$
\text{architecture diversity}
\uparrow
\quad\Rightarrow\quad
\text{held-out fidelity}
\uparrow
$$

---

17. 최종적인 연구 가설

전체 실험이 성공한다면 다음과 같은 단계적 주장을 할 수 있다.

단계 1

$$
D\rightarrow S
$$

특정 모델에서 training signal을 압축한다.

단계 2

서로 다른 weight에서 동시에 성립:

$$
G_{\theta_1}(S)\approx G_{\theta_1}(D)
$$

$$
G_{\theta_2}(S)\approx G_{\theta_2}(D)
$$

따라서 weight-independent representation을 얻는다.

단계 3

Held-out weight에서도 성립:

$$
G_{\theta_{\mathrm{heldout}}}(S)
\approx
G_{\theta_{\mathrm{heldout}}}(D)
$$

따라서 특정 weight configuration에 대한 overfitting을 배제한다.

단계 4

서로 다른 architecture에서도 성립:

$$
G_{f_A}(S)\approx G_{f_A}(D)
$$

$$
G_{f_B}(S)\approx G_{f_B}(D)
$$

$$
G_{f_C}(S)\approx G_{f_C}(D)
$$

따라서 architecture-independent information에 가까워진다.

단계 5

DPO에서도 discrimination 실패:

$$
D_{\mathrm{chosen}},S_{\mathrm{rejected}}
$$

및

$$
S_{\mathrm{chosen}},D_{\mathrm{rejected}}
$$

모두에서 유의미한 preference learning이 발생하지 않는다.

이 단계가 성공하면 단순히

«"dataset을 10배 줄였다"»

보다 훨씬 강한 주장을 할 수 있다.

---

18. 궁극적인 목표

최종적으로 찾고 싶은 것은 단순한 compressed dataset이 아니다.

다음과 같은 Universal Training Representation이다.

$$
D
\longrightarrow
S_{\mathrm{universal}}
$$

such that

$$
|S_{\mathrm{universal}}|
\ll
|D|
$$

그리고 다양한 neural architecture f에 대해

$$
G_f(S_{\mathrm{universal}})
\approx
G_f(D)
$$

를 만족한다.

더 강하게는 held-out architecture f_{\mathrm{new}}에 대해서도

$$
G_{f_{\mathrm{new}}}(S_{\mathrm{universal}})
\approx
G_{f_{\mathrm{new}}}(D)
$$

이며,

$$
\mathrm{DPO}(D>S_{\mathrm{universal}})
$$

와

$$
\mathrm{DPO}(S_{\mathrm{universal}}>D)
$$

모두 의미 있는 discrimination을 만들어내지 못하는 것이다.

---

19. 가장 궁극적인 질문

이 실험이 성공한다면 다음 질문으로 이어진다.

«서로 다른 neural architecture가 동일한 데이터를 학습할 때 공통적으로 필요로 하는 정보에는 얼마나 작은 representation이 존재하는가?»

즉,

$$
\boxed{
\text{8M tokens}
\rightarrow
\text{1024 dense vectors}
}
$$

와 같은 압축이 가능한지를 넘어,

$$
\boxed{
\text{different neural learners}
\rightarrow
\text{shared information bottleneck}
}
$$

이 존재하는지를 탐구하는 것이다.

이 관점에서 Dataset Distillation은 단순한 compression 문제가 아니라,

$$
\boxed{
\text{What information is actually necessary for a neural network to learn from a dataset?}
}
$$

라는 문제로 확장된다.
