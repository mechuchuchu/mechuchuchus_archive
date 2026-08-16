# allenai/SERA-8B(FT) vs Qwen/Qwen3-8B(sera's base) — SAE Activation 비교 실험 노트

날짜: 2026-08-08
관련 맥락: "weight-only open model이라도 full FT(Tulu/Molmo/Sera 스타일)로 조지면 base의 데이터 편향/블랙박스가 덮이는가?" 논쟁에서, functional 기준(behavior 변화) 대신 internal activation 기준으로 직접 측정해보기로 함.

## 배경 / 동기

- L2 weight distance, effective rank 같은 weight-space 지표는 "얼마나 멀리 움직였나"는 보여줘도 "barrier를 넘었나/basin이 넓은가"를 구분 못 함 (basin 탈출 기준 자체가 모호하다는 결론).
- → weight space보다 causally 직접적인 **activation space**에서 base 대비 FT 모델이 표현을 얼마나 재구성했는지 보기로 함.
- 방법: Qwen 공식 SAE로 base(qwen) vs FT(sera, coding agent) 모델의 동일 입력에 대한 activation 비교. Layer 0 / 10 / 24 / 35, active feature set, top-10 순위, top value / mean activation 비교.

## 실험 1: 일반 문장 (도메인 무관)

문장: "The capital of France is", "Artificial intelligence is changing the world.", "Deep learning models require a lot of data.", "Python is a popular programming language."

### Active feature Jaccard overlap (sera vs qwen, 4문장 평균)
| Layer | Jaccard | Top-10 overlap |
|---|---|---|
| L0 | 0.85~0.94 | 9~10/10 |
| L10 | 0.80~0.85 | 9~10/10 |
| L24 | 0.72~0.92 | 7~10/10 |
| L35 | 0.68~0.80 | 7~9/10 |

### Magnitude (L35 top value)
- sera/qwen ratio: **0.38~0.48** (일관되게 절반 수준)
- mean activation은 sera가 오히려 비슷하거나 약간 높음 (분포 전체는 안 바뀜)
- → L35의 특정 outlier(top-1) feature 하나만 확 눌린 패턴. sink-feature 계열 추정.

**주의**: 이 결과는 이후 재해석됨 — sera가 coding agent라서 일반 문장에는 애초에 FT 효과가 잘 안 걸릴 수 있음. 아래 실험 2로 재검증.

## 실험 2: Agentic/coding 문장 (sera의 실제 도메인)

문장: "Run the tests and report the results.", "I'll edit this file to fix the issue.", "Let me check the codebase for similar patterns.", "The build failed, investigating the error log now."

### Active feature Jaccard overlap (4문장 평균)
| Layer | Jaccard (agentic) | Jaccard (일반, 비교용) |
|---|---|---|
| L0 | 0.87~0.92 | 0.85~0.94 (거의 동일) |
| L10 | 0.67~0.77 | 0.80~0.85 (↓) |
| L24 | 0.58~0.71 | 0.72~0.92 (↓↓ 최대 격차) |
| L35 | 0.49~0.68 | 0.68~0.80 (↓) |

### Magnitude (L35 top value)
- sera/qwen ratio: 0.39~0.54 — 실험 1과 거의 동일 수준.
- qwen 쪽 top-1 feature ID가 4문장 모두 **동일 (feature 28396)** → 도메인 무관하게 항상 튀는 sink-like feature.
- sera 쪽 top-1은 문장마다 다름 (50166, 24363, 28396 등) → 그 sink feature가 억제되면서 다른 feature가 대신 top으로 올라오는 것으로 보임.

## 잠정 결론

1. **L35 outlier(sink) feature 억제(ratio ~0.4~0.5)는 도메인 무관, global한 SFT 부수효과로 보임.** 일반 문장/agentic 문장 모두 똑같이 관측됨. 특정 지식/편향이 덮인 결과라기보단 SFT 과정(instruction format, sequence length 분포 변화 등) 자체의 일반적 side-effect일 가능성.
2. **L10~L24의 active set 재구성(Jaccard 하락)은 domain-specific.** Agentic 문장에서만 뚜렷하게 커짐 (특히 L24: 0.72~0.92 → 0.58~0.71). Coding agent SFT가 mid-to-late layer의 semantic/task representation을 실제로 도메인에 맞게 재구성했다는 direct evidence.
3. L0(표면/토큰 레벨)는 도메인 무관하게 거의 안 바뀜 — 예상대로 초반 layer는 거의 보존.
4. → "FT가 base 편향을 덮는가"에 대한 답은 layer/메커니즘별로 나뉨: mid-layer 표현은 도메인 특화적으로 실제 재구성되지만, final-layer 특정 outlier feature 억제는 도메인과 무관한 global 현상.

## 방법론 노트 (재현용)

- 파싱: `Active Features` (tensor set), `Features/Top Feature Starting with the ID with the largest value` (활성값 내림차순 정렬된 feature id 리스트), `Top Value`, `Mean Activation` 필드를 정규식으로 추출.
- Jaccard = |sera ∩ qwen| / |sera ∪ qwen| (active feature id set, layer별).
- Top-10 overlap = 상위 10개 feature id 겹치는 개수.
- Layer 0 / 10 / 24 / 35 로 고정 (초반/중반/후반/최종층 대표 샘플링).

## 추후 실험 (TODO)

- [ ] Feature 28396이 정확히 뭘 encode하는지 확인 (Qwen SAE feature dashboard 있으면 조회, 없으면 activating example 직접 수집해서 추정)
- [ ] L24에서 sera/qwen 갈리는 구체적 feature id들 뽑아서 (jaccard로 빠진 것들) 의미 분석 — 어떤 concept이 agent FT로 새로 생기고/사라졌는지
- [ ] Minimal pair 세트 실험 (원래 계획했던 성별/지역/언어권 편향 축) — 편향이 "덮이는지" 직접 타겟팅
- [ ] 코드 스타일 minimal pair (snake_case vs camelCase 등) — 컨벤션 편향
- [ ] 언어/프레임워크 triplet (Python/Flask vs JS/Express vs Go) — 언어 선호 편향
- [ ] Layer 해상도 높이기 (0/10/24/35 대신 더 촘촘하게) — 정확히 몇 층부터 divergence 시작하는지 확인
- [ ] Linear interpolation (base→FT weight path) loss barrier 측정과 함께 봐서, activation divergence가 barrier 존재 여부랑 상관있는지 교차검증
- [ ] Sample 수 늘리기 (현재 문장당 1개, 4개 문장뿐이라 통계적으로 약함) — 카테고리별 문장 10개 이상으로 재실험
