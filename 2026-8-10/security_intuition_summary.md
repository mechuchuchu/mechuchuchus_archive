# 취약점/보안 직관 테스트 정리 (2026-08-10)

## 출발점
Claude Mythos가 OpenBSD 27년 묵은 취약점, Firefox 271개 취약점 등을 발견한 사건에서 시작.
"왜 취약점 없는 코드를 만드는 게 그렇게 어렵냐"는 질문으로 이어짐.

## 핵심 결론 (한 줄)
**"완벽한 보안"은 성립 불가능한 개념. 실무는 "어디까지 방어하고 어디부터 그냥 믿을지" trust boundary를 정직하게 긋는 것.**

---

## 대화 흐름 & 논점

### 1. 왜 취약점이 안 없어지나
- 코드가 어려워서가 아니라 **state space가 조합적으로 폭발**해서 검증이 계산적으로 불가능.
- 사람 리뷰/기존 fuzzing은 이 공간에서 "그럴듯한" 좁은 subspace만 샘플링함.
- AI(Mythos)가 잘하는 건 이 subspace를 넓힌 것이지, 공간 자체를 줄인 게 아님.

### 2. "규칙트리는 안전하지 않나?"
- 규칙 자체(트리 로직)가 완벽해도 취약점은 대부분 **트리 밖**에서 남:
  - rule interaction / ordering bug
  - 트리가 커버 안 하는 입력(coverage gap)
  - side channel (timing 등)
  - 실행 환경(컴파일러, 런타임)의 배신

### 3. "권한만 조절하면 취약점 0 아니냐?"
- Permission 시스템 자체도 코드로 구현 → 그 안에도 버그 생김.
- Enforcement layer 버그, side channel(간접 추론), confused deputy 문제.
- **Access control = attack surface를 줄이는 것이지 0으로 만드는 게 아님.**

### 4. "많이 발견됐으니 이제 거의 없지 않나?"
- 반대: 발견이 많다는 건 "기존 탐색 도구의 한계가 드러난 것"이지 "고갈"이 아님.
- 코드베이스 자체도 계속 커지고 바뀜 (움직이는 타겟).
- 탐색 도구가 세질수록 이전엔 안 보이던 새 레이어의 취약점이 열림.

### 5. 공격자 vs 방어자 비대칭
- 공격자: 취약점 하나만 찾으면 됨 (OR)
- 방어자: 발견된 모든 걸 다 막아야 함 (AND), 패치가 새 버그 만들 수도 있음
- Discovery 속도(AI)가 방어 속도(사람 patch cycle)보다 빨라지는 게 문제.

### 6. State space 크기 감 잡기 (TCP 예시)
- TCP FSM의 "11개 상태"는 **abstraction**일 뿐, 실제 state space는 그 안의 모든 변수(seq/ack number, SACK block list, timer 등) 조합.
- "다이어그램/문서 ≠ 실제 구현" — 압축된 표현과 실제 사이 gap에서 버그가 남.
- (예은 본인 비유: 이게 basin escape에서 L2/effective rank가 실제 고차원 landscape를 압축해서 보여주는 것과 같은 구조)

### 7. "바이트=신호인데 왜 취약점이 생기나?"
- 바이트 자체는 중립. 프로그램이 그걸 **해석(parsing)** 하는 순간부터 위험해짐.
- Von Neumann 구조상 데이터가 나중에 명령어로 재해석될 수 있음.
- (비유: embedding vector도 중립인데 forward pass 안에서 해석되며 model behavior를 결정 — soft prompt steering과 동일 패턴)

### 8. 실전 코드 리뷰 (3라운드)
1. 평문 비밀번호 소켓 코드 → 하드코딩 비번, 평문 전송, timing attack, rate limit 없음, decode 예외처리 없음 등
2. TLS + PBKDF2 + hmac.compare_digest 코드 → crypto는 맞았지만 decode 예외, TLS 버전 미고정, 입력 길이 제한 없음, rate limiting 없음, 로깅 없음
3. Framing + rate limiter + logging + semaphore 코드 → **rate limiter 이중 카운팅 버그**, **unbounded dict growth**, **TLS handshake가 accept loop 블로킹**, **클로저 late-binding 버그**, 죽은 UTF-8 검증 코드

→ 매 라운드 "이제 완벽"이라고 생각했지만 다른 레이어에서 계속 새 문제 발견됨.
→ **레이어를 하나 추가할 때마다 그 레이어 자체가 새로운 버그 표면이 됨.**

### 9. 입력값 기반 버그 분석
- length=0, boundary value(128), UTF-8 파싱 등 개별 케이스 점검.
- 결론: 이 코드는 input layer는 비교적 견고. 남은 버그는 input이 아니라 **concurrency/상태 관리**에서 나옴.

### 10. "그럼 취약점 0 쉬운 거 아니냐?"에 대한 반박
- 지금까지 안 찾았다 ≠ 없다 (오늘 3라운드가 실증).
- 좁은 프로토콜 하나에서 0 달성 ≠ 스케일업해도 0.
- 코드가 안 바뀌어도 **의존하는 모든 레이어**(OpenSSL, interpreter, OS, 하드웨어)가 다 완벽해야 진짜 0.

### 11. Random byte fuzzing 시뮬레이션
- 이 서버 기준: crash까지는 잘 안 가지만 TLS handshake DoS, rate limiter 오작동 유발, connection exhaustion 등은 랜덤 fuzzing으로도 드러날 수 있음.
- Semantic-aware fuzzing(Mythos류)이 구조적 로직 버그를 훨씬 효율적으로 찾음.

### 12. 최종 방어 코드 (server.py)
아래 항목들을 명시적으로 고침:
- rate limiter 실패만 카운트 (이중 카운팅 제거)
- TTL + LRU eviction으로 dict unbounded growth 방지
- TLS handshake를 워커 스레드로 이동 + 자체 timeout (accept loop 블로킹 방지)
- 클로저 인자 기본값 바인딩으로 late-binding 버그 방지
- 불필요했던 UTF-8 사전 검증(죽은 코드) 제거
- cipher suite 명시적 제한
- graceful shutdown 추가

**파일 상단에 의도적으로 스코프 밖에 둔 항목을 명시**:
OS/커널 취약점, 언어/라이브러리 자체 CVE, 사이드채널, 분산 DDoS, 키 관리 프로세스.

### 13. 스택 전체로 확장
```
애플리케이션 코드
→ 언어 런타임 (CPython/OpenSSL)
→ OS 커널 (TCP stack)
→ 하이퍼바이저
→ NIC 펌웨어
→ 공유기/라우터
→ ISP 인프라
→ 물리 신호/사이드채널
→ 하드웨어 (Spectre/Meltdown, Rowhammer)
```
어느 레이어든 하나만 뚫려도 전체가 뚫림. 대부분 레이어는 개발자 control 밖.

→ 그래서 실무 철학은 "안전하다 증명"이 아니라 **defense in depth** + **assume breach**.

---

## 연구와의 연결점 (예은이 스스로 도출)
- Basin escape 회의주의(L2/effective rank로 판단 불가) ↔ TCP FSM 11개 상태가 실제 고차원 state를 압축해서 보여주는 것 — 같은 구조.
- "잘 정렬된 모델은 탈옥이 어렵지만 불가능하다는 보장은 없다" ↔ 보안도 "패치할수록 안전해지지만 0이 된다는 보장은 없다."
- Trust boundary 문제: 판단 도구 자체가 신뢰 가능하다는 전제 위에서만 판단이 성립 — measurement layer의 신뢰성 문제와 동형.

---

## 산출물
- `server.py` — 방어적으로 재작성한 TLS 인증 서버 (스코프 명시 포함)
