"""
방어적으로 작성한 TLS 인증 서버 — 교육/직관 테스트용.

명시적 스코프 (이 밖은 다루지 않음):
  - OS/커널 레벨 취약점 (예: TCP stack 버그)
  - Python interpreter / OpenSSL 라이브러리 자체의 CVE
  - 물리적 접근, 사이드채널(전력분석 등)
  - DDoS (분산 다중 IP 공격) — 단일 서버 코드로 막을 수 있는 영역이 아님
  - 키/인증서 발급, 저장, 로테이션 절차 (운영 프로세스 영역)

즉 "이 파일 안의 로직 레벨 방어"만 다룸. 이것도 "취약점 0"이 아니라
"알려진 카테고리들을 의식적으로 방어했다"는 뜻일 뿐임.
"""

import socket
import ssl
import hashlib
import hmac
import logging
import threading
import time
import signal
import sys
from collections import defaultdict, deque

# --------------------------------------------------
# 설정
# --------------------------------------------------

HOST = "127.0.0.1"
PORT = 5000

MAX_PASSWORD_BYTES = 128
MIN_PASSWORD_BYTES = 1

HANDSHAKE_TIMEOUT = 5      # TLS handshake 자체에 대한 timeout
SOCKET_TIMEOUT = 5         # 인증 메시지 주고받을 때 timeout

PBKDF2_ITERATIONS = 300_000

MAX_WORKERS = 20
MAX_PENDING_HANDSHAKES = 10  # accept 이후 handshake 대기 슬롯 (accept loop 보호용)

MAX_ATTEMPTS = 5
RATE_WINDOW = 60.0
BASE_BACKOFF = 2.0
MAX_BACKOFF = 60.0

RATE_STATE_TTL = 600.0     # 오래된 IP 기록 정리 주기 (메모리 누수 방지)
RATE_STATE_MAX_IPS = 10_000  # 딕셔너리 상한 (unbounded growth 방지)

SALT = bytes.fromhex("여기에_salt_hex_값")
PASSWORD_HASH = bytes.fromhex("여기에_hash_hex_값")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------
# Rate limiter
#   - 실패만 카운트 (성공은 카운트하지 않음)
#   - IP별 기록에 TTL을 둬서 unbounded growth 방지
#   - 상한(RATE_STATE_MAX_IPS) 넘으면 가장 오래된 기록부터 정리
# --------------------------------------------------

class RateLimiter:
    def __init__(self):
        self._lock = threading.Lock()
        self._failures = defaultdict(deque)   # ip -> deque[timestamp]
        self._blocked_until = {}              # ip -> timestamp
        self._last_seen = {}                  # ip -> timestamp (LRU eviction용)

    def _evict_if_needed(self, now):
        if len(self._last_seen) <= RATE_STATE_MAX_IPS:
            return
        # 가장 오래 안 본 IP부터 제거 (단순 LRU 근사)
        stale = sorted(self._last_seen.items(), key=lambda kv: kv[1])
        for ip, _ in stale[: len(self._last_seen) - RATE_STATE_MAX_IPS]:
            self._failures.pop(ip, None)
            self._blocked_until.pop(ip, None)
            self._last_seen.pop(ip, None)

    def is_allowed(self, ip):
        now = time.monotonic()
        with self._lock:
            self._last_seen[ip] = now

            blocked = self._blocked_until.get(ip, 0.0)
            if now < blocked:
                return False

            dq = self._failures[ip]
            while dq and now - dq[0] > RATE_WINDOW:
                dq.popleft()

            if len(dq) >= MAX_ATTEMPTS:
                backoff = min(
                    BASE_BACKOFF ** (len(dq) - MAX_ATTEMPTS + 1),
                    MAX_BACKOFF,
                )
                self._blocked_until[ip] = now + backoff
                logger.warning("rate limited ip=%s backoff=%.1fs", ip, backoff)
                return False

            self._evict_if_needed(now)
            return True

    def register_failure(self, ip):
        now = time.monotonic()
        with self._lock:
            self._failures[ip].append(now)
            self._last_seen[ip] = now


rate_limiter = RateLimiter()


# --------------------------------------------------
# TCP framing: [2바이트 length][payload]
# --------------------------------------------------

def recv_exact(sock, size):
    if size == 0:
        return b""
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("connection closed during recv")
        data.extend(chunk)
    return bytes(data)


def recv_message(sock, max_size):
    header = recv_exact(sock, 2)
    length = int.from_bytes(header, byteorder="big", signed=False)
    if length > max_size:
        raise ValueError(f"message too large: {length} > {max_size}")
    if length < MIN_PASSWORD_BYTES:
        raise ValueError("empty message not allowed")
    return recv_exact(sock, length)


def send_message(sock, data: bytes):
    if len(data) > 65535:
        raise ValueError("outgoing message too large")
    header = len(data).to_bytes(2, byteorder="big")
    sock.sendall(header + data)


# --------------------------------------------------
# 인증
#   - decode는 절대 하지 않음: password는 raw bytes로만 다룸
#     (UTF-8 여부를 신경 쓸 이유가 없음 — PBKDF2는 bytes를 받으므로
#      "유효한 UTF-8인지 미리 검증"은 불필요한 죽은 로직이자
#      추가 예외 표면일 뿐이었음)
# --------------------------------------------------

def verify_password(password_bytes: bytes) -> bool:
    if not (MIN_PASSWORD_BYTES <= len(password_bytes) <= MAX_PASSWORD_BYTES):
        return False
    candidate_hash = hashlib.pbkdf2_hmac(
        "sha256", password_bytes, SALT, PBKDF2_ITERATIONS
    )
    return hmac.compare_digest(candidate_hash, PASSWORD_HASH)


# --------------------------------------------------
# 클라이언트 처리
# --------------------------------------------------

def handle_client(conn: ssl.SSLSocket, ip: str):
    try:
        conn.settimeout(SOCKET_TIMEOUT)

        if not rate_limiter.is_allowed(ip):
            send_message(conn, b"TRY_LATER")
            return

        send_message(conn, b"PASSWORD")

        try:
            password_bytes = recv_message(conn, MAX_PASSWORD_BYTES)
        except ValueError as e:
            logger.warning("bad framing from %s: %s", ip, e)
            send_message(conn, b"AUTH_FAILED")
            rate_limiter.register_failure(ip)
            return

        if verify_password(password_bytes):
            logger.info("auth success ip=%s", ip)
            send_message(conn, b"AUTH_OK")
        else:
            logger.warning("auth failure ip=%s", ip)
            rate_limiter.register_failure(ip)
            send_message(conn, b"AUTH_FAILED")

    except socket.timeout:
        logger.warning("timeout ip=%s", ip)
    except (ConnectionError, BrokenPipeError, ssl.SSLError):
        logger.info("connection closed ip=%s", ip)
    except Exception:
        # 클라이언트발 예외로 서버 전체가 죽지 않도록 최후 방어선
        logger.exception("unexpected error ip=%s", ip)
    finally:
        try:
            conn.close()
        except Exception:
            pass


# --------------------------------------------------
# TLS 컨텍스트
# --------------------------------------------------

def build_tls_context() -> ssl.SSLContext:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(certfile="server.crt", keyfile="server.key")
    # 약한 cipher 명시적으로 배제 (시스템 기본값에만 의존하지 않음)
    ctx.set_ciphers("ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM")
    return ctx


# --------------------------------------------------
# Server main loop
#   - handshake도 워커 스레드에서 수행 (accept loop 블로킹 방지)
#   - handshake 자체에도 timeout 적용 (slow-handshake DoS 방지)
#   - semaphore로 동시 처리 수 제한 (accept는 계속 받되 처리만 제한)
# --------------------------------------------------

def serve():
    tls_context = build_tls_context()
    worker_semaphore = threading.BoundedSemaphore(MAX_WORKERS)
    shutdown_event = threading.Event()

    def handle_signal(signum, frame):
        logger.info("shutdown signal received")
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((HOST, PORT))
        sock.listen(64)
        sock.settimeout(1.0)  # shutdown_event를 주기적으로 체크하기 위함

        logger.info("listening on %s:%d", HOST, PORT)

        while not shutdown_event.is_set():
            try:
                raw_conn, addr = sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            ip = addr[0]
            raw_conn.settimeout(HANDSHAKE_TIMEOUT)

            acquired = worker_semaphore.acquire(timeout=0.5)
            if not acquired:
                # 워커 다 찼으면 즉시 닫음 (accept loop는 계속 살아있음)
                logger.warning("worker pool full, dropping ip=%s", ip)
                raw_conn.close()
                continue

            def worker(raw_conn=raw_conn, ip=ip):
                # 인자 기본값으로 바인딩 -> 클로저 late-binding 버그 방지
                try:
                    try:
                        tls_conn = tls_context.wrap_socket(raw_conn, server_side=True)
                    except (ssl.SSLError, socket.timeout, OSError) as e:
                        logger.warning("tls handshake failed ip=%s: %s", ip, e)
                        raw_conn.close()
                        return
                    handle_client(tls_conn, ip)
                finally:
                    worker_semaphore.release()

            threading.Thread(target=worker, daemon=True).start()

    logger.info("server stopped")


if __name__ == "__main__":
    serve()
