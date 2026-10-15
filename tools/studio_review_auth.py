"""In-memory reviewer credentials, expiring sessions and bounded login attempts."""
import hashlib
import re
import secrets
import threading
import time

SESSION_SECONDS = 8*60*60
LOGIN_WINDOW = 60
PASSWORD_ITERATIONS = 600000


class ReviewDenied(PermissionError):
    pass


class ReviewLimited(ReviewDenied):
    pass


def password_hash(password, salt):
    return hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, PASSWORD_ITERATIONS)


class ReviewAuth:
    def __init__(self, accounts):
        if not isinstance(accounts, list) or not 1 <= len(accounts) <= 12:
            raise ValueError('请设置 1–12 个审核账号')
        self.accounts = {}
        for row in accounts:
            if not isinstance(row, dict):
                raise ValueError('审核账号格式无效')
            name, password = row.get('username'), row.get('password')
            if not isinstance(name, str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,32}', name):
                raise ValueError('账号请使用 1–32 位字母、数字、下划线或短横线')
            if not isinstance(password, str) or not 10 <= len(password) <= 128:
                raise ValueError('每个账号请设置 10–128 位密码')
            if name in self.accounts:
                raise ValueError('账号名称不能重复')
            salt = secrets.token_bytes(16)
            self.accounts = {**self.accounts, name: (salt, password_hash(password, salt))}
        self.dummy = (secrets.token_bytes(16), secrets.token_bytes(32))
        self.nonce = secrets.token_urlsafe(32)
        self.sessions, self.attempts = {}, {}
        self.lock = threading.RLock()

    def throttle(self, peer):
        now = time.time()
        self.attempts = {ip: [stamp for stamp in stamps if stamp > now-LOGIN_WINDOW]
                         for ip, stamps in self.attempts.items() if any(stamp > now-LOGIN_WINDOW for stamp in stamps)}
        if len(self.attempts.get(peer, [])) >= 5 or sum(map(len, self.attempts.values())) >= 30:
            raise ReviewLimited('登录尝试过于频繁，请一分钟后再试')
        self.attempts = {**self.attempts, peer: self.attempts.get(peer, [])+[now]}

    def login(self, username, password, peer):
        with self.lock:
            self.throttle(peer)
            if not isinstance(username, str) or not isinstance(password, str) or len(password) > 128:
                raise ReviewDenied('账号或密码不正确')
            salt, expected = self.accounts.get(username, self.dummy)
            actual = password_hash(password, salt)
            if not secrets.compare_digest(actual, expected) or username not in self.accounts:
                raise ReviewDenied('账号或密码不正确')
            now = time.time()
            self.sessions = {key: row for key, row in self.sessions.items() if row['expires'] > now}
            if len(self.sessions) >= 128:
                raise ReviewLimited('登录会话过多，请稍后再试')
            cookie = secrets.token_urlsafe(48)
            row = {'username': username, 'csrf': secrets.token_urlsafe(32), 'expires': now+SESSION_SECONDS}
            self.sessions = {**self.sessions, cookie: row}
            return cookie, dict(row)

    def session(self, cookie):
        with self.lock:
            row = self.sessions.get(cookie)
            if not row or row['expires'] <= time.time():
                self.logout(cookie)
                raise ReviewDenied('请登录审核账号')
            return dict(row)

    def logout(self, cookie):
        with self.lock:
            self.sessions = {key: row for key, row in self.sessions.items() if key != cookie}

    def revoke(self):
        with self.lock:
            self.sessions = {}
            self.accounts = {}
