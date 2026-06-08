# auth.py — Sample authentication module for CodeRAG testing

import hashlib
import os


class UserAuthenticator:
    """Handles user login and session management."""

    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.active_sessions = {}

    def hash_password(self, password: str) -> str:
        """Hash a plaintext password using SHA-256 + salt."""
        salt = os.urandom(16).hex()
        hashed = hashlib.sha256((password + salt + self.secret_key).encode()).hexdigest()
        return f"{salt}:{hashed}"

    def verify_password(self, password: str, stored_hash: str) -> bool:
        """Check if a plaintext password matches a stored hash."""
        salt, hashed = stored_hash.split(":")
        check = hashlib.sha256((password + salt + self.secret_key).encode()).hexdigest()
        return check == hashed

    def login(self, username: str, password: str, user_db: dict) -> str | None:
        """
        Attempt login. Returns a session token if successful, None otherwise.
        user_db format: {username: hashed_password}
        """
        if username not in user_db:
            return None
        if not self.verify_password(password, user_db[username]):
            return None
        token = os.urandom(32).hex()
        self.active_sessions[token] = username
        return token

    def logout(self, token: str) -> bool:
        """Invalidate a session token. Returns True if token existed."""
        if token in self.active_sessions:
            del self.active_sessions[token]
            return True
        return False

    def get_user(self, token: str) -> str | None:
        """Return the username for a session token, or None if invalid."""
        return self.active_sessions.get(token)


def require_login(func):
    """Decorator that checks for a valid session token before running a function."""
    def wrapper(token, authenticator, *args, **kwargs):
        user = authenticator.get_user(token)
        if not user:
            raise PermissionError("Invalid or expired session token.")
        return func(token, authenticator, *args, **kwargs)
    return wrapper


@require_login
def get_profile(token: str, authenticator: UserAuthenticator) -> dict:
    """Return the profile of the logged-in user."""
    username = authenticator.get_user(token)
    return {"username": username, "role": "user"}
