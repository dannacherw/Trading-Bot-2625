"""
api/auth.py
HTTP Basic Auth middleware with role-based access control.

Roles:
  ADMIN  — full read + write access (kill switch, resume, config)
  VIEWER — read-only access to all dashboards and data

Credentials are stored in .env:
  API_ADMIN_USER  = "admin"
  API_ADMIN_PASS  = "your_password_here"
  API_VIEWER_USER = "viewer"          # comma-separated for multiple viewers
  API_VIEWER_PASS = "viewer_password"

Passwords are compared using hmac.compare_digest to prevent timing attacks.
In production, use properly hashed passwords (bcrypt). For a single-machine
deployment shared with a small team, this is pragmatically sufficient.

Usage:
    from api.auth import require_admin, require_viewer

    @router.post("/kill")
    async def kill(user: str = Depends(require_admin)):
        ...

    @router.get("/scans")
    async def get_scans(user: str = Depends(require_viewer)):
        ...
"""
from __future__ import annotations

import hmac
import os
import secrets
from enum import Enum
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic()


class Role(str, Enum):
    ADMIN  = "admin"
    VIEWER = "viewer"


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _load_credentials() -> dict[str, tuple[str, Role]]:
    """
    Build a {username: (password, role)} map from environment variables.

    Supports multiple viewer accounts via comma-separated lists:
        API_VIEWER_USER=alice,bob
        API_VIEWER_PASS=alice_pw,bob_pw
    """
    creds: dict[str, tuple[str, Role]] = {}

    # Admin (single account)
    admin_user = _env("API_ADMIN_USER",  "admin")
    admin_pass = _env("API_ADMIN_PASS",  "")
    if admin_pass:
        creds[admin_user] = (admin_pass, Role.ADMIN)
    else:
        import warnings
        warnings.warn(
            "API_ADMIN_PASS not set. Dashboard admin access is DISABLED. "
            "Set API_ADMIN_PASS in your .env file.",
            stacklevel=2,
        )

    # Viewers (may be comma-separated for multiple people)
    viewer_users = [u.strip() for u in _env("API_VIEWER_USER", "").split(",") if u.strip()]
    viewer_passes = [p.strip() for p in _env("API_VIEWER_PASS", "").split(",") if p.strip()]

    for user, pw in zip(viewer_users, viewer_passes):
        if user and pw:
            creds[user] = (pw, Role.VIEWER)

    return creds


# Loaded once at import time — restart the API to pick up credential changes
_CREDENTIALS: dict[str, tuple[str, Role]] = _load_credentials()


def _authenticate(credentials: HTTPBasicCredentials) -> tuple[str, Role]:
    """
    Validate credentials against _CREDENTIALS.
    Uses constant-time comparison to resist timing attacks.
    Returns (username, role) on success, raises 401 on failure.
    """
    entry = _CREDENTIALS.get(credentials.username)

    if entry is None:
        # Use a dummy comparison to maintain constant-time behaviour
        # even when the username doesn't exist
        secrets.compare_digest(
            credentials.password.encode(), b"__dummy__"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )

    stored_password, role = entry
    password_ok = hmac.compare_digest(
        credentials.password.encode("utf-8"),
        stored_password.encode("utf-8"),
    )
    if not password_ok:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )

    return credentials.username, role


def require_viewer(
    credentials: Annotated[HTTPBasicCredentials, Depends(security)]
) -> str:
    """Dependency: require VIEWER or ADMIN role. Returns username."""
    username, _role = _authenticate(credentials)
    return username


def require_admin(
    credentials: Annotated[HTTPBasicCredentials, Depends(security)]
) -> str:
    """Dependency: require ADMIN role only. Returns username."""
    username, role = _authenticate(credentials)
    if role != Role.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return username
