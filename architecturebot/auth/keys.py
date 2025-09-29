import json
import os
import secrets
import time
from dataclasses import dataclass, asdict
from typing import Dict, Optional


STORAGE_FILE = os.path.join(os.path.dirname(__file__), "keys.json")
UNLIMITED = -1


@dataclass
class ApiKey:
    key: str
    remaining: int
    created_at: float


def _load_store() -> Dict[str, ApiKey]:
    if not os.path.exists(STORAGE_FILE):
        return {}
    with open(STORAGE_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: ApiKey(**v) for k, v in raw.items()}


def _save_store(store: Dict[str, ApiKey]) -> None:
    data = {k: asdict(v) for k, v in store.items()}
    with open(STORAGE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def generate_keys(count: int = 20, quota: int = 10) -> list[str]:
    store = _load_store()
    keys: list[str] = []
    for _ in range(count):
        new_key = secrets.token_urlsafe(24)
        store[new_key] = ApiKey(key=new_key, remaining=quota, created_at=time.time())
        keys.append(new_key)
    _save_store(store)
    return keys


def generate_unlimited_key() -> str:
    store = _load_store()
    new_key = secrets.token_urlsafe(24)
    store[new_key] = ApiKey(key=new_key, remaining=UNLIMITED, created_at=time.time())
    _save_store(store)
    return new_key


def get_key_info(key: str) -> Optional[ApiKey]:
    store = _load_store()
    return store.get(key)


def decrement_quota(key: str) -> int:
    store = _load_store()
    info = store.get(key)
    if not info:
        raise KeyError("Invalid key")
    if info.remaining == UNLIMITED:
        return UNLIMITED
    if info.remaining <= 0:
        return 0
    info.remaining -= 1
    store[key] = info
    _save_store(store)
    return info.remaining


