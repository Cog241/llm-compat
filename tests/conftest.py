'''Shared fixtures. All automated API traffic stays on the local mock server.'''

from __future__ import annotations

from typing import Any, Callable

import pytest

from llm_compat.config import ProviderConfig
from tests.mock_server import MockAPIServer


@pytest.fixture(autouse=True)
def _bypass_proxies_for_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('NO_PROXY', '127.0.0.1,localhost')
    monkeypatch.setenv('no_proxy', '127.0.0.1,localhost')


@pytest.fixture
def mock_server() -> Any:
    with MockAPIServer() as server:
        yield server


@pytest.fixture
def config_factory(
    mock_server: MockAPIServer,
) -> Callable[..., ProviderConfig]:
    def make(**overrides: Any) -> ProviderConfig:
        values = {
            'name': 'mock',
            'base_url': mock_server.base_url,
            'model': 'mock-chat',
            'timeout': 5.0,
        }
        values.update(overrides)
        return ProviderConfig(**values)

    return make
