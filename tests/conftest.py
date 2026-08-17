"""Shared fixtures. cai.hook / cai.command / cai.tool / cai.mcp_server register
into the process-default Environment when no load() is running; start every test
from a fresh default so a registration made by one test never leaks into
another. config overrides (--base-url / --api-key) are module state too and get
the same reset."""
import pytest

from cai import config
from cai.environment import Environment


@pytest.fixture(autouse=True)
def _fresh_default_environment():
    Environment._default = None
    config._overrides.clear()
    yield
    Environment._default = None
    config._overrides.clear()
