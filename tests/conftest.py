from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from loopguard.checkpoint import RunStore


@pytest.fixture
def store(tmp_path: Path) -> RunStore:
    return RunStore(tmp_path / "runs")


async def instant_sleep(_seconds: float) -> None:
    await asyncio.sleep(0)


@pytest.fixture
def no_sleep():
    return instant_sleep
