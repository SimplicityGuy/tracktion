"""Type stubs for aiofiles package."""

from collections.abc import Awaitable
from pathlib import Path
from typing import Any

__version__: str

class AiofilesContextManager:
    def __aenter__(self) -> Awaitable[Any]: ...
    def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Awaitable[None]: ...

def open(
    file: str | Path | int,
    mode: str = ...,
    buffering: int = ...,
    encoding: str | None = ...,
    errors: str | None = ...,
    newline: str | None = ...,
    closefd: bool = ...,
    opener: Any = ...,
    loop: Any = ...,
    executor: Any = ...,
) -> AiofilesContextManager: ...
async def remove(path: str | Path) -> None: ...
