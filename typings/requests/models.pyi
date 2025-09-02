"""Type stubs for requests.models module."""

from typing import Any

class Response:
    status_code: int
    text: str
    content: bytes
    headers: dict[str, str]
    url: str

    def json(self) -> Any: ...
    def raise_for_status(self) -> None: ...
