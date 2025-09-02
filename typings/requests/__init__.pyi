"""Type stubs for requests package."""

from typing import Any

from requests.models import Response as Response

__version__: str

class Session:
    headers: dict[str, str]

    def get(self, url: str, params: dict[str, Any] | None = ..., **kwargs: Any) -> Response: ...
    def post(
        self, url: str, data: dict[str, Any] | str | None = ..., json: dict[str, Any] | None = ..., **kwargs: Any
    ) -> Response: ...
    def request(
        self,
        method: str,
        url: str,
        params: dict[str, Any] | None = ...,
        data: dict[str, Any] | str | None = ...,
        headers: dict[str, str] | None = ...,
        timeout: float | tuple[float, float] | None = ...,
        **kwargs: Any,
    ) -> Response: ...
    def close(self) -> None: ...

def get(url: str, params: dict[str, Any] | None = ..., **kwargs: Any) -> Response: ...
def post(
    url: str, data: dict[str, Any] | str | None = ..., json: dict[str, Any] | None = ..., **kwargs: Any
) -> Response: ...
def delete(url: str, **kwargs: Any) -> Response: ...

class RequestException(Exception): ...  # noqa: N818 - External library type stub must match original name
class HTTPError(RequestException): ...
class ConnectionError(RequestException): ...
class Timeout(RequestException): ...
