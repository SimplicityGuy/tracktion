"""Type stubs for dateutil.parser module."""

from datetime import datetime
from typing import Any

class parserinfo:  # noqa: N801 - External library type stub must match original name
    def __init__(
        self,
        dayfirst: bool = ...,
        yearfirst: bool = ...,
    ) -> None: ...

def parse(timestr: str, parserinfo: parserinfo | None = ..., **kwargs: Any) -> datetime: ...
def isoparse(dt_str: str) -> datetime: ...

class ParserError(ValueError): ...
