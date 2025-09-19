from __future__ import annotations

from typing import Any, Callable, ClassVar, TypeVar

_T = TypeVar("_T")


class BaseModel:
    model_config: ClassVar[Any]
    model_fields: ClassVar[dict[str, Any]]

    def __init__(self, **data: Any) -> None: ...

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]: ...

    @classmethod
    def model_validate(cls: type[_T], obj: Any, *args: Any, **kwargs: Any) -> _T: ...

    @classmethod
    def model_validate_json(cls: type[_T], json_data: str, *args: Any, **kwargs: Any) -> _T: ...


class ConfigDict(dict[str, Any]): ...


def Field(
    default: Any | None = None,
    *,
    default_factory: Callable[[], Any] | None = None,
    alias: str | None = None,
    description: str | None = None,
    title: str | None = None,
    ge: float | None = None,
    gt: float | None = None,
    le: float | None = None,
    lt: float | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
) -> Any: ...


def field_validator(*fields: str, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...


def model_validator(*args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...


class HttpUrl(str): ...


class NonNegativeFloat(float): ...


class NonNegativeInt(int): ...


class PositiveFloat(float): ...


class ValidationError(Exception):
    def errors(self) -> list[dict[str, Any]]: ...
