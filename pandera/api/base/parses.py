"""Data validation base parse."""

from collections import namedtuple
import inspect
from itertools import chain
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    NamedTuple,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    no_type_check,
)
from multimethod import multidispatch as _multidispatch

from pandera.backends.base import BaseParseBackend


class ParseResult(NamedTuple):
    """Parse result for user-defined parses."""

    parse_output: Any
    parse_passed: bool
    parsed_object: Any


_T = TypeVar("_T", bound="BaseParse")


# pylint: disable=invalid-name
class multidispatch(_multidispatch):
    """
    Custom multidispatch class to handle copy, deepcopy, and code retrieval.
    """

    @property
    def __code__(self):
        """Retrieves the 'base' function of the multidispatch object."""
        assert len(self) > 0, f"multidispatch object {self} has no functions registered"
        fn, *_ = [*self.values()]  # type: ignore[misc]
        return fn.__code__

    def __reduce__(self):
        """
        Handle custom pickling reduction method by initializing a new
        multidispatch object, wrapped with the base function.
        """
        state = self.__dict__
        # make sure all registered functions at time of pickling are captured
        state["__registered_functions__"] = [*self.values()]
        return (
            multidispatch,  # object creation function
            (state["__wrapped__"],),  # arguments to said function
            state,  # arguments to `__setstate__` after creation
        )

    def __setstate__(self, state):
        """Custom unpickling logic."""
        self.__dict__ = state
        # rehydrate the multidispatch object with unpickled registered functions
        for fn in state["__registered_functions__"]:
            self.register(fn)


class MetaParse(type):  # pragma: no cover
    """Parse metaclass."""

    BACKEND_REGISTRY: Dict[Tuple[Type, Type], Type[BaseParseBackend]] = {}  # noqa
    """Registry of parase backends implemented for specific data objects."""

    Parse_FUNCTION_REGISTRY: Dict[str, Callable] = {}  # noqa
    """Built-in parse function registry."""

    REGISTERED_CUSTOM_PARSES: Dict[str, Callable] = {}  # noqa
    """User-defined custom parses."""

    def __getattr__(cls, name: str) -> Any:
        """Prevent attribute errors for registered parses."""
        attr = {
            **cls.__dict__,
            **cls.PARSE_FUNCTION_REGISTRY,
            **cls.REGISTERED_CUSTOM_PARSES,
        }.get(name)
        if attr is None:
            raise AttributeError(
                f"'{cls}' object has no attribute '{name}'. "
                "Make sure any custom parses have been registered "
                "using the extensions api."
            )
        return attr

    def __dir__(cls) -> Iterable[str]:
        """Allow custom parses to show up as attributes when autocompleting."""
        return chain(
            super().__dir__(),
            cls.PARSE_FUNCTION_REGISTRY.keys(),
            cls.REGISTERED_CUSTOM_PARSES.keys(),
        )

    # pylint: disable=line-too-long
    # mypy has limited metaclass support so this doesn't pass typecheck
    # see https://mypy.readthedocs.io/en/stable/metaclasses.html#gotchas-and-limitations-of-metaclass-support
    # pylint: enable=line-too-long
    @no_type_check
    def __contains__(cls: Type[_T], item: Union[_T, str]) -> bool:
        """Allow lookups for registered parses."""
        if isinstance(item, cls):
            name = item.name
            return hasattr(cls, name)

        # assume item is str
        return hasattr(cls, item)


class BaseParse(metaclass=MetaParse):
    """Parse base class."""

    def __init__(
        self,
        name: Optional[str] = None,
    ):
        self.name = name

    @classmethod
    def register_builtin_parse_fn(cls, fn: Callable):
        """Registers a built-in parse function"""
        cls.PARSE_FUNCTION_REGISTRY[fn.__name__] = multidispatch(fn)
        return fn

    @classmethod
    def get_builtin_parse_fn(cls, name: str):
        """Gets a built-in parse function"""
        return cls.PARSE_FUNCTION_REGISTRY[name]

    @classmethod
    def from_builtin_parse_name(
        cls,
        name: str,
        init_kwargs,
        **parse_kwargs,
    ):
        """Create a Parse object from a built-in parse's name."""
        kws = {**init_kwargs, **parse_kwargs}

        return cls(
            cls.get_builtin_parse_fn(name),
            **kws,
        )

    @classmethod
    def register_backend(cls, type_: Type, backend: Type[BaseParseBackend]):
        """Register a backend for the specified type."""
        cls.BACKEND_REGISTRY[(cls, type_)] = backend

    @classmethod
    def get_backend(cls, parse_obj: Any) -> Type[BaseParseBackend]:
        """Get the backend associated with the type of ``parse_obj`` ."""
        parse_obj_cls = type(parse_obj)
        classes = inspect.getmro(parse_obj_cls)
        for _class in classes:
            try:
                return cls.BACKEND_REGISTRY[(cls, _class)]
            except KeyError:
                pass
        raise KeyError(
            f"Backend not found for class: {parse_obj_cls}. Looked up the " f"following base classes: {classes}"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented

        are_parse_fn_objects_equal = self._get_parse_fn_code() == other._get_parse_fn_code()

        try:
            are_strategy_fn_objects_equal = all(
                getattr(self.__dict__.get("strategy"), attr) == getattr(other.__dict__.get("strategy"), attr)
                for attr in ["func", "args", "keywords"]
            )
        except AttributeError:
            are_strategy_fn_objects_equal = True

        are_all_other_parse_attributes_equal = {
            k: v for k, v in self.__dict__.items() if k not in ["_parse_fn", "strategy"]
        } == {k: v for k, v in other.__dict__.items() if k not in ["_parse_fn", "strategy"]}

        return are_parse_fn_objects_equal and are_strategy_fn_objects_equal and are_all_other_parse_attributes_equal

    def _get_parse_fn_code(self):
        parse_fn = self.__dict__["_parse_fn"]
        try:
            code = parse_fn.__code__.co_code
        except AttributeError:
            # try accessing the functools.partial wrapper
            code = parse_fn.func.__code__.co_code

        return code

    def __hash__(self) -> int:
        return hash(self._get_parse_fn_code())

    def __repr__(self) -> str:
        return f"<Parse {self.name}>"
