"""Data validation parse definition."""

from typing import (
    Callable,
    Optional,
    TypeVar,
    Union,
)

import pandas as pd

from pandera.api.base.parses import BaseParse, ParseResult

T = TypeVar("T")


# pylint: disable=too-many-public-methods
class Parse(BaseParse):
    """Parse a data object for certain properties."""

    def __init__(
        self,
        parse_fn: Callable,
        element_wise: bool = False,
        name: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        **parse_kwargs,
    ) -> None:
        """Apply a parse function to a data object."""
        super().__init__(name=name)

        self._parse_fn = parse_fn
        self._parse_kwargs = parse_kwargs
        self.element_wise = element_wise
        self.name = name or getattr(self._parse_fn, "__name__", self._parse_fn.__class__.__name__)
        self.title = title
        self.description = description

    def __call__(
        self,
        parse_obj: Union[pd.DataFrame, pd.Series],
        column: Optional[str] = None,
    ) -> ParseResult:
        # pylint: disable=too-many-branches
        """Validate pandas DataFrame or Series.

        :param parse_obj: pandas DataFrame of Series to parse.
        :param column: for dataframe parses, apply the parse function to this
            column.
        :returns: ParseResult tuple containing:

            ``parse_output``: boolean scalar, ``Series`` or ``DataFrame``
            indicating which elements passed the parse.

            ``parse_passed``: boolean scalar that indicating whether the parse
            passed overall.

            ``parsed_object``: the parseed object itself. Depending on the
            options provided to the ``Parse``, this will be a pandas Series, a
            ``Dict[str, Series]`` where the keys are distinct groups.

            ``failure_cases``: subset of the parse_object that failed.
        """
        backend = self.get_backend(parse_obj)(self)
        return backend(parse_obj, column)
