"""Base classes for objects that inherit form Pandas Series or DataFrame."""
from typing import Optional, Type

import pandas as pd


class BaseDataFrame(pd.DataFrame):
    """Base class for objects that inherit from Pandas DataFrame.

    A same-dimensional slice of an object that inherits from this class will
    be of equivalent type (instead of being a Pandas DataFrame).

    A one-dimensional slice of an object that inherits from this class can
    be of different type, depending on whether the DataFrame-like object was
    sliced vertically or horizontally. For this, the additional properties
    ``_constructor_sliced_horizontal`` and ``_constructor_sliced_vertical`` are
    introduced to extend the functionality that is provided by Pandas'
    ``_constructor_sliced`` property. Both properties are set to
    ``pandas.Series`` by default and only need to be overwritten when
    necessary.
    """
    _internal_names = pd.DataFrame._internal_names + ['base_class_view']
    _internal_names_set = set(_internal_names)

    def __repr__(self) -> str:
        return self.base_class_view.__repr__()

    @property
    def _constructor(self) -> Type["BaseDataFrame"]:
        # TODO: do not set default?
        return self.__class__

    @property
    def _constructor_sliced(self) -> Type[pd.Series]:
        # dynamically create a subclass of _FastF1BaseSeriesConstructor that
        # has a reference to this self (i.e. the object from which the slice
        # is created) as a class property
        # type(...) returns a new subclass of a Series
        return type('_DynamicBaseSeriesConstructor',  # noqa: return type
                    (_BaseSeriesConstructor,),
                    {'__meta_created_from': self})

    @property
    def _constructor_sliced_horizontal(self) -> Type[pd.Series]:
        return pd.Series

    @property
    def _constructor_sliced_vertical(self) -> Type[pd.Series]:
        return pd.Series

    @property
    def base_class_view(self) -> pd.DataFrame:
        """For a nicer debugging experience; can view DataFrame through
        this property in various IDEs"""
        return pd.DataFrame(self)


class _BaseSeriesConstructor(pd.Series):
    """Base class for an intermediary and dynamically defined constructor
    class that implements horizontal and vertical slicing of Pandas DataFrames
    with different result objects types."""

    __meta_created_from: Optional[BaseDataFrame]

    def __new__(cls, data=None, index=None, *args, **kwargs) -> pd.Series:
        parent = getattr(cls, '__meta_created_from')
        if index is None:
            index = getattr(data, 'index', None)

        if (parent is None) or (index is None):
            constructor = pd.Series
        elif parent.index is index:
            constructor = parent._constructor_sliced_vertical
        else:
            constructor = parent._constructor_sliced_horizontal

        return constructor(data=data, index=index, *args, **kwargs)


class BaseSeries(pd.Series):
    """Base class for objects that inherit from Pandas Series.

    A same-dimensional slice of an object that inherits from this class will
    be of equivalent type (instead of being a Pandas Series).
    """
    @property
    def _constructor(self) -> Type[pd.Series]:
        return self.__class__
