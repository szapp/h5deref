"""
h5deref
=======

Load HDF5 files (h5 file extension) and dereference all contained
objects recursively and store the result into a numpy record array or
a python dictionary.
"""
from .h5deref import (
    h5deref,
    WorkaroundWarning,
)

__version__ = '0.1.0'

__all__ = [
    'h5deref',
    'WorkaroundWarning',
]
