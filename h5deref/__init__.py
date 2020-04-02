"""
h5deref
=======

Save and load HDF5 files (h5 file extension) and resolve all references
recursively into suitable formats.

Complicated, nested structures like dictionaries or structured numpy
arrays may be saved as they come and will be rebuild the same way on
loading. Python data types like dicts, lists, tuples, ranges or slices
are maintained and loaded exactly as they are saved.

Usage
=====

Usage is straight forward for loading and saving files.

Saving
------

To write the variables `var`, `var2` and `var3` to a new file, write

>>> from h5deref import h5save
>>> h5save('/file/path.h5', {'a': var, 'b': var2, 'c': var3})

Additional h5py options like compression can be passed to the datasets.

>>> from h5deref import h5save
>>> h5save('/file/path.h5', {'a': var}, compression='gzip')

Loading
-------

All content from a file is loaded into a structured numpy array with

>>> from h5deref import h5load
>>> data = h5load('/file/path.h5')
>>> data.var2.nested.variables  # Access easily through numpy recarray

The content can alternatively be loaded into a Python dictionary.

>>> from h5deref import h5load
>>> data = h5load('/file/path.h5', dict=True)
>>> data.keys()
dict_keys(['a', 'b', 'c'])

To speed up loading, individual keys can be specified to be loaded only.

>>> from h5deref import h5load
>>> data = h5load('/file/path.h5', keys=['/a', '/c/subname'])
>>> data.dtype.names
('a', 'c')

These can be written into different variables directly. They are
unpacked in alpha-numerical order, not in order of specification.

```python
from h5deref import h5load
a, c = h5load('/file/path.h5', keys=['/a', '/c/subname'])[()]
```

For more information, see the readme.
"""
from .save import save as h5save
from .load import load as h5load

__version__ = '0.1.0'

__all__ = [
    'h5save',
    'h5load',
]
