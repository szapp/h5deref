"""
h5deref
=======

Save and load HDF5 files (h5 file extension) and resolve all
references recursively into suitable formats.

Usage
=====

Usage is straight forward for loading and saving files.

Saving
------

To write the variables `var`, `var2` and `var3` to a new file, write

>>> import h5deref
>>> h5deref.save('/file/path.h5', {'a': var, 'b': var2, 'c': var3})

Additional h5py options like compression can be passed to the datasets.

>>> import h5deref
>>> h5deref.save('/file/path.h5', {'a': var}, compression='gzip')

Loading
-------

All content from a file is loaded into a structured numpy array with

>>> import h5deref
>>> data = h5deref.load('/file/path.h5')

The content can alternatively be loaded into a Python dictionary.

>>> import h5deref
>>> data = h5deref.load('/file/path.h5', dict=True)
>>> data.keys()
dict_keys(['a', 'b', 'c'])

To speed up loading, individual keys can be specified to be loaded only.

>>> import h5deref
>>> data = h5deref.load('/file/path.h5', keys=['/a', '/c/subname'])
>>> data.dtype.names
('a', 'c')
"""
from .save import save
from .load import load

__version__ = '0.1.0'

__all__ = [
    'save',
    'load',
]
