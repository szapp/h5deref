h5deref
=======

Save and load HDF5 files (h5 file extension) and resolve all references
recursively into suitable formats.

Complicated, nested structures like dictionaries or structured numpy arrays may
be saved as they come and will be rebuild the same way on loading. Python data
types like dicts, lists, tuples, ranges or slices are maintained and loaded
exactly as they are saved.

Usage
=====

Usage is straight forward for loading and saving files.

Saving
------

To write the variables `var`, `var2` and `var3` to a new file, write

```python
import h5deref
h5deref.save('/file/path.h5', {'a': var, 'b': var2, 'c': var3})
```

Additional h5py options like compression can be passed to the datasets.

```python
import h5deref
h5deref.save('/file/path.h5', {'a': var}, compression='gzip')
```

Loading
-------

All content from a file is loaded into a structured numpy array with

```python
import h5deref
data = h5deref.load('/file/path.h5')
data.var2.nested.variables  # Access easily through numpy recarray
```

The content can alternatively be loaded into a Python dictionary.

```python
import h5deref
data = h5deref.load('/file/path.h5', dict=True)
data.keys()  # dict_keys(['a', 'b', 'c'])
```

To speed up loading, individual keys can be specified to be loaded only.

```python
import h5deref
data = h5deref.load('/file/path.h5', keys=['/a', '/c/subname'])
data.dtype.names  # ('a', 'c')
```
These can be written into different variables directly. They are unpacked in
alpha-numerical order, not in order of specification.

```python
import h5deref
a, c = h5deref.load('/file/path.h5', keys=['/a', '/c/subname'])[()]
```

MAT Files
---------

MATLAB files (mat file extension) saved with v7.3 are HDF5 files under the hood.
This packages understands (most of) MATLAB's data types and enables to load and
write them as well, while correctly transposing the data (by identifying MATLAB
files by their file extension).

These MATLAB files may thus function as convenient interface to share heavy
MATLAB structs or python dictionaries/structured numpy arrays between python and
MATLAB without any data conversion or complicated loading scripts.

```python
from h5deref import save as h5save, load as h5load  # Convenient access
data = h5load('/file/path.mat').data
# ...
h5save('/file/path.mat', {'data': data})
```
On loading, singleton dimensions are squeezed for easier indexing in python.
Saving MATLAB files will append singleton dimensions to the beginning of
one-dimensional arrays for compatibility with MATLAB. When saving a previously
loaded MATLAB file, it is unavoidable that one-dimensional matrices may end up
transposed, i.e. `(432, 1) ---loading---> (432,) ---saving---> (1, 432)`. For
indexing in MATLAB, this does not make a difference.

Support for MATLAB data types is implemented on an as-needed basis. Loading and
saving further data types may be implemented in the future.
