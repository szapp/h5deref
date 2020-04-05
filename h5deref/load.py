"""
Reading functions
"""
import h5py
import numpy as np

__all__ = [
    'load',
]


def load(fp, obj=None, **kwargs):  # noqa: C901
    """
    Recursive function to collect data from a HDF5 file with references

    Parameters
    ----------
    fp : h5py._hl.files.File or str
        Open H5 file handle or valid file path

    obj : optional
        Object to possibly dereference. If not provided, `fp` will be
        traversed instead. Default is None

    Other Parameters
    ----------------
    dict : bool, optional
        Return dict instead of numpy.recarray if True. Default is False

    transpose : bool, optional
        Transpose the data structures. This is useful if the file was
        created with MATLAB. If the function is called with `fp` as str
        this will be set automatically from the file extension (mat).
        Default is False

    keys : list of str, optional
        Only load the groups/datasets with the specified names. The
        complete path (including '/') is expected. Default is None

    Returns
    -------
    obj : numpy.recarray or dict
        Record array or dict containing the entire (copied) structure

    Notes
    -----
    The content is set to be retrieved following the order in which
    they were originally added (tracking order) if this was enabled. If
    `keys` is supplied, the retrieval is set to alpha-numerical order
    instead to enable proper variable assignment on unpacking.

    The objects within the root of the file are not ordered in HDF5 and
    are therefore always returned in alpha-numerical order.
    """
    # Open file from file path
    if not isinstance(fp, h5py._hl.files.File):
        fp = str(fp)
        if fp.lower().endswith('.mat') and 'transpose' not in kwargs:
            kwargs['transpose'] = True
        with h5py.File(fp, mode='r') as f:
            obj = load(f, obj, **kwargs)
        return obj

    # Start at root of file
    if obj is None:
        obj = fp

    # Dereference H5 objects
    if isinstance(obj, h5py.h5r.Reference):
        obj = fp[obj]

    # Restore Python specific data type attribute
    if isinstance(obj, (h5py._hl.dataset.Dataset, h5py._hl.group.Group)):
        tp = obj.attrs.get('type')
        if obj.attrs.get('MATLAB_empty'):
            if obj.attrs.get('MATLAB_class') == b'char':
                obj = ''
            else:
                obj = np.empty(0)
    else:
        tp = None

    # Recover NoneType
    if isinstance(obj, h5py._hl.dataset.Dataset):
        if obj.shape is None:
            obj = None
        elif obj.attrs.get('MATLAB_class') == b'char':
            if obj.shape[1] == 1:
                obj = ''.join(map(chr, obj[()]))
            else:
                obj = [''.join(map(chr, ll)) for ll in obj[()].T]

    # Recurse into datasets and groups
    if isinstance(obj, h5py._hl.dataset.Dataset):
        islogical = obj.attrs.get('MATLAB_class') == b'logical'

        # Copy, transpose and squeeze dimensions of numpy array
        obj = np.squeeze(obj).T if kwargs.get('transpose') else np.squeeze(obj)

        # Recurse into data set
        if obj.dtype == 'O' and obj.size:
            fi = np.nditer(obj, flags=['refs_ok'], op_flags=['readwrite'])
            for it in fi:
                it[()] = load(fp, it[()], **kwargs)

        # MATLAB developers are not aware of the bool type in H5
        if islogical:
            obj = obj.astype('bool')

        # Use a single object directly
        if obj.size == 1:
            obj = obj[()]

    elif isinstance(obj, (h5py._hl.group.Group, h5py._hl.files.File)):

        # Retrieve in tracking order or alpha numerical order
        if 'keys' in kwargs:
            # Alpha-numerical order
            fields = sorted(obj.keys())
            fields = {f: obj[f] for f in fields}
        elif 'MATLAB_fields' in obj.attrs:
            # Tracked order: special for MATLAB files (due to save.py)
            fields = obj.attrs.get('MATLAB_fields')
            fields = [''.join(i.astype(str)) for i in fields]
            fields = dict(zip(fields, [obj[f] for f in fields]))
        else:
            # Tracked order (if any)
            fields = obj

        # Only traverse into requested keys (if specified)
        path = kwargs.get('_path', '') + '/'
        items = dict()
        for name, val in fields.items():
            fullpath = path + name

            # Exclude reference group always
            if fullpath == '/#refs#':
                continue

            # Add current item if specified or if no specifications
            for key in list(kwargs.get('keys', fullpath)):
                key = ('/' + key.lstrip('/'))[:len(fullpath)]
                if fullpath.startswith(key):
                    items[name] = val
                    break

        if len(items) == 0:
            return None

        # Recurse into groups and maintain their names for indexing
        arrs = []
        dt = []
        kwargs_child = kwargs.copy()
        for name, it in items.items():
            kwargs_child['_path'] = path + name
            a = load(fp, it, **kwargs_child)

            # Differentiate between dict and np.recarray
            if kwargs.get('dict') or tp == 'dict':
                dt.append(name)
            else:
                # Save python lists and other objects from greedy numpy
                if not isinstance(a, (np.ndarray, np.generic)):
                    if not it.attrs.get('type'):
                        a = np.array(a)
                    else:
                        b = np.empty(shape=(), dtype='O')  # No shape
                        b[()] = a
                        a = b
                if (np.prod(a.shape) * np.dtype(a.dtype).itemsize
                        > np.iinfo(np.int32).max):
                    # Wrap arrays with too large shape (numpy cannot handle)
                    b = np.empty(shape=(), dtype='O')  # No shape
                    b[()] = a
                    a = b

                dt.append((name, a.dtype, a.shape))

            arrs.append(a)

        # Collect into record array or dict
        if kwargs.get('dict') or tp == 'dict':
            obj = dict(zip(dt, arrs))
        else:
            # Check for common dimensions
            dims = [a.shape for a in arrs]
            idx = [i for i, j in enumerate(zip(*dims)) if len(set(j)) != 1]

            # Are there shared dimensions between all arrays?
            if idx or (not idx and len(set(map(len, dims))) == 1):
                # Collapse dimensions to parent
                idx = idx[0] if idx else len(dims[0])

                # Remove common dimensions from elements
                for i in range(len(dt)):
                    shape = dt[i][2][idx:]  # Cut off common dimensions
                    dt[i] = dt[i][:2] + (shape,)  # Stitch back together

                # Create empty array with first x common dimensions
                obj = np.empty(dims[0][:idx],
                               dtype=np.dtype(dt)).view(np.recarray)

                # Fill data to match the common dimensions
                for arr, d in zip(arrs, dt):
                    name = d[0]
                    obj[name] = arr
            else:
                # Size-less record
                obj = np.rec.fromarrays(arrs, dtype=np.dtype(dt))

    # Resolve Python specific types
    if tp == 'list':
        obj = (obj.tolist() if isinstance(obj, (np.ndarray, np.generic))
               else list(obj))
    elif tp == 'tuple':
        obj = tuple(obj)
    elif tp == 'range':
        obj = range(*obj)
    elif tp == 'slice':
        obj = slice(*obj)
    elif tp == 'NoneType':
        obj = None

    return obj
