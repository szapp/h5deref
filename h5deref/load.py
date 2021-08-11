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
                # Recover corresponding native type
                dtt = obj.attrs.get('MATLAB_class', b'')
                dtt = np.sctypeDict.get(dtt.decode(),
                                        {b'logical': 'bool'}.get(dtt))
                obj = np.empty(0, dtype=dtt)
    else:
        tp = None

    # Recover NoneType
    if isinstance(obj, h5py._hl.dataset.Dataset):
        if obj.shape is None:
            obj = None
        elif obj.attrs.get('MATLAB_class') == b'char':
            if obj.ndim == 2 and obj.shape[1] == 1:
                obj = ''.join(map(chr, obj[:, 0]))
            else:
                obj = [''.join(map(chr, ll)) for ll in obj[()].T]

    # Recurse into datasets and groups
    if isinstance(obj, h5py._hl.dataset.Dataset):
        islogical = obj.attrs.get('MATLAB_class') == b'logical'

        # Copy, transpose and squeeze dimensions of numpy array
        if kwargs.get('transpose'):
            if obj.ndim == 2 and 1 in obj.shape:
                obj = np.squeeze(obj)
            else:
                obj = obj[()].T
        else:
            obj = obj[()]

        if isinstance(obj, (np.ndarray, np.generic)):
            # Strip h5py reference dtype
            if obj.dtype == 'O' and obj.dtype.hasobject:
                obj = obj.view('object')

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
            for key in np.atleast_1d(kwargs.get('keys', fullpath)).tolist():
                key = '/' + key.lstrip('/') + '/'
                key = key[:len(fullpath) + 1]
                if (fullpath + '/').startswith(key):
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
                        a, b = b, None
                elif a.dtype == 'O' and a.shape:
                    af = a.ravel()  # Lazy way of np.nditer
                    if all([isinstance(b, (np.ndarray, np.generic))
                            for b in af]):
                        # Collapse nested structured arrays
                        c_names = set([b.dtype.names for b in af])
                        if len(set([(b.shape, b.dtype) for b in af])) == 1:
                            # All elements share dtype and shape
                            a = np.stack(a)
                        elif c_names != {None} and len(c_names) == 1:
                            # Elements share dtype by different shape
                            c_names = c_names.pop()
                            c_shps = [set([b[n].shape for b in af])
                                      for n in c_names]
                            c_dt = [np.result_type(*[b[n] for b in af])
                                    for n in c_names]

                            # Contract differing dtypes to object
                            dtt = []
                            for n, s, t in zip(c_names, c_shps, c_dt):
                                if len(s) != 1:
                                    dtt.append((n, 'O'))
                                else:
                                    dtt.append((n, t, s.pop()))

                            # Construct and fill new array
                            b = np.empty_like(a, dtt)
                            fi = np.nditer((a, b), flags=['refs_ok'],
                                           op_flags=['readwrite'])
                            for ia, ib in fi:
                                for n in c_names:
                                    ib[()][n] = ia[()][n]
                            a, af, b = b, None, None
                if (np.prod(a.shape) * np.dtype(a.dtype).itemsize
                        > np.iinfo(np.int32).max):
                    # Wrap arrays with too large shape (numpy cannot handle)
                    b = np.empty(shape=(), dtype='O')  # No shape
                    b[()] = a
                    a, b = b, None

                dt.append((name, a.dtype, a.shape))

            arrs.append(a)

        # Collect into record array or dict
        if kwargs.get('dict') or tp == 'dict':
            obj = dict(zip(dt, arrs))
        else:
            # Check for common dimensions
            dims = [a.shape for a in arrs]
            idx = 0
            for d in zip(*dims):
                if len(set(d)) != 1:
                    break
                idx += 1

            # Are there shared dimensions between all arrays?
            if idx and len(dims) > 1:
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
        if isinstance(obj, (np.ndarray, np.generic)):
            obj = obj.tolist()
        elif isinstance(obj, str):
            obj = list([obj])
        else:
            obj = list(obj)
    elif tp == 'tuple':
        if isinstance(obj, str):
            obj = tuple([obj])
        else:
            obj = tuple(obj)
    elif tp == 'range':
        obj = range(*obj)
    elif tp == 'slice':
        obj = slice(*obj)
    elif tp == 'NoneType':
        obj = None

    return obj
