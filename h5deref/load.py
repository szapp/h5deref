"""
Reading functions
"""
import h5py
import numpy as np
import warnings

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
    """
    # Open file from file path
    if not isinstance(fp, h5py._hl.files.File):
        fp = str(fp)
        if fp.lower().endswith('.mat') and not kwargs.get('transpose'):
            kwargs['transpose'] = True
        with h5py.File(fp, mode='r', track_order=False) as f:
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
        if kwargs.get('transpose'):
            if obj.ndim == 2 and obj.shape[1] == 1:
                obj = np.squeeze(obj, axis=1)
            else:
                obj = np.asarray(obj).T
        else:
            obj = np.asarray(obj)

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

        # Only traverse into requested keys (if specified)
        path = kwargs.get('_path', '') + '/'
        items = {name: val for name, val in obj.items()
                 if [True for i in np.array(kwargs.get('keys', [path+name]),
                                            ndmin=1, copy=False)
                     if (path+name).startswith(i[:len(path+name)])]
                 and path+name != '/#refs#'}

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
                    if not obj.attrs.get('type'):
                        a = np.array(a, ndmin=1)
                    else:
                        b = np.empty(1, 'O')
                        b[0] = a
                        a = b
                if (np.prod(a.shape) * np.dtype(a.dtype).itemsize
                        > np.iinfo(np.int32).max):
                    # Skip way too large objects (numpy cannot handle)
                    warnings.warn(f'Field \'{path+name}\' has too large '
                                  'dimensions. Skipping. Try using '
                                  '\'dict=True\'', UserWarning)
                    continue
                dt.append((name, a.dtype, a.shape))

            arrs.append(a)

        # Collect into record array or dict
        if kwargs.get('dict') or tp == 'dict':
            obj = dict(zip(dt, arrs))
        else:
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
