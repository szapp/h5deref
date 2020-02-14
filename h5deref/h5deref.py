"""
Core functions
"""
import h5py
import numpy as np
import warnings

__all__ = [
    'h5deref',
    'WorkaroundWarning',
]


class WorkaroundWarning(UserWarning):
    """
    Issued by `h5deref` when assuming an empty data set.
    This is a work around to detect zero size arrays created by MATLAB.
    """
    pass


def h5deref(fp, obj=None, **kwargs):  # noqa: C901
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

    workaround : bool, optional
        Empty arrays created by MATLAB are stored as zero-valued, one-
        dimensional arrays of size two `[0, 0]` in HDF5 and cannot be
        uniquely recognized. If set to True, such arrays will be
        detected and replaced by empty arrays of size zero when loading.
        Default is True

    Returns
    -------
    obj : numpy.recarray or dict
        Record array or dict containing the entire (copied) structure

    Warns
    -----
        When assuming an empty data set. This is a work around to
        correctly recognize zero size arrays created by MATLAB. This
        work around can be disabled with `workaround=False`.

        The warnings can be turned off by

        >>> import warnings
        >>> warnings.simplefilter('ignore', h5deref.WorkaroundWarning)

    """
    # Open file from file path
    if isinstance(fp, str):
        if fp.lower().endswith('.mat') and 'transpose' not in kwargs:
            kwargs['transpose'] = True
        with h5py.File(fp, mode='r') as f:
            obj = h5deref(f, obj, **kwargs)
        return obj

    # Start at root of file
    if obj is None:
        obj = fp

    # Dereference H5 objects
    if isinstance(obj, h5py.h5r.Reference):
        obj = fp[obj]

    # Recurse into datasets and groups
    if isinstance(obj, h5py._hl.dataset.Dataset):
        # Special case for empty variables from MATLAB
        if np.array_equal(obj, np.zeros(2, int)) and kwargs.get('workaround',
                                                                True):
            warnings.warn(f'Assuming empty array in {kwargs.get("_path")}',
                          WorkaroundWarning, stacklevel=2)
            obj = np.empty(0, dtype='uint8')

        # Copy, transpose and squeeze dimensions of numpy array
        obj = np.squeeze(obj).T if kwargs.get('transpose') else np.squeeze(obj)

        # Recurse into data set
        if obj.dtype == 'O':
            fi = np.nditer(obj, flags=['refs_ok'], op_flags=['readwrite'])
            for it in fi:
                it[()] = h5deref(fp, it[()], **kwargs)

        # Use a single object directly
        if obj.size == 1:
            obj = obj[()]

    elif isinstance(obj, (h5py._hl.group.Group, h5py._hl.files.File)):

        # Only traverse into requested keys (if specified)
        path = kwargs.get('_path', '') + '/'
        items = {name: val for name, val in obj.items()
                 if [True for i in kwargs.get('keys', [path+name])
                     if i.startswith(path+name)] and path+name != '/#refs#'}

        if len(items) == 0:
            return None

        # Recurse into groups and maintain their names for indexing
        arrs, names, dim = [], [], set()
        kwargs_child = kwargs.copy()
        for name, it in items.items():
            kwargs_child['_path'] = path + name
            names.append(name)
            a = np.empty(1, dtype='O')
            a[0] = h5deref(fp, it, **kwargs_child)
            dim.add(len(a[0]) if '__len__' in dir(a[0]) else 1)
            arrs.append(a)

        # Collapse if same dimensions
        if len(dim) == 1:
            arrs = np.array([a[0] for a in arrs])

        # Collect into record array or dict
        if kwargs.get('dict'):
            obj = dict(zip(names, arrs))
        else:
            obj = np.rec.fromarrays(arrs, names=names)

    return obj
