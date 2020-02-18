"""
Writing functions
"""
import h5py
import numpy as np

__all__ = [
    'save'
]


def _getrefs(par):
    """Get references group"""
    return par.file.require_group('#refs#')


def _sortarray(par, key, val, **kwargs):
    """Add lists, numpy arrays, and numpy record arrays"""
    if np.asarray(val).dtype.names:
        rf = par.create_group(key)
        for k, v in zip(val.dtype.names, val):
            _sortinto(rf, k, v, **kwargs)
    if np.asarray(val).dtype == 'O':
        rf = par.create_dataset(key, shape=(len(val),), dtype=h5py.ref_dtype,
                                **kwargs)
        for i, v in enumerate(val):
            incr = len(_getrefs(par).items())
            _sortinto(_getrefs(par), str(incr), v, **kwargs)
            rf[i] = _getrefs(par)[str(incr)].ref
    else:
        par.create_dataset(key, data=val, **kwargs)


def _sortvalue(par, key, val, **kwargs):
    """Add simple values"""
    par[key] = np.empty(0, dtype='uint8') if val is None else val


def _sortdict(par, key, val, **kwargs):
    """Add dict and recurse into it"""
    p = par.create_group(key)
    for k, v in val.items():
        _sortinto(p, k, v, **kwargs)


def _sortinto(par, key, val, **kwargs):
    """Wrapper for adder functions"""
    if isinstance(val, (np.ndarray, list)):
        _sortarray(par, key, val, **kwargs)
    elif isinstance(val, dict):
        _sortdict(par, key, val, **kwargs)
    else:
        _sortvalue(par, key, val, **kwargs)


def save(fp, data, **kwargs):
    """
    Create and write a HDF5 file while formatting python data types
    automatically into suitable HDF5 format

    Parameters
    ----------
    fp : h5py._hl.files.File or str
        Open H5 file handle or valid file path

    data : dict
        Python dictionary containing key names and variables to store

    Notes
    -----
    Additional parameters (`**kwargs`) cam be passed to `create_dataset`
    of `h5py`. This is useful to add further options like compression.
    """
    # Open file from file path
    if isinstance(fp, str):
        with h5py.File(fp, mode='w') as f:
            save(f, data, **kwargs)
    else:
        for key, val in data.items():
            _sortinto(fp, key, val, **kwargs)
