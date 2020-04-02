"""
Writing functions
"""
import h5py
import numpy as np

__all__ = [
    'save'
]


def _sortarray(par, key, val, transp=False, **kwargs):  # noqa: C901
    """Add lists, numpy arrays, and numpy record arrays"""
    valasarr = np.asarray(val)
    if valasarr.dtype.names:
        rf = par.create_group(key)
        for k in val.dtype.names:
            _sortinto(rf, k, val[k], transp, **kwargs)
        if transp:
            rf.attrs['MATLAB_class'] = b'struct'

            # Create struct fields
            fields = ('_',) + val.dtype.names  # Extra element necessary
            rf.attrs['MATLAB_fields'] = np.array(
                [np.array([c.encode() for c in f]) for f in fields],
                dtype=h5py.vlen_dtype(np.dtype('|S1')))[1:]

            # Create canonical empty entry 'a'
            refs = par.file.require_group('#refs#')
            can = refs.require_dataset('a', shape=(2,), dtype='uint64',
                                       exact=True)
            can.attrs['MATLAB_empty'] = 1
            can.attrs['MATLAB_class'] = b'canonical empty'

    elif valasarr.dtype.kind in 'O':
        shape = (len(val), 1) if transp else (len(val),)
        rf = par.create_dataset(key, shape=shape, dtype=h5py.ref_dtype,
                                **kwargs)
        for i, v in enumerate(val):
            refs = par.file.require_group('#refs#')
            incr = len(refs.items())
            _sortinto(refs, str(incr), v, transp, **kwargs)
            rf[i] = refs[str(incr)].ref
    elif valasarr.dtype.kind == 'U' and transp:
        val = np.atleast_1d(val)
        val = val.view(np.uint32).reshape(*val.shape, -1).astype('uint16')
        par.create_dataset(key, data=val.T, **kwargs)
        par[key].attrs['MATLAB_class'] = b'char'
        par[key].attrs['MATLAB_int_decode'] = 2
    else:
        if transp:
            dt = 'uint8' if valasarr.dtype == 'bool' else None
            par.create_dataset(key, data=np.atleast_2d(val).T, dtype=dt,
                               **kwargs)
            dt = valasarr.dtype
            if valasarr.dtype == 'float32':
                par[key].attrs['MATLAB_class'] = b'single'
            elif valasarr.dtype == 'float64':
                par[key].attrs['MATLAB_class'] = b'double'
            elif valasarr.dtype == 'bool':
                par[key].attrs['MATLAB_class'] = b'logical'
                par[key].attrs['MATLAB_int_decode'] = 1
            else:
                par[key].attrs['MATLAB_class'] = dt.name.encode()
        else:
            par.create_dataset(key, data=val, **kwargs)

    if transp and isinstance(par[key], h5py._hl.dataset.Dataset):
        par[key].attrs['H5PATH'] = ('/'+par.name.replace('/', '')).encode()


def _sortdict(par, key, val, transp=False, **kwargs):
    """Add dict and recurse into it"""
    p = par.create_group(key)
    for k, v in val.items():
        _sortinto(p, k, v, transp, **kwargs)


def _sortinto(par, key, val, transp=False, **kwargs):  # noqa: C901
    """Wrapper for adder functions"""
    if isinstance(val, (np.ndarray, np.record, np.generic)):
        _sortarray(par, key, val, transp, **kwargs)
    elif isinstance(val, dict):
        _sortdict(par, key, val, transp, **kwargs)
        par[key].attrs['type'] = 'dict'
        if transp:
            par[key].attrs['MATLAB_class'] = b'struct'
    elif isinstance(val, list):
        if len(val) == 0 and transp:
            _sortarray(par, key, np.zeros(2, dtype='uint64'))
            par[key].attrs['MATLAB_empty'] = 1
        else:
            _sortarray(par, key, val, transp, **kwargs)
            par[key].attrs['type'] = 'list'
    elif isinstance(val, tuple):
        _sortarray(par, key, val, transp, **kwargs)
        par[key].attrs['type'] = 'tuple'
    elif isinstance(val, slice):
        _sortarray(par, key, [val.start, val.stop, val.step], transp,
                   **kwargs)
        par[key].attrs['type'] = 'slice'
    elif isinstance(val, range):
        _sortarray(par, key, [val.start, val.stop, val.step], transp, **kwargs)
        par[key].attrs['type'] = 'range'
    elif isinstance(val, type(None)):
        par.create_dataset(key, dtype='bool')  # Empty data set
        par[key].attrs['type'] = 'NoneType'
        if transp:
            par[key].attrs['MATLAB_empty'] = 1
    elif isinstance(val, str) and transp:
        val = np.array(list(map(ord, val)), dtype='uint16')
        _sortarray(par, key, val, transp, **kwargs)
        par[key].attrs['MATLAB_class'] = b'char'
        par[key].attrs['MATLAB_int_decode'] = 2
    else:
        par[key] = val


def save(fp, data, transpose=None, **kwargs):
    """
    Create and write a HDF5 file while formatting python data types
    automatically into suitable HDF5 format

    Parameters
    ----------
    fp : h5py._hl.files.File or str
        Open H5 file handle or valid file path

    data : dict
        Python dictionary containing key names and variables to store

    transpose : bool, optional
        Transpose all arrays. This is useful for writing MATLAB files.
        If the function is called with `fp` as str this will be set
        automatically from the file extension (mat). Default is False

    Notes
    -----
    Additional parameters (`**kwargs`) cam be passed to `create_dataset`
    of `h5py`. This is useful to add further options like compression.
    """
    # Open file from file path
    if not isinstance(fp, h5py._hl.files.File):
        fp = str(fp)
        matlabfile = fp.lower().endswith('.mat')
        if matlabfile:
            transpose = transpose or True
            if not kwargs.get('compression'):
                kwargs['compression'] = 'gzip'
                kwargs['compression_opts'] = 3
        with h5py.File(fp, mode='w', userblock_size=0x200*matlabfile) as f:
            save(f, data, transpose, **kwargs)

        # Add MATLAB file support
        if matlabfile:
            with open(fp, 'r+') as f:
                f.write('MAT')
                f.seek(0x7d)
                f.write('\x02\x49\x4d')
    else:
        for key, val in data.items():
            _sortinto(fp, key, val, transpose, **kwargs)
