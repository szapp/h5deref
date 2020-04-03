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
    elif valasarr.dtype.kind == 'U' and transp:
        val = np.atleast_1d(val)
        val = val.view(np.uint32).reshape(*val.shape, -1).astype('uint16')
        par.create_dataset(key, data=val.T, **kwargs)
        par[key].attrs['MATLAB_class'] = b'char'
        par[key].attrs['MATLAB_int_decode'] = 2
    elif valasarr.dtype.kind in ('O', 'U'):
        shape = (len(val), 1) if transp else (len(val),)
        rf = par.create_dataset(key, shape=shape, dtype=h5py.ref_dtype,
                                **kwargs)
        for i, v in enumerate(val):
            refs = par.file.require_group('#refs#')
            incr = str(len(refs.items()))
            _sortinto(refs, incr, v, transp, **kwargs)
            rf[i] = refs[incr].ref
    else:
        kwargs_child = kwargs.copy()

        # Remove filters for scalers
        if valasarr.size == 1:
            kwargs_child.update({'chunks': None, 'compression': None,
                                 'compression_opts': None, 'shuffle': None,
                                 'fletcher32': None, 'scaleoffset': None})

        if transp:
            dt = 'uint8' if valasarr.dtype == 'bool' else None
            par.create_dataset(key, data=np.atleast_2d(val).T, dtype=dt,
                               **kwargs_child)
            _setmatlabtype(par[key], val)
        else:
            par.create_dataset(key, data=val, **kwargs_child)


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
        _sortarray(par, key, [val.start, val.stop, val.step], transp, **kwargs)
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
    elif transp:
        _sortarray(par, key, np.asarray(val), transp, **kwargs)
    else:
        par[key] = val


def _setmatlabtype(obj, val):
    """Set MATLAB attributes"""
    dt = np.asarray(val).dtype

    if dt in ('float32', 'float16'):
        obj.attrs['MATLAB_class'] = b'single'
    elif dt == 'float64':
        obj.attrs['MATLAB_class'] = b'double'
    elif dt == 'bool':
        obj.attrs['MATLAB_class'] = b'logical'
        obj.attrs['MATLAB_int_decode'] = 1
    else:
        obj.attrs['MATLAB_class'] = dt.name.encode()


def _fixmatlabstruct(fp):  # noqa: C901
    """Verify MATLAB structs: It cannot load mixed non-scalar structs"""
    groups = []

    def collectgroups(name, obj):
        if isinstance(obj, h5py._hl.group.Group) and name != '/#refs#':
            groups.append(obj)
    fp.visititems(collectgroups)

    for group in groups:
        # Iterate over all immediate children to check for refs
        for child in group.values():
            if (isinstance(child, h5py._hl.dataset.Dataset) and
                    child.dtype == h5py.h5r.Reference):
                break
        else:
            continue

        # If there is a reference, turn all into references
        refs = fp.require_group('#refs#')
        for childname, child in group.items():
            if not isinstance(child, h5py._hl.dataset.Dataset):
                continue
            if child.dtype == h5py.h5r.Reference:
                continue

            # Create a new dataset (create_dataset_like does not work)
            rf = group.create_dataset('__h5dereftemp__', shape=child.shape,
                                      dtype=h5py.ref_dtype)

            # Differentiate scalar and non-scalar datasets
            if child.shape:
                for i, v in enumerate(child):
                    incr = str(len(refs.items()))
                    refs[incr] = v
                    rf[i] = refs[incr].ref
            else:
                incr = str(len(refs.items()))
                refs[incr] = child
                rf[()] = refs[incr].ref

            # Update the group-child relationship
            del group[childname]
            group[childname] = group['__h5dereftemp__']
            del group['__h5dereftemp__']


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

        if transpose:
            _fixmatlabstruct(fp)
