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
        rf = par.create_group(key, track_order=True)
        for k in val.dtype.names:
            _sortinto(rf, k, val[k], transp, **kwargs)
    elif valasarr.dtype.kind == 'U' and transp:
        val = np.atleast_1d(val)
        val = val.view(np.uint32).reshape(*val.shape, -1).astype('uint16')
        par.create_dataset(key, data=val.T, **kwargs)
        if transp:
            _setmatlabtype(par[key], 'U')  # valasarr.dtype not working
    elif valasarr.dtype.kind in ('O', 'U'):
        if transp:
            valasarr = np.atleast_2d(valasarr).T
        rf = par.create_dataset(key, shape=valasarr.shape,
                                dtype=h5py.ref_dtype,
                                **_removescalarfilters(kwargs))
        refs = par.file.require_group('#refs#')
        fi = np.nditer(valasarr, flags=['refs_ok', 'multi_index'],
                       itershape=valasarr.shape)
        for v in fi:
            incr = str(len(refs.items()))
            _sortinto(refs, incr, v[()], transp, **kwargs)  # Filters
            rf[fi.multi_index] = refs[incr].ref
        if transp and valasarr.dtype.kind == 'U':
            _setmatlabtype(par[key], valasarr.dtype)
    else:
        if valasarr.size == 1:
            kwargs = _removescalarfilters(kwargs)

        if transp:
            dt = 'uint8' if valasarr.dtype == 'bool' else None
            par.create_dataset(key, data=np.atleast_2d(val).T, dtype=dt,
                               **kwargs)
            _setmatlabtype(par[key], valasarr.dtype)
        else:
            par.create_dataset(key, data=val, **kwargs)


def _sortdict(par, key, val, transp=False, **kwargs):
    """Add dict and recurse into it"""
    p = par.create_group(key, track_order=True)
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
            _setmatlabtype(par[key], 0)
        else:
            _sortarray(par, key, val, transp, **kwargs)
        par[key].attrs['type'] = 'list'
        if transp:
            _setmatlabtype(par[key], 'O')  # Cell array
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
        if transp:
            # Closest to NoneType in MATLAB is empty list []
            _sortarray(par, key, np.zeros(2, dtype='uint64'))
            _setmatlabtype(par[key], 0)
        else:
            par.create_dataset(key, dtype='bool')  # Empty data set
        par[key].attrs['type'] = 'NoneType'
    elif isinstance(val, str) and transp:
        val = np.fromiter(map(ord, val), dtype='uint16')
        _sortarray(par, key, val, transp, **kwargs)
        _setmatlabtype(par[key], np.dtype(np.str_))
    elif transp:
        _sortarray(par, key, np.asarray(val), transp, **kwargs)
    else:
        par[key] = val


def _removescalarfilters(filters):
    """Remove any filter not supported by scalar datasets"""
    filters_updated = filters.copy()
    filters_updated.update({'chunks': None, 'compression': None,
                            'compression_opts': None, 'shuffle': None,
                            'fletcher32': None, 'scaleoffset': None})
    return filters_updated


def _setmatlabtype(obj, dt):
    """Set MATLAB attributes"""
    if not dt:  # Empty dataset
        obj.attrs['MATLAB_class'] = np.bytes_('double')
        obj.attrs['MATLAB_empty'] = np.uint8(1)
    elif dt in ('float32', 'float16'):
        obj.attrs['MATLAB_class'] = np.bytes_('single')
    elif dt == 'float64':
        obj.attrs['MATLAB_class'] = np.bytes_('double')
    elif dt == 'bool':
        obj.attrs['MATLAB_class'] = np.bytes_('logical')
        obj.attrs['MATLAB_int_decode'] = np.uint8(1)
    elif dt == 'U':
        obj.attrs['MATLAB_class'] = np.bytes_('char')
        obj.attrs['MATLAB_int_decode'] = np.uint8(2)
    elif dt == 'O':  # Only for non-numpy lists
        obj.attrs['MATLAB_class'] = np.bytes_('cell')
    else:
        obj.attrs['MATLAB_class'] = np.bytes_(dt.name)


def _fixmatlabstruct(fp):  # noqa: C901
    """Verify MATLAB structs: It cannot load mixed non-scalar structs"""
    groups = []

    def collectgroups(name, obj):
        if (isinstance(obj, (h5py._hl.files.File, h5py._hl.group.Group)) and
                name.lstrip('/') != '#refs#'):
            groups.append(obj)
    fp.visititems(collectgroups)

    for group in groups:
        group.attrs['MATLAB_class'] = np.bytes_('struct')

        # Create struct fields
        fields = ('_',) + tuple(group.keys())  # Extra element necessary
        group.attrs['MATLAB_fields'] = np.array(
            [np.fromiter(f, '|S1') for f in fields],
            dtype=h5py.vlen_dtype(np.dtype('|S1')))[1:]

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

            if child.shape is None:
                incr = str(len(refs.items()))
                group.copy(child, refs, name=incr)
                del group[childname]
                group[childname] = refs[incr].ref
            else:
                # Create a new dataset (create_dataset_like does not work)
                rf = group.create_dataset('__h5dereftemp__', shape=child.shape,
                                          dtype=h5py.ref_dtype)

                # Iterate over dataset entries
                fi = np.nditer(child, flags=['refs_ok', 'multi_index'],
                               itershape=child.shape)
                for v in fi:
                    incr = str(len(refs.items()))
                    refs.create_dataset_like(incr, child,
                                             shape=np.atleast_2d(v).shape,
                                             chunks=None, maxshape=None)
                    refs[incr][()] = v
                    for atr_key, atr_val in child.attrs.items():
                        refs[incr].attrs[atr_key] = atr_val
                    rf[fi.multi_index] = refs[incr].ref

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
    Additional parameters (`**kwargs`) can be passed to `create_dataset`
    of `h5py`. This is useful to add further options like compression.

    The order of adding the data is set to be tracked by HDF5.
    """
    # Open file from file path
    if not isinstance(fp, h5py._hl.files.File):
        fp = str(fp)
        fileargs = {'track_order': True}
        matlabfile = fp.lower().endswith('.mat')
        if matlabfile:
            transpose = transpose or True
            if not kwargs.get('compression'):
                kwargs['compression'] = 'gzip'
                kwargs['compression_opts'] = 3
            fileargs['userblock_size'] = 0x200

        with h5py.File(fp, mode='w', **fileargs) as f:
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
