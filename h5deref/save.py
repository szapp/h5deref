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
        if valasarr.size < 2:
            kwargs = _removescalarfilters(kwargs)

        if transp:
            if valasarr.size == 0:
                _sortarray(par, key, np.zeros(2, dtype='uint64'))
                _setmatlabtype(par[key], 0)
            else:
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
    key = str(key)
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
        """Callback function to collect all suitable struct groups"""
        if (isinstance(obj, h5py._hl.group.Group)
                and name != '#refs#'
                and obj.attrs.get('MATLAB_class', None) != b'struct'):
            groups.append(obj)

    def dynamiciterator():
        """Dynamically reassessing groups iterator"""
        while True:
            fp.visititems(collectgroups)
            if groups:
                yield groups[-1]  # Start with last
            else:
                raise StopIteration

    # Iterate over all groups to make them MATLAB compatible structs
    for group in dynamiciterator():
        groups = []  # Reset groups for iterator
        group.attrs['MATLAB_class'] = np.bytes_('struct')

        # Create struct fields
        fieldnames = np.empty(len(group.keys()),
                              dtype=h5py.vlen_dtype(np.dtype('|S1')))
        fieldnames[:] = [np.fromiter(f, '|S1') for f in group.keys()]
        group.attrs['MATLAB_fields'] = fieldnames

        # Recurse into groups to obtain shape (visititems not suitable)
        def groupshape(obj):
            """Determine common shape"""
            if isinstance(obj, h5py._hl.group.Group):
                # Collect shapes from children
                dims = [groupshape(chld) for chld in obj.values()]

                # Obtain first n common dimensions
                commondim = ()
                for d in zip(*dims):
                    if len(set(d)) != 1:
                        break
                    commondim += (d[0],)

                # Pass upward
                return commondim
            else:
                if obj.ndim == 2 and obj.shape[1] == 1:
                    return (obj.shape[0],)
                else:
                    # Reversed, because MATLAB transposes
                    return obj.shape[::-1]

        # Iterate over all children to determine if it should be scalar
        commondim = groupshape(group)
        idx = len(commondim)
        if len(commondim) == 1:
            commondim += (1,)

        # Different shapes = non-scalar: nothing to do
        if not idx:
            for child in group.values():
                if not isinstance(child, h5py.h5r.Reference):
                    continue

                # One-sized references can just be resolved into group
                if child.size == 1:
                    childname = child.name
                    del fp[child.name]
                    group.move(fp[child[()].item()].name, childname)
                else:
                    # Object arrays might need to be cell arrays
                    child.attrs['MATLAB_class'] = np.bytes_('cell')
            continue

        # Turn all children into references to make it non-scalar
        refs = fp.require_group('#refs#')

        # Simple loop over all group items. Assumes there are no more
        # groups within this group that haven't been resolved already.
        # Reshape a dataset/group/reference and turn it into reference
        for childname, child in group.items():
            # Skip references with correct shape
            if (getattr(child, 'dtype', None) == h5py.h5r.Reference
                    and getattr(child, 'shape', ()) == commondim):
                print('nothing')
                continue

            # Create a new dataset without any filters
            rf = group.create_dataset('__h5dereftemp__', shape=commondim,
                                      dtype=h5py.ref_dtype)

            # Iterate over dataset entries
            fi = np.nditer(rf, flags=['refs_ok', 'multi_index'],
                           itershape=commondim)
            ndim = len(commondim)

            if isinstance(child, h5py._hl.dataset.Dataset):
                for _ in fi:
                    # Obtain index for dataset
                    if child.ndim == 2 and child.shape[1] == 1:
                        index = fi.multi_index[:ndim-1] + (Ellipsis,)
                    else:
                        index = ((Ellipsis,)*(ndim > 0) +
                                 fi.multi_index[:ndim-1])

                    # Differentiate between data and reference
                    if child.dtype == h5py.h5r.Reference:
                        v = np.atleast_2d(fp[child.name][index]).T
                    else:
                        v = np.atleast_2d(child[index]).T

                    # Create new dataset for each element with filters
                    incr = str(len(refs.items()))
                    refs.create_dataset_like(incr, child, shape=v.shape,
                                             chunks=None, maxshape=None)
                    refs[incr][()] = v
                    for atr_key, atr_val in child.attrs.items():
                        refs[incr].attrs[atr_key] = atr_val
                    rf[fi.multi_index] = refs[incr].ref
            else:
                for _ in fi:
                    # Create new group for each split
                    incr = str(len(refs.items()))
                    refs.create_group(incr)

                    for ckdname, ckd in child.items():
                        # Leave it like this, until needed
                        if isinstance(ckd, h5py._hl.group.Group):
                            raise NotImplementedError('Nested group')

                        # Store datasets
                        if ckd.ndim == 2 and ckd.shape[1] == 1:
                            index = fi.multi_index[:ndim-1] + (Ellipsis,)
                        else:
                            index = ((Ellipsis,)*(ndim > 0)
                                     + fi.multi_index[:ndim-1])

                        if ckd.dtype == h5py.h5r.Reference:
                            v = np.atleast_2d(fp[ckd.name][index]).T
                            refs[incr][ckdname] = v
                        else:
                            v = np.atleast_2d(ckd[index]).T
                            refs[incr].create_dataset_like(ckdname, ckd,
                                                           shape=v.shape,
                                                           chunks=None,
                                                           maxshape=None)
                            refs[incr][ckdname][()] = v
                            for atr_key, atr_val in ckd.attrs.items():
                                refs[incr][ckdname].attrs[atr_key] = atr_val

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
