"""Dataset fetcher.

Authors: Mainak Jas <mjas@mgh.harvard.edu>
"""

import pooch


def fetch_hcp_sample(path=None):
    """Fetch sample HCP dataset.

    Parameters
    ----------
    path : str
        If path is provided, save data to path.
        Else the data is saved to the cache directory
        of the operating system. See return values
        to get the automatically determined path.
    
    Returns
    -------
    path : str
        The path to where the data is saved.
    """

    if path is None:
        path = pooch.os_cache('mvcmi')
    
    urls = {'aparc.a2009s:subc4-label_names.txt': 'https://osf.io/urg7e/download',
            'irmia_2012_order.txt': 'https://osf.io/kh9bj/download',
            'label_names.npz': 'https://osf.io/ajv4k/download',
            'label_ts.npz': 'https://osf.io/5kynb/download',
            'node_table_irmia_2012.txt': 'https://osf.io/8yh9d/download'
        }
    registry = {'aparc.a2009s:subc4-label_names.txt':
                '0cc724bb1ed44e9dfb59d33f143212efcfd4d005678d740a129a5d228c9dee82',
                'irmia_2012_order.txt':
                '0e4feeebd97965190f7b1d81fa600c80f3db018e33d603ff1af88b491c30b5de',
                'label_names.npz':
                '75aaf91bd499fac51152b76c6e086b9b58980d186de6b9a1ba1a894f9b00cda1',
                'node_table_irmia_2012.txt':
                '1f55c6c7c12fb19dfe497102d34ea9901043c55e24ce880436c75b42fef77db7',
                'label_ts.npz':
                '3d381cdbeca7ba0d63a22625b6b9a6b464c5e8f87fb88581858054473cfcb72e'}

    fetcher = pooch.create(path=path, base_url="https://osf.io/", urls=urls,
                           registry=registry)

    for fname in urls:
        fetcher.fetch(fname, progressbar=True)
    
    return path


def load_label_ts(fname, n_parcels=None):
    """Load the label time series.

    Parameters
    ----------
    fname : str
        The path to the label time series.
    n_parcels : int | None
        If None, keep all parcels. Else,
        keep n_parcels.

    Returns
    -------
    label_ts : list of n_parcels
        The label time series.
    """
    label_ts_fname = data_path / 'label_ts.npz'
    label_ts_load = np.load(label_ts_fname)
    keys = label_ts_load.keys()
    n_elems = len(keys)
    label_ts = [None] * n_elems
    for key in keys:
        idx = int(key.split('_')[1])
        label_ts[idx] = label_ts_load[key][:, 0:]

    if n_parcels is not None:
        label_ts = label_ts[:n_parcels]

    print("done reading in label_ts\n")
    print("%d\n" % n_elems)    

    return label_ts
