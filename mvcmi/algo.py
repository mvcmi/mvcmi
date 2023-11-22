"""
Multivariate connectivity methods

Authors: Martin Luessi
         Padmavathi Sundaram <padma@nmr.mgh.harvard.edu>
         Mainak Jas <mjas@mgh.harvard.edu>
"""

import numpy as np
from numpy.linalg import qr, slogdet

from scipy import linalg, signal

from joblib import Parallel, delayed
from sklearn.decomposition import PCA


def compute_ccoef_pca(label_ts):
    """Compute correlation coefficient."""
    # we are already in PCA space
    pca_ts = np.concatenate([ts[0][None, :] for ts in label_ts], axis=0)
    ccoef = np.corrcoef(pca_ts)
    ccoef[np.diag_indices(len(ccoef))] = 0
    return ccoef


def partial_corr_mvar(label_ts):
    """Compute partial correlation matrices."""
    print("mvcon partial_corr_mvar")
    n_signals = len(label_ts)
    label_ts_comb = np.concatenate(label_ts, axis=0)
    print(label_ts_comb.shape)
    n_signals_tot = label_ts_comb.shape[0]
    idx = np.cumsum(np.r_[[0], [len(ts) for ts in label_ts]])
    print(idx.shape)
    pcor = np.zeros((n_signals_tot, n_signals_tot), dtype=np.float32)
    for ii in range(n_signals):
        print(ii)
        # for some reason this is much faster than scipy's lstsq
        A = np.r_[label_ts_comb[:idx[ii]], label_ts_comb[idx[ii + 1]:]].T
        x = np.linalg.lstsq(A, label_ts[ii].T, rcond=None)[0]
        pcor[idx[ii]:idx[ii + 1], :idx[ii]] = x[:idx[ii]].T
        pcor[idx[ii]:idx[ii + 1], idx[ii + 1]:] = x[idx[ii]:].T

    return pcor


def cmui_from_pcor_mvar(label_ts, pcor):
    """Compute conditional MI from correlation matrices."""
    print("mvcon cmui_from_pcor_mvar")
    n_signals = len(label_ts)
    idx = np.cumsum(np.r_[[0], [len(ts) for ts in label_ts]])
    cmui = np.zeros((n_signals, n_signals))
    for ii in range(n_signals):
        for jj in range(ii):
            a = np.dot(pcor[idx[ii]:idx[ii + 1],
                            idx[jj]:idx[jj + 1]].astype(np.float64),
                       pcor[idx[jj]:idx[jj + 1],
                            idx[ii]:idx[ii + 1]].astype(np.float64))

            #            print np.linalg.det(np.eye(a.shape[0]) - a)
            cmui[ii, jj] = -0.5 * np.linalg.slogdet(np.eye(a.shape[0]) - a)[1]
            cmui[jj, ii] = cmui[ii, jj]

            if cmui[ii, jj] > 1.0:
                U, s, V = np.linalg.svd(a)
                smedian = np.median(s)
                thresh = smedian * 2.858  # from Gavish & Donoho, arXiv 2013
                n = np.sum(s > thresh)
                cmui[ii, jj] = -0.5 * np.sum(np.log(1 - s[0:n]))
                cmui[jj, ii] = cmui[ii, jj]

    return cmui


def compute_cmi(label_ts):
    """Compute conditional MI.

    Parameters
    ----------
    label_ts : list
        The label time series.
    
    Returns
    -------
    cmi : array of shape (n_labels, n_labels)
        The conditional mutual information matrix.
    """
    print("mvcon compute_cmi")
    pcor_mv = partial_corr_mvar(label_ts)
    cmi = cmui_from_pcor_mvar(label_ts, pcor_mv)
    return cmi
