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

def z_score(data_cmis, null_cmis, alpha=None):
    """Compute z-score from the data and null CMIs.

    Parameters
    ----------
    data_cmis : array, shape (n_labels, n_labels)
        The data CMI matrix.
    null_cmis : array, shape (n_seeds, n_labels, n_labels)
        The null CMI matrix.
    alpha : float | None
        The alpha to use for the thresholding. Must be
        between 0. and 1.

    Returns
    -------
    z_cmis : array, shape (n_labels, n_labels)
        The z-scored CMI matrix.
    """

    p = data_cmis.shape[0]
    idx_lt = np.tril_indices(p, k=-1)

    z_cmi = data_cmis.copy()

    # z-score
    mu, sig = null_cmis.mean(axis=0), null_cmis.std(axis=0)
    z_cmi[idx_lt] -= mu[idx_lt]
    z_cmi[idx_lt] /= sig[idx_lt]

    z_cmi = np.abs(z_cmi)

    # Thresholding
    if alpha is not None:
        percentile = (1 - alpha / (p * (p - 1) / 2.)) * 100.
        null_cmis -= mu
        null_cmis /= sig
        z_thresh = np.percentile(null_cmis, percentile, axis=0)
        z_cmi = np.clip(z_cmi, z_thresh, None)

        # remove nans
        z_cmi2 = np.zeros_like(z_cmi)
        z_cmi2[idx_lt] = z_cmi[idx_lt]
        z_cmi = z_cmi2.copy()

    return z_cmi
