#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 11:29:46 2017

@author: padma
"""
import numpy as np
from scipy import linalg
from sklearn.decomposition import PCA


def reduce_dim(this_label_ts, dim_red, min_dim=15, max_dim=100, n_use=None):
    """Reduce dimensionality using PCA.

    Parameters
    ----------
    this_label_ts : array, shape (n_voxels, n_times)
        The label time series.
    dim_red : float
        The fraction explained variance to target (between 0 and 1.),
        only applicable if n_use is not None.
    min_dim : int
        The minimum dimensionality, only applicable if n_use is not None.
    max_dim : int
        The maximum number of dimensions, only applicable if n_use is not None.
    n_use : int | None
        If None, then infer the number of components based on dim_red,
        min_dim, and max_dim. This is typically for the data. If not None,
        then use n_use components for the PCA reduction. This is typically
        for the noise.
    """
    max_dim = np.min(this_label_ts.shape)

    for try_no in range(3):
        try:
            rng = np.random.RandomState(42 + try_no)
            if n_use is None:
                pca = PCA(n_components=max_dim, whiten=False, random_state=rng,
                        svd_solver='randomized')
                pca.fit(this_label_ts.T)
                this_exp_var = np.cumsum(pca.explained_variance_ratio_)
                n_use = max(np.sum(this_exp_var < dim_red), min_dim)
            pca = PCA(n_components=n_use, whiten=False, random_state=rng,
                    svd_solver='randomized')
            ts_red = pca.fit_transform(this_label_ts.T).T

            # flip the sign of the components s.t. they are positively
            # correlated with the mean
            ts_mean = np.mean(this_label_ts, axis=0)
            sign_flip_idx = np.where(np.sum(ts_mean[None, :] * ts_red, axis=1) < 0)[0]
            ts_red[sign_flip_idx] *= -1
            break # exit the loop if we get here, exit the loop
        except linalg.LinAlgError as err:
            if try_no == 2:
                raise err
    return ts_red
