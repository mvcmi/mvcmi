"""Functions for dimensionality reduction."""

# Authors: Padma Sundaram <padma@nmr.mgh.harvard.edu>
#          Mainak Jas <mjas@mgh.harvard.edu>

import numpy as np
from scipy import linalg
from sklearn.decomposition import PCA

from joblib import Parallel, delayed

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


def generate_noise_ts(label_ts, label_ts_red, min_dim, max_dim, dim_red=0.95,
                      seed1=0, seed2=50, n_jobs=1):
    """Generate null distribution.

    Parameters
    ----------
    label_ts : list of arrays of shape (n_label_voxels, n_times)
        The data label time series BEFORE dimensionality reduction.
    label_ts_red : list of arrays of shape (n_label_voxels, n_times)
        The data label time series AFTER dimensionality reduction.
    min_dim : float
        dim_red to be passed to reduce_dim function for reducing
        dimension of noise.
    max_dim : float
        dim_red to be passed to reduce_dim function for reducing
        dimension of noise.
    dim_red : float
        dim_red to be passed to reduce_dim function for reducing
        dimension of noise.
    seed1 : int
        The starting seed for generating the distribution.  
    seed2 : int
        The ending seed for generating the distribution; seed1
        and seed2 will determine the number of data points in the
        null distribution.
    n_jobs : int
        The number of jobs for parallel processing of PCA.

    Returns
    -------
    noise_ts : list of arrays of shape (n_label_voxels, n_times)
        The noise time series with same dimensions as label time series
        and variance scaled to the variance of the data.
    """
    p = len(label_ts_red)

    psz = list()  # parcel sizes
    psz_red = list()  # parcel size of reduced data
    label_vars = list()  # label variances
    for this_label_ts, this_label_ts_red in zip(label_ts, label_ts_red):
        psz.append(np.shape(this_label_ts))
        psz_red.append(np.shape(this_label_ts_red))
        label_vars.append(np.var(this_label_ts_red))
      
    noise_ts = list()
    print('Generating noise time series')
    for seed in np.arange(seed1, seed2 + 1, 1):
        print(seed)
        rng = np.random.RandomState(seed)

        # Generate the noise time series
        this_noise_ts = list()
        for ii in range(p):
            label_ts_noise = rng.randn(*psz[ii])
            this_noise_ts.append(label_ts_noise)
            
        # Apply dimensionality reduction on noise time series
        noise_ts_red = Parallel(n_jobs=n_jobs, verbose=4)(delayed(reduce_dim)
            (this_ts, dim_red=dim_red, min_dim=min_dim, max_dim=max_dim, n_use=n_use[0])
            for this_ts, n_use in zip(this_noise_ts, psz_red))  

        # Scale noise time series by variance of data time series
        for label_noise_ts, label_var in zip(noise_ts_red, label_vars):
            scale = np.sqrt(label_var) / np.std(label_noise_ts, axis=1)            
            label_noise_ts *= scale[:, None]
        
        noise_ts.append(noise_ts_red)

    return noise_ts
