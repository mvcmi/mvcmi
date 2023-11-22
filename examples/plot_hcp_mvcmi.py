"""
=====================
Run MVCMI on HCP data
=====================

This example demonstrates how to run MVCMI on
pre-processed HCP data.
"""

# Authors: Padma Sundaram
#          Mainak Jas

import numpy as np
import matplotlib.pyplot as plt

from mvcmi import compute_cmi, compute_ccoef_pca, reduce_dim
from mvcmi.datasets import fetch_hcp_sample, load_label_ts

from joblib import Parallel, delayed

data_path = fetch_hcp_sample()

n_jobs = 1 # number of cores to use to run PCA in parallel
n_parcels = 10 # just to make example run faster
dim_red = 0.95

# %
# load the preprocessed data
label_ts_fname = data_path / 'label_ts.npz'
label_ts = load_label_ts(label_ts_fname)

# %
# reduce dimensionality
n_times = label_ts[0].shape[1] 

min_dim = 2
max_dim = n_times - 15

parcel_sizes = [None] * len(label_ts)
label_ts_red = Parallel(n_jobs=n_jobs, verbose=4)(delayed(reduce_dim)(
    this_ts, dim_red=dim_red, min_dim=min_dim,max_dim=max_dim,n_use=n_use)
    for this_ts, n_use in zip(label_ts, parcel_sizes))

# %
# do the actual CMI computation
print ("computing cmi")
cmimtx = compute_cmi(label_ts_red)

# %
# compare to correlation coefficient
print ("computing sccoef_pca")
corrmtx = compute_ccoef_pca(label_ts_red)

# %
# plot the matrix
plt.imshow(cmimtx)
plt.imshow(corrmtx)
