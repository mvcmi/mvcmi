"""
=====================
Run MVCMI on HCP data
=====================

This example demonstrates how to run MVCMI on
pre-processed HCP data.
"""

# Authors: Padma Sundaram
#          Mainak Jas

# %%
# we will first load the necessary modules
import numpy as np
import matplotlib.pyplot as plt

from mvcmi import compute_cmi, compute_ccoef_pca, reduce_dim
from mvcmi.datasets import fetch_hcp_sample, load_label_ts

from joblib import Parallel, delayed

n_jobs = 1 # number of cores to use when running PCA in parallel
n_parcels = 10 # just to make example run faster
dim_red = 0.95

# %%
# load the preprocessed data
data_path = fetch_hcp_sample()

label_ts_fname = data_path / 'label_ts.npz'
label_ts = load_label_ts(label_ts_fname, n_parcels=n_parcels)

# %%
# reduce dimensionality using PCA
n_times = label_ts[0].shape[1] 

min_dim = 2
max_dim = n_times - 15

parcel_sizes = [None] * len(label_ts)
label_ts_red = Parallel(n_jobs=n_jobs, verbose=4)(delayed(reduce_dim)(
    this_ts, dim_red=dim_red, min_dim=min_dim, max_dim=max_dim, n_use=n_use)
    for this_ts, n_use in zip(label_ts, parcel_sizes))

# %%
# do the actual CMI computation
print("computing cmi")
cmimtx = compute_cmi(label_ts_red)

# %%
# compare to correlation coefficient
print("computing sccoef_pca")
corrmtx = compute_ccoef_pca(label_ts_red)

# %%
# plot the CMI matrix
plt.imshow(cmimtx)
plt.colorbar()

# %%
# plot the correlation matrix
plt.figure()
plt.imshow(corrmtx)
plt.colorbar()

# %%
# now let us compute CMI for the null distribution
null_ts = generate_null_dist(label_ts_red, min_dim, max_dim, dim_red=dim_red,
                              seed1=0, seed2=50, n_jobs=n_jobs)
null_cmi = compute_cmi(null_ts)

# %%
# 