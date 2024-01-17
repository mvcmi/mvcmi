"""
=====================
Run MVCMI on HCP data
=====================

This example demonstrates how to run MVCMI on
pre-processed HCP data.
"""

# Authors: Padma Sundaram <padma@nmr.mgh.harvard.edu>
#          Mainak Jas <mjas@mgh.harvard.edu>

# %%
# we will first load the necessary modules
import numpy as np
import matplotlib.pyplot as plt

from mvcmi import compute_cmi, compute_ccoef_pca, generate_noise_ts, z_score
from mvcmi.pca import reduce_dim
from mvcmi.datasets import fetch_hcp_sample, load_label_ts

from joblib import Parallel, delayed

n_jobs = 30 # number of cores to use when running PCA in parallel
n_parcels = 10 # just to make example run faster
dim_red = 0.95

# %%
# load the preprocessed data
path = '/autofs/space/meghnn_001/users/mjas/github_repos/mvcmi_open/examples/mvcmi_data'
data_path = fetch_hcp_sample(path=path)

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
data_cmi = compute_cmi(label_ts_red)

# %%
# compare to correlation coefficient
print("computing sccoef_pca")
corrmtx = compute_ccoef_pca(label_ts_red)

# %%
# plot the CMI matrix
plt.imshow(data_cmi)
plt.colorbar()

# %%
# plot the correlation matrix
plt.figure()
plt.imshow(corrmtx)
plt.colorbar()

# %%
# now let us compute CMI for the null distribution. Generally, the number
# of seeds are determined empirically. For the HCP dataset, it was observed
# that 50 seeds are sufficient to obtain stable null distribution.
noise_ts = generate_noise_ts(label_ts, label_ts_red, min_dim, max_dim,
                             dim_red=dim_red, seed1=0, seed2=10,
                             n_jobs=n_jobs)
null_cmis = list()
for seed, this_noise_ts in enumerate(noise_ts):  # iterate over seeds
    print(f'mvcmi for seed {seed}')
    null_cmis.append(compute_cmi(this_noise_ts))

# %%
# finally, we z-score the CMI values (and optionally threshold)
z_cmi = z_score(data_cmi, np.array(null_cmis))

# %%
# let us plot the z-scored CMI values
plt.figure()
plt.imshow(z_cmi)
plt.colorbar()
