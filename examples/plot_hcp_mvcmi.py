"""
=====================
Run MVCMI on HCP data
=====================

This example demonstrates how to run MVCMI on
pre-processed HCP data.
"""

import numpy as np
import matplotlib.pyplot as plt

from mvcmi import compute_cmi, compute_ccoef_pca, reduce_dim
from mvcmi.datasets import fetch_hcp_sample

from joblib import Parallel, delayed

data_path = fetch_hcp_sample()

n_jobs = 1
n_parcels = 10 # just to make example run faster
dim_red = 0.95
outdir = ''

# load the raw data
label_ts_fname = data_path / 'label_ts.npz'
label_ts_load = np.load(label_ts_fname)
keys = label_ts_load.keys()
n_elems = len(keys)
label_ts = [None] * n_elems
for key in keys:
    idx = int(key.split('_')[1])
    label_ts[idx] = label_ts_load[key][:, 0:]

label_ts = label_ts[:n_parcels]

print("done reading in label_ts\n")
print("%d\n" % n_elems)    
        
p = len(label_ts)
n_times = label_ts[0].shape[1] 

min_dim = 2
max_dim = n_times - 15
label_ts_red = []

parcel_sizes = [None] * len(label_ts)

label_ts_red = Parallel(n_jobs=n_jobs, verbose=4)(delayed(reduce_dim)(
    this_ts, dim_red=dim_red, min_dim=min_dim,max_dim=max_dim,n_use=n_use)
    for this_ts, n_use in zip(label_ts, parcel_sizes))

for ii in np.arange(0, p): #range(p):
    parcel_sizes[ii] = np.shape(label_ts[ii])[0]
    
#print "done PCA reduction of parcels"

var_red = np.zeros(p)
for jj, this_ts in enumerate(label_ts_red):
    var_red[jj] = np.var(this_ts)

print ("computing cmi")
cmimtx = compute_cmi(label_ts_red)

print ("computing sccoef_pca")
corrmtx = compute_ccoef_pca(label_ts_red)

plt.imshow(cmimtx)
plt.imshow(corrmtx)