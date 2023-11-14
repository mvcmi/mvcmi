"""
=====================
Run MVCMI on HCP data
=====================

This example demonstrates how to run MVCMI on
pre-processed HCP data.
"""

starttime = datetime.now()

# load the raw data
label_ts_fname = '%s/label_ts.npz' % (label_ts_dir)
label_ts_load = np.load(label_ts_fname)
keys = label_ts_load.keys()
n_elems = len(keys)
label_ts = [None] * n_elems
for key in keys:
    idx = int(key.split('_')[1])
    label_ts[idx] = label_ts_load[key][:, 0:] #n_skip_start:]

print("done reading in label_ts\n")
print("%d\n" % n_elems)    
        
p = len(label_ts) # number of parcels
n_times = label_ts[0].shape[1] 

min_dim = 2
max_dim = n_times - 15
label_ts_red = []
psz_red = np.zeros(p, dtype=int)

n_jobs = 8
parcel_sizes = [None] * len(label_ts)

label_ts_red = Parallel(n_jobs=n_jobs, verbose=4)(delayed(myparcf.reduce_dim)(this_ts, dim_red=dim_red, min_dim=min_dim,max_dim=max_dim,n_use=n_use) for this_ts, n_use in zip(label_ts, parcel_sizes))

for ii in np.arange(0,p): #range(p):
    psz_red[ii] = np.shape(label_ts_red[ii])[0]
    parcel_sizes[ii] = np.shape(label_ts[ii])[0]
    
#print "done PCA reduction of parcels"

var_red = np.zeros(p) #Parallel(n_jobs=n_jobs, verbose=4)(delayed(myparcf.comp_var)(this_ts) for this_ts in label_ts_red)
for jj, this_ts in enumerate(label_ts_red):
    var_red[jj] = np.var(this_ts)

endtime = datetime.now()

dt = endtime - starttime
print ('Took %5.4f sec' % (dt.total_seconds()))

print ("computing cmi")
cmimtx = mvcon.compute_cmi(label_ts_red, method=method)

print ("computing sccoef_pca")
corrmtx = mvcon.compute_ccoef_pca(label_ts_red)

outdict = {'parc': 'aparc', 'dim': dim_red, 'corrmtx': corrmtx,
            'cmimtx': cmimtx, 'psz': parcel_sizes, 'psz_red': psz_red}
np.save('%s/dataout_%d.npy' % (outdir, np.int64(dim_red*100)), outdict)
np.savez_compressed('%s/label_ts_red_var_%d.npz' % (outdir, np.int64(dim_red*100)), *var_red)
