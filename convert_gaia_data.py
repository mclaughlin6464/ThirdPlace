# The gaia data is currently in a large list of gzip'd csvs. Why? Because astronomy. I'm glad it's not in fits files, frankly. I'm gonna first try putting in HDF5, which will be a marginal improvement. If that doesn't quite work we can always do a SQL database but that may be more trouble I"m prepared to commit to. 

import numpy as np
from os import path
import pandas as pd
from glob import glob

gaia_dir = '/oak/stanford/orgs/kipac/edarragh/gaiadr2'
#ex_file = pd.read_csv(path.join(gaia_dir, 'GaiaSource_2024241623026358656_2024280144613411456.csv.gz'))

data_files = sorted(glob(path.join(gaia_dir, 'GaiaSource*.csv.gz')))

ra_bins = np.linspace(0, 360, 72+1)
dec_bins = np.linspace(-90,90, 36+1)
out_dir = '/scratch/users/swmclau2'
f  = pd.HDFStore(path.join(out_dir, 'gaiadr2.hdf5'), 'w')

for k, d in enumerate(data_files):
    #print('HI')
    data = pd.read_csv(d)

    #min_itemsize = dict(zip(data.columns, np.ones_like(data.columns)*30))
    for i, ra in enumerate(ra_bins[:-1]):
        data_ra_slice = data[(data['ra']>=ra) & (data['ra']<ra_bins[i+1])]
        if len(data_ra_slice) == 0:
            continue
        group_key = 'RA_%d_%d/'%(ra, ra_bins[i+1])
        for j, dec in enumerate(dec_bins[:-1]):
            data_ra_dec_slice = data_ra_slice[(data_ra_slice['dec']>=dec)                                         & (data_ra_slice['dec']<dec_bins[j+1])]
            if len(data_ra_dec_slice)>0:
                dset_key = 'DEC_%d_%d'%(dec, dec_bins[j+1])
                try:
                    f.put(group_key+dset_key, data_ra_dec_slice, format='table', append = True)#, min_itemsize=min_itemsize)
                except:
                    print 'Skipping row', i,j,k
                    print data_ra_dec_slice

    if k%1000==0:
        print k,

f.close()

