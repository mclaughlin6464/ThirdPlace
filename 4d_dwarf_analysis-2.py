import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import h5py
import tables
import os 
from skimage import measure 
from scipy import ndimage
from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
import pywt
import cv2
from imutils import contours
import argparse
import imutils
import networkx 
from networkx.algorithms.components.connected import connected_components
from skimage.feature import canny 
from scipy.stats import poisson 
from sklearn.mixture import GaussianMixture as GMM
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse
device = 'cpu'


def main():
    number_stars = [] 
    number_stars_used = []
    sigma = []  
    ra_avg = []
    ra_err = []
    dec_avg = []
    dec_err = []
    pmra_avg = []
    pmra_err = []
    pmdec_avg = []
    pmdec_err = []
    names = []
    dwarf_all = pd.read_csv('dwarf_list.csv') 
    for name in ['reticulumii']: #dwarf_all['name']:
        mask = dwarf_all['name'] == name
        dwarf = dwarf_all[mask] 
        median_ra = (dwarf['ra']).values[0]
        median_dec =  (dwarf['dec']).values[0]
        b = np.abs((dwarf['b']).values[0])
        loc, loc1, loc2, loc3 = load_data(median_ra, median_dec)
        data = pd.read_hdf('/oak/stanford/orgs/kipac/edarragh/gaiadr2.hdf5', loc) 
        data1 = pd.read_hdf('/oak/stanford/orgs/kipac/edarragh/gaiadr2.hdf5', loc1)
        data2 = pd.read_hdf('/oak/stanford/orgs/kipac/edarragh/gaiadr2.hdf5', loc2)
        data3 = pd.read_hdf('/oak/stanford/orgs/kipac/edarragh/gaiadr2.hdf5', loc3)

        data = data.append(data1)
        data = data.append(data2)
        data = data.append(data3)

        data = mask_data(data, median_ra, median_dec)
        plot_field(data,  name)
        plot_proper_motion(data,  name)

        rand_ra, rand_dec, rand_pmra, rand_pmdec = make_random_image(data, median_ra, median_dec) 

        full_field = np.c_[data['ra'].as_matrix(), data['dec'].as_matrix(), 
                   data['pmra'].as_matrix(), data['pmdec'].as_matrix()]

        random_field = np.c_[rand_ra, rand_dec, rand_pmra, rand_pmdec]

        field_image, edges = make_field_image(full_field, median_ra, median_dec)
        random_image, edges = make_field_image(random_field, median_ra, median_dec)

        filtered_ims = field_image 

        plot_name = name + '/position_proper_motion.png'
        plt.subplot(121)
        plt.imshow(ndimage.filters.gaussian_filter(filtered_ims.sum(axis = (2,3)).T, 6.0), origin='lower')
        plt.grid(color='w', linestyle='-', linewidth=1)
        plt.subplot(122)
        plt.imshow(ndimage.filters.gaussian_filter(filtered_ims.sum(axis = (0,1)).T, 6.0), origin='lower')
        plt.grid(color='w', linestyle='-', linewidth=1)
        plt.savefig(plot_name)
        plt.close() 
        plot_name = name + '/position_vs_proper_motion.png'
        plt.subplot(121)
        plt.imshow(ndimage.filters.gaussian_filter(filtered_ims.sum(axis = (1,3)).T, 6.0), origin='lower')
        plt.grid(color='w', linestyle='-', linewidth=1)
        plt.subplot(122)
        plt.imshow(ndimage.filters.gaussian_filter(filtered_ims.sum(axis = (0,2)).T, 6.0),  origin='lower')
        plt.grid(color='w', linestyle='-', linewidth=1)
        plt.savefig(plot_name)
        plt.close() 
 
        wavelets = ['bior1.3', 'bior4.4', 'bior5.5', 'bior6.8'] #, 'bior4.4','bior5.5', 'bior6.8']
        wav_len = [6, 8, 12, 16]


        wv = WaveNet(128, 128, J=4,wavelets = wavelets, nonlinear_op = foo)

        wv = wv.to(device)
        out = wv(torch.Tensor(field_image).to(device)).to('cpu').numpy()
        rand_out = wv(torch.Tensor(random_image).to(device)).to('cpu').numpy()

        out = (ndimage.filters.gaussian_filter(out,3.0)-np.mean(ndimage.filters.gaussian_filter(rand_out,3.0)))/np.std(ndimage.filters.gaussian_filter(rand_out,3.0))

        a = b*0.2 + 14.0 #23.947
        mask = out < a 
        out[mask] = 0 
        out = out/a 
       
        txtname = name + '/significance_cut.txt'
        np.savetxt(txtname, np.array([a])) 


        plot_name = name + '/final_cuts.png'
        plt.subplot(121)
        plt.imshow(out[8:120, 8:120,8:120,8:120].sum(axis = (2,3)).T, origin='lower')
        plt.grid(color='w', linestyle='-', linewidth=1)
        plt.subplot(122)
        plt.imshow(out[8:120,8:120,8:120,8:120].sum(axis = (0,1)).T, origin='lower')
        plt.grid(color='w', linestyle='-', linewidth=1)
        plt.savefig(plot_name)
        plt.close()
 
        out1 = out[8:120, 8:120,8:120,8:120]

        struct = np.ones((3,3,3,3))
        labels, n = ndimage.measurements.label(out1, structure=struct)

        significance = []
        blobs = []
        ss = []

        for k in range(1,n+1):
     
            short = out1.copy()
            mask = labels == k 
            sig = np.max(short[mask]) 
            short[mask] = 1 
            mask = np.invert(mask)
            short[mask] = 0 
            s = np.sum(short) 
            if(s > 1): 
                ss.append(s)
                significance.append(sig)  
                ww = np.nonzero(short)
                X = np.mean(ww[0]) 
                Y = np.mean(ww[1]) 
                Z = np.mean(ww[2]) 
                T = np.mean(ww[3]) 
                R = len(ww[0])**(1/2) #max(len(ww[0])**(1/4), 15)
                blobs.append([X, Y, Z, T, R]) 
        txtname = name + '/significance.txt'
        np.savetxt(txtname, significance) 

        txtname = name + '/ss.txt'
        np.savetxt(txtname, ss) 

#        mask = find_labels2(rd)
#        if np.max(mask) == 0:
#            continue 

#        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#        cnts = imutils.grab_contours(cnts)
#        cnts = contours.sort_contours(cnts)[0]
#        blobs = [] 
   
#        plot_name = name + '/position_blobs.png'
#        fig, ax = plt.subplots()
#        ax.imshow(rd)
#        for (i, c) in enumerate(cnts):
#            ((cX, cY), radius) = cv2.minEnclosingCircle(c)
#            if(cX > 16 and cX < 112 and cY > 16 and cY < 112):
#                blobs.append([cX, cY, radius])
#                c = plt.Circle((cX, cY), radius, color='red', linewidth=2, fill=False)
#                ax.add_patch(c)
#            ax.set_axis_off()
#        plt.tight_layout()
#        plt.savefig(plot_name)
#        plt.close() 

#        mask = find_labels2(prpd)
#        if np.max(mask) == 0:
#            continue
#
#        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#        cnts = imutils.grab_contours(cnts)
#        cnts = contours.sort_contours(cnts)[0]
#        blobs2 = []

#        plot_name = name + '/proper_motion_blobs.png'
#        fig, ax = plt.subplots()
#        ax.imshow(prpd)
#        for (i, c) in enumerate(cnts):
#            ((cX, cY), radius) = cv2.minEnclosingCircle(c)
#            #if(cX > 16 and cX < 112 and cY > 16 and cY < 112):
#            blobs2.append([cX, cY, radius])
#            c = plt.Circle((cX, cY), radius, color='red', linewidth=2, fill=False)
#            ax.add_patch(c)
#            ax.set_axis_off()
#        plt.tight_layout()
#        plt.savefig(plot_name)
#        plt.close()

#        mask1 = blobs
        clusters = find_clusters(data, blobs, median_ra, median_dec) 
 #       G = to_graph(clusters)
 #       p = (connected_components(G))

 #       clusters2 = [] 
 #       for i in p:
 #           list_temp = list(i)
 #           mask = data['ra'].isin(list_temp)
 #           stars = data[mask]
 #           clusters2.append(stars)
#
#        clusters2.sort(key=len)
#        clusters2 = clusters2[::-1]
        clusters2 = clusters
        plot_clusters(clusters2, data, name, median_ra, median_dec)

        distance = [] 
        index = 0 
        for cluster in clusters2:
            fname = name + '/' + name  + str(index) + '.csv'
            pd.DataFrame.to_csv(cluster, fname)
            avg_ra = np.sum(cluster['ra']/cluster['ra_error'])/np.sum(1/cluster['ra_error'])
            avg_dec = np.sum(cluster['dec']/cluster['dec_error'])/np.sum(1/cluster['dec_error'])
            distance.append(np.sqrt((avg_ra-median_ra)**2 + (avg_dec-median_dec)**2))
            index = index + 1
 
        names.append(name) 
        if (len(clusters2) == 0):
          
            number_stars.append(0)
            number_stars_used.append(0)
            ra_avg.append(0)
            dec_avg.append(0)
            ra_err.append(0)
            dec_err.append(0)
            pmra_avg.append(0)
            pmdec_avg.append(0)
            pmra_err.append(0)
            pmdec_err.append(0)
        else: 
            closest_cluster = np.argmin(distance)
#            cluster = clusters2[closest_cluster]
#            mask1 = np.abs(cluster['ra_error']) < 5 
#            mask2 = np.abs(cluster['dec_error']) < 5
#            mask3 = np.abs(cluster['pmra_error']) < 5
#            mask4 = np.abs(cluster['pmdec_error']) < 5
            cluster1 = clusters2[closest_cluster] #[mask1 & mask2 & mask3 & mask4] 
            if (len(cluster1) == 0):
                number_stars.append(len(cluster))
                number_stars_used.append(0)
                ra_avg.append(0)
                dec_avg.append(0)
                ra_err.append(0)
                dec_err.append(0)
                pmra_avg.append(0)
                pmdec_avg.append(0)
                pmra_err.append(0)
                pmdec_err.append(0)

            else: 
                number_stars.append(len(cluster))
                number_stars_used.append(len(cluster1))  
                ra_avg.append(np.sum(cluster1['ra']/cluster1['ra_error'])/np.sum(1/cluster1['ra_error']))
                dec_avg.append(np.sum(cluster1['dec']/cluster1['dec_error'])/np.sum(1/cluster1['dec_error']))
                ra_err.append(np.sqrt(np.sum(cluster1['ra_error']**2)))
                dec_err.append(np.sqrt(np.sum(cluster1['dec_error']**2)))
                pmra_avg.append(np.sum(cluster1['pmra']/cluster1['pmra_error'])/np.sum(1/cluster1['pmra_error']))
                pmdec_avg.append(np.sum(cluster1['pmdec']/cluster1['pmdec_error'])/np.sum(1/cluster1['pmdec_error']))
                pmra_err.append(np.sqrt(np.sum(cluster1['pmra_error']**2)))
                pmdec_err.append(np.sqrt(np.sum(cluster1['pmdec_error']**2)))
    columns = ['name','nstars', 'nstars_cut','ra_avg', 'ra_err', 'dec_avg', 'dec_err', 'pmra_avg', 'pmra_err', 'pmdec_avg', 'pmdec_err']
    final_values = pd.DataFrame(columns=columns)
    final_values['name'] = names 
    final_values['nstars'] = number_stars
    final_values['nstars_cut'] = number_stars_used
    final_values['ra_avg'] = ra_avg
    final_values['ra_err'] = ra_err
    final_values['dec_avg'] = dec_avg
    final_values['dec_err'] = dec_err  
    final_values['pmra_avg'] = pmra_avg
    final_values['pmra_err'] = pmra_err
    final_values['pmdec_avg'] = pmdec_avg
    final_values['pmdec_err'] = pmdec_err      
    pd.DataFrame.to_csv(final_values, 'detection_rates.csv')

def load_data(ra, dec):
    ra_bins = np.linspace(0, 360, 72+1)
    dec_bins = np.linspace(-90,90, 36+1)

    for i in range(len(ra_bins)):
        if (ra_bins[i] > ra):
            if ( ra_bins[i] - ra > 4):
                i2 = i-1 
            else:
                i2 = i+1
            break

    for j in range(len(dec_bins)):
        if (dec_bins[j] > dec):
            if ( dec_bins[j] - dec > 4):
                j2 = j-1 
            else:
                j2 = j+1 
            break
    if i2 == 73: i2 = 1 
    if i2 == -1: i2 = 72 
    if j2 == 37: j2 = 1
    if j2 == -1: j2 = 36 
    if(ra == 0.7):
        group_key_ra = 'RA_%d_%d/'%(0, 5)
        group_key_dec = 'DEC_%d_%d'%(-60, -55)
        group_key_ra2 = 'RA_%d_%d/'%(355, 360)
        group_key_dec2 = 'DEC_%d_%d'%(-65, 60)
    else: 
        group_key_ra = 'RA_%d_%d/'%(ra_bins[i-1], ra_bins[i])
        group_key_dec = 'DEC_%d_%d'%(dec_bins[j-1], dec_bins[j])
        group_key_ra2 = 'RA_%d_%d/'%(ra_bins[i2-1], ra_bins[i2])
        group_key_dec2 = 'DEC_%d_%d'%(dec_bins[j2-1], dec_bins[j2])

    loc = group_key_ra + group_key_dec
    loc1 = group_key_ra2 + group_key_dec
    loc2 = group_key_ra + group_key_dec2
    loc3 = group_key_ra2 + group_key_dec2    
    return loc, loc1, loc2, loc3 

def mask_data(data, median_ra, median_dec):
    mask = data['ra'] > median_ra - 1 
    mask1 = data['ra'] < median_ra  + 1 
    mask2 = data['dec'] > median_dec - 1 
    mask3 = data['dec'] < median_dec + 1 
    mask4 = data['parallax'] < 0.1
    mask5 = (np.invert(np.isnan(data['parallax'])))
    mask6 = (np.invert(np.isnan(data['ra'])))
    mask7 = (np.invert(np.isnan(data['dec'])))
    mask8 = (np.invert(np.isnan(data['pmra'])))
    mask9 = (np.invert(np.isnan(data['pmdec'])))
    mask10 = data['pmra'] > -4.5
    mask11 = data['pmra'] < 4.5
    mask12 = data['pmdec'] > -4.5
    mask13 = data['pmdec'] < 4.5
    data = data[mask & mask1 & mask2 & mask3 & mask4 & mask5 & mask6 & mask7 & mask8 & mask9 & mask10 & mask11 & mask12 & mask13]
    return data 


def plot_field(data, name):
    fname = name + '/field_image.png'
    plt.scatter(data['ra'], data['dec'], c='b',alpha=0.1)
    plt.xlabel('ra')
    plt.ylabel('dec')
    plt.savefig(fname) 
    plt.close() 

def plot_proper_motion(data, name):
    fname = name + '/proper_motion.png'
    plt.scatter(data['pmra'], data['pmdec'], c='b',alpha=0.1)
    plt.xlabel('pmra')
    plt.ylabel('pmdec')
    plt.savefig(fname)
    plt.close()

def make_random_image(data, median_ra, median_dec): 
    ra_min, ra_max = median_ra-1, median_ra+1
    dec_min, dec_max = median_dec-1, median_dec+1
    vra_min, vra_max =  -4.5, 4.5
    vdec_min, vdec_max =  -4.5, 4.5

    x = np.linspace(ra_min, ra_max, 50)
    y = np.linspace(dec_min, dec_max, 50)
    mean = np.zeros((49,49))
    for i in range(len(x)-1):
        for j in range(len(y)-1):
            mask1 = data['ra'] > x[i]
            mask2 = data['ra'] < x[i+1]
            mask3 = data['dec'] > y[j]
            mask4 = data['dec'] < y[j+1]  
            mean[i][j] = len(data[mask1 & mask2 & mask3 & mask4])

    rand_ra = []
    rand_dec = [] 
    for i in range(len(x)-1):
        for j in range(len(y)-1):
            num_points = poisson(np.mean(mean)).rvs()
            if num_points == 0:
                continue
            else: 
                rnd_ra = np.random.uniform(x[i], x[i+1], size=num_points)
                rnd_dec = np.random.uniform(y[j], y[j+1], size=num_points)
                for q in range(len(rnd_ra)): 
                    rand_ra.append(rnd_ra[q]) 
                    rand_dec.append(rnd_dec[q])

    X = np.array(list(zip(data['pmra'], data['pmdec'])))

    gmm = GMM(n_components=3).fit(X)
    sample = gmm.sample(len(rand_ra))
    rand_pmra = (sample[0][:,0]) 
    rand_pmdec = (sample[0][:,1])

    return rand_ra, rand_dec, rand_pmra, rand_pmdec 


def make_field_image(full_field, median_ra, median_dec, n_pos= 129, n_vel=129):
    pos_min, pos_max =  np.min(full_field, axis =0)[:2], np.max(full_field, axis =0)[:2]
    ra_min, ra_max = median_ra-1.0, median_ra+1.0
    dec_min, dec_max = median_dec-1.0, median_dec+1.0
    
    
    vra_min, vra_max = -4.5, 4.5
    vdec_min, vdec_max =-4.5, 4.5


    field_image, edges = np.histogramdd(full_field,\
                            bins=(np.linspace(ra_min, ra_max, n_pos),
                                  np.linspace(dec_min, dec_max, n_pos), 
                                  np.linspace(vra_min, vra_max, n_vel),
                                  np.linspace(vdec_min, vdec_max, n_vel) ) )
    
    return field_image, edges

def find_labels(thresh):
    temp_thresh = thresh
    mask = temp_thresh != 0 
    temp_thresh = thresh 
    temp_thresh[mask] = 255 
    labels = measure.label(temp_thresh, neighbors=8, background=0)

    for label in np.unique(labels):

        if label == 0:
            continue
        
        mask = labels == label 
        labels[mask] = 1 
        num_pix = np.sum(labels[mask])
        if num_pix > 25:
            continue
        else:
            labels[mask] = 0 
    thresh[labels==0] = 0 
#    thresh[labels==1] = 1
    return thresh 

def find_labels2(thresh):
    temp_thresh = thresh
    mask = temp_thresh != 0
    temp_thresh = thresh
    temp_thresh[mask] = 255
    labels = measure.label(temp_thresh, neighbors=8, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")

    for label in np.unique(labels):

        if label == 0:
            continue

        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        
        if numPixels > 25:
            mask = cv2.add(mask, labelMask)
    return mask

class WaveNet(nn.Module):
    def __init__(self, n_pos, n_vel, J=1,\
                 wavelets = ['bior1.1', 'bior2.2', 'bior3.3'],\
                 nonlinear_op = lambda x:x): # TODO customize size and features
        super(WaveNet, self).__init__()
        
        self.n_pos = n_pos
        self.n_vel = n_vel
        
        self.nonlinear_op = nonlinear_op
        #self.sigmas = sigmas
        self.wavelets = wavelets
        self.wavelet_filters ={}
        self.iwavelet_filters = {}
        #for s in sigmas:
        #    gauss_filter1 = GaussianFilter(n_pos, n_vel, s)
        #    gauss_filter2 = GaussianFilter(n_vel, n_pos, s)
        #    self.gauss_filters[s] = (gauss_filter1, gauss_filter2)
        
        #self.gauss_filter = lambda im : torch.Tensor(ndimage.filters.gaussian_filter(im, 1) )

        for i,w in enumerate(wavelets):
        
            wavelet = DWTForward(J=J, wave = w)
            setattr(self, 'Wavelet_%d'%i, wavelet)
            self.wavelet_filters[w] = wavelet
            
            iwavelet = DWTInverse(wave = w)
            setattr(self, 'InvWavelet_%d'%i, iwavelet)
            self.iwavelet_filters[w] = iwavelet
        
        self.upsample1 = lambda input: F.interpolate(input, size = (n_pos, n_pos))
        self.upsample2 = lambda input: F.interpolate(input, size = (n_vel, n_vel))
        #model.cuda()?
        
    def forward(self, x):
        assert x.shape[0] == x.shape[1] and x.shape[0] == self.n_pos
        assert x.shape[2] == x.shape[3] and x.shape[2] == self.n_vel
        
        # Gaussian smoothing
        
        low_wave_outs = []
        high_wave_outs = []
        #for s, (g1, g2) in self.gauss_filters.items():
        #    o = g1(x)
        #    o = o.permute(2,3,0,1)
            #print x.shape
        #    o = g2(o)
        #    smooth_outs.append(o.permute(2,3,0,1) )

        
        for w, (wv) in self.wavelet_filters.items():
            #for so in smooth_outs:
            o, hi1 = wv(x)
            #print(o.size(), _i[0].size(), _i[1].size())
            o = o.permute(2,3,0,1)
            
            o, hi2 = wv(o)
            
            o = np.exp(5)*o

            hi1[1] = hi1[1]*np.exp(9)
            hi1[2] = hi1[2]*np.exp(9)
            hi1[3] = hi1[3]*np.exp(9)

            hi2[1] = hi2[1]*np.exp(9)
            hi2[2] = hi2[2]*np.exp(9)
            hi2[3] = hi2[3]*np.exp(9)


            low_wave_outs.append(o)
            high_wave_outs.append((hi1, hi2))

#        nonlin_outs = self.nonlinear_op(low_wave_outs)
        
        output = torch.zeros((self.n_pos, self.n_pos,\
                              self.n_vel, self.n_vel)).to(device)

        for  _o, his, (w,  iwv) in zip(low_wave_outs, high_wave_outs,self.iwavelet_filters.items()):

            o = iwv([_o, his[1]])
            o = o.permute(2,3,0,1)
            o = iwv([o, his[0]])

            output += o #.permute(2,3,0,1)
        
        return output

def foo(filters, weights = [100,50,25,0,0]):
    assert len(filters) == len(weights), "Shape mismatch in nonlinear op foo"
    out = []

    for f, w in zip(filters, weights):
        out.append(f*w)
    return out


def perform_cuts(coeff_tot, coeff_pm_tot, coeff_ra_tot, coeff_dec_tot):

    rd = find_labels(coeff_tot) 
    prpd = find_labels(coeff_pm_tot) 
    rpr = find_labels(coeff_ra_tot) 
    dpd = find_labels(coeff_dec_tot) 
    rpra = rpr.sum(axis=0)
    dpda = dpd.sum(axis=0)
    rprb = rpr.sum(axis=1)
    dpdb = dpd.sum(axis=1)

    for i in range(rd[0].shape[0]):
        rd[i][rpra==0]     = 0 
        rd[:,i][dpda==0]   = 0 
        prpd[i][rprb==0]     = 0 
        prpd[:,i][dpdb==0]   = 0 
    return rd, prpd 


def find_clusters(data, blobs1, median_ra, median_dec):
    stars = []
    for blob in blobs1: 
        cut_idx1 = ((data['ra']-(2./128*(blob[0]-64 + 8)+median_ra))**2 + (data['dec']-(2./128*(blob[1]-64 + 8)+median_dec))**2) < (2./128*blob[4])**2
        cut_idx2 = ((data['pmra']-(8./128*(blob[2]-64 + 8)))**2 + (data['pmdec']-(8./128*(blob[3]-64+8)))**2) < (8./128*blob[4])**2

        stars.append(data[cut_idx1 & cut_idx2])

#    clusters = [] 
#    for blob in blobs2: 
#        for star in stars: 
#            cut_idx = ((star['pmra']-(8./128*(blob[0]-64)))**2 + (star['pmdec']-(8./128*(blob[1]-64)))**2) < (8./128*blob[2])**2
#            clusters.append(star[cut_idx])
    return stars

def to_graph(l):
    G = networkx.Graph()
    for part in l:
        # each sublist is a bunch of nodes
        G.add_nodes_from(part['ra'])
        # it also imlies a number of edges:
        G.add_edges_from(to_edges(part['ra']))
    return G

def to_edges(l):
    """ 
        treat `l` as a Graph and returns it's edges 
        to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]
    """
    it = iter(l)
    last = next(it)

    for current in it:
        yield last, current
        last = current 


def plot_clusters(clusters, data, name, median_ra, median_dec): 
    color = ['grey', 'maroon', 'royalblue', 'orange', 'darkorchid', 'darkolivegreen', 'lightseagreen','yellow', 'navy']
    color_index = 0 
    plot_name = name + '/' + name + '_position_all.png'
    plt.scatter(data['ra'].values, data['dec'].values, c='m',alpha=0.005)
    for cluster in clusters:
        if (len(cluster)>=5):
            plt.scatter(cluster['ra'].values, cluster['dec'].values, c=color[color_index],alpha=0.5)
            color_index = color_index + 1 
    plt.scatter(median_ra, median_dec, color='r', s=50)
    plt.xlim([median_ra-1,median_ra+1])
    plt.ylim([median_dec-1,median_dec+1])
    plt.xlabel('ra')
    plt.ylabel('dec')
    plt.savefig(plot_name)
    plt.close() 

    color_index = 0 
    plot_name = name + '/' + name + '_proper_motion_all.png'
    plt.scatter(data['pmra'].values, data['pmdec'].values, c='m',alpha=0.005)
    for cluster in clusters:
        if (len(cluster)>=5):
            plt.scatter(cluster['pmra'].values, cluster['pmdec'].values, c=color[color_index],alpha=0.5)
            color_index = color_index + 1 
    plt.xlim([-4,4])
    plt.ylim([-4,4])
    plt.xlabel('pmra')
    plt.ylabel('pmdec')
    plt.savefig(plot_name)
    plt.close()
    plot_name = name + '/' + name + '_color_magnitude_all.png'
    color_index = 0 
    for cluster in clusters:
        if (len(cluster)>=5):
            plt.scatter((cluster['g_rp'].values), cluster['phot_g_mean_mag'].values, c=color[color_index],alpha=0.5)
            color_index = color_index + 1 
    plt.xlabel('BP-RP')
    plt.ylabel('G')
    plt.gca().invert_yaxis()
    plt.savefig(plot_name)
    plt.close()

    color_index = 0 
    plot_name = name + '/' + name + '_parallax_all.png'
    for cluster in clusters:
        if (len(cluster)>=5):
            plt.hist(cluster['parallax'], bins=25, color =color[color_index])
            color_index = color_index + 1 
    plt.xlabel('parallax')
    plt.ylabel('number of sources')
    plt.savefig(plot_name) 
    plt.close()

if __name__ == '__main__': main()



