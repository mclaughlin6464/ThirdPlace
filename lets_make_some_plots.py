import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import healpy as hp

dwarfs = pd.read_csv('dwarf_list.csv')
ra_peak = np.loadtxt('ra_peak.txt')  
dec_peak = np.loadtxt('dec_peak.txt') 
pmra_peak = np.loadtxt('pmra_peak.txt')
pmdec_peak = np.loadtxt('pmdec_peak.txt')

significance = np.loadtxt('significance_dwarfs.txt')  
significance_all = np.loadtxt('significance_all.txt') 

detect = np.sqrt((ra_peak-dwarfs['ra'][:56])**2 + (dec_peak-dwarfs['dec'][:56])**2)/dwarfs['theta'][:56] 
pmdetect =  np.sqrt((pmra_peak-dwarfs['pmra'][:56])**2 + (pmdec_peak-dwarfs['pmdec'][:56])**2) 

#mask for dwarfs with no returned hotspots 
def mask_sig(significance, detect, pmdetect):
    mask = significance != 0 
    significance = significance[mask] 
    detect = detect[mask] 
    pmdetect = pmdetect[mask] 
    return significance, detect, pmdetect 

#mask for distance > 5*dwarf size and proper motion distance > 3 (for dwarfs with pm values) 
def mask_all(significance, detect, pmdetect):
    mask1 = significance != 0 
    mask2 = detect < 5. 
    mask3 = pmdetect > 1.5 
    mask3 = np.invert(mask3) 
    significance = significance[mask1 & mask2 & mask3] 
    detect = detect[mask1 & mask2 & mask3]           
    pmdetect = pmdetect[mask1 & mask2 & mask3]       
    return significance, detect, pmdetect 

def significance_histogram(significance_all, significance, detect, pmdetect):
    significance, detect, pmdetect = mask_sig(significance, detect, pmdetect)
    significance_cut, detect_cut, pmdetect_cut = mask_all(significance, detect, pmdetect)

    plt.hist(significance_all, bins = 96, range = (1,13),  alpha=0.5, label='significance_all')
    plt.hist(significance, bins = 96, range = (1,13),color='purple', alpha=0.25, label='significance_dwarfs')
    plt.hist(significance_cut, bins = 96, range = (1,13),color='red', alpha=0.25, label='significance_dwarfs_cut') 

    plt.legend(fontsize=20) 

    plt.xlabel('significance', fontsize=20)
    plt.ylabel('number of hotspots', fontsize=20)

    plt.show()

def significance_vs_distance(significance, detect, pmdetect):
    significance, detect, pmdetect = mask_sig(significance, detect, pmdetect)
    significance_cut, detect_cut, pmdetect_cut = mask_all(significance, detect, pmdetect)

    mask = np.isnan(pmdetect) 
    mask = np.invert(mask) 

    significance = significance[mask]
    pmdetect = pmdetect[mask]
    detect = detect[mask] 

    plt.scatter(detect, pmdetect, c=np.log10(significance), label='dwarfs_all', s=100)
#    plt.scatter(detect_cut, significance_cut, label='dwarfs_cut',c='red', marker ='*', s=100)

#    plt.legend(fontsize=20)

    plt.colorbar() 

    plt.xscale('log')
    plt.yscale('log')

    plt.title('offset from literature values colored as log10(significance)', fontsize=20)

    plt.xlabel('distance to hotspot/dwarf extent', fontsize=20)
    plt.ylabel('distance to hotspot (mas/yr)', fontsize=20)


    plt.show() 

def sky_map(dwarfs, significance):
    NSIDE = 16
    NPIX = hp.nside2npix(NSIDE) 
    names = np.array(dwarfs['name'][:56])
    ra = np.array(dwarfs['ra'][:56])
    dec = np.array(dwarfs['dec'][:56])
    mask = significance != 0 
    sig1 = significance[mask] 
    ra1 = ra[mask]
    dec1 = dec[mask] 
    m = np.zeros(NPIX) 
    pix = hp.ang2pix(NSIDE,ra1,dec1,lonlat=True)  
    m[pix] = np.log10(sig1) 
    hp.mollview(m, coord='CG', title="dwarf sky map colored by significance")
    for i in range(len(names)): hp.projtext(ra[i], dec[i], names[i], lonlat=True,coord='CG', color='darkorange')

    plt.show()

def sensitivity_comparison(dwarfs, significance): 
    distance  = np.array([8, 16, 32, 64, 128, 256, 512]) 
    A_0_des   = np.array([21.5, 24.1, 17.2, 8.6, 6.6, 6.3])
    Mv_0_des  = np.array([7.8, 8.3, 5.2, 1.2, -1.1, -2.3])
    r_0_des   = np.array([3.8, 4.2, 4.3, 4.1, 4.1, 4.3])
    A_0_ps1   = np.array([23.8, 19.0, 14.1, 11.0, 7.5, 6.8])
    Mv_0_ps1  = np.array([7.1, 5.0, 1.8, -0.3, -2.2, -4.0])
    r_0_ps1   = np.array([4.0, 4.1, 4.2, 4.3, 4.2, 4.4])

    mask = significance == 0 
    sig = significance.copy()
    sig[mask] = 1 
    sig = np.log10(sig)
    dwarfs = dwarfs[:56] 

    x = np.linspace(-12, 0, 52)
    for i in range(len(A_0_des)): 
        mask1 = dwarfs['distance'] > distance[i]
        mask2 = dwarfs['distance'] < distance[i+1]
        dwarfs_cut = dwarfs[mask1 & mask2]
        sig_cut = sig[mask1 & mask2]
        y_des = A_0_des[i]/(x[:len(x)-2*(i)]-Mv_0_des[i]) + r_0_des[i] 
        y_ps1 = A_0_ps1[i]/(x[:len(x)-3*(i)]-Mv_0_ps1[i]) + r_0_ps1[i] 

        plt.scatter(dwarfs_cut['mag_V'], np.log10(dwarfs_cut['r1/2']), c=sig_cut, s=100)
        mask = sig_cut == 0 
        plt.scatter(dwarfs_cut['mag_V'][mask], np.log10(dwarfs_cut['r1/2'][mask]), c='red', s=100, marker='x')

        plt.plot(x[:len(x)-2*(i)], y_des, color='black', linestyle='--', label='des') 
        plt.plot(x[:len(x)-3*(i)], y_ps1, color='red', linestyle='--', label='ps1') 
       
        plt.legend(fontsize=20)
        plt.colorbar() 

        title = str(distance[i]) + ' < D < ' + str(distance[i+1])

        plt.title(title, fontsize=20)
        plt.xlabel('M_v', fontsize=20)
        plt.ylabel('log10(r/pc)', fontsize=20)

        plt.ylim([0,3])
        plt.show()

sensitivity_comparison(dwarfs, significance)
sky_map(dwarfs, significance)
significance_vs_distance(significance, detect, pmdetect)
significance_histogram(significance_all, significance, detect, pmdetect)
