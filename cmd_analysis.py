from ugali import isochrone 
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import gaia_query 

def feh2z(feh):
        # Section 3 of Dotter et al. 2008
    Y_p     = 0.245            # Primordial He abundance (WMAP, 2003)
    c       = 1.54             # He enrichment ratio 

        # This is not well defined...
        #Z_solar/X_solar = 0.0229  # Solar metal fraction (Grevesse 1998)
    ZX_solar = 0.0229
    return (1 - Y_p)/( (1 + c) + (1/ZX_solar) * 10**(-feh))

def plot_cmd(data, dwarf): 
    gdata = gaia_query.gaia_query(data, dwarf['ra'], dwarf['dec'])
    gdata= gaia_query.gaia_to_DECam_mag('r', gdata) 
    gdata= gaia_query.gaia_to_DECam_mag('g', gdata) 

    plt.scatter(gdata['mag_g_DECam'] - gdata['mag_r_DECam'], gdata['mag_g_DECam'] +  30.781, label=dwarf['name'].values[0])

    distance = 5*np.log10(dwarf['distance']*1000/10)

    z = [-3.0, -0.25] #np.linspace(-3.0, -0.25, 12) 
    for i in range(len(z)): 
        metallicity = feh2z(z[i]) 
        iso1 = isochrone.factory(name='Padova',age=12,metallicity=metallicity,distance_modulus = distance, band_1 = 'g', band_2 = 'r') 
        print(metallicity) 
        plt.scatter(iso1.mag_1-iso1.mag_2,iso1.mag_1+iso1.distance_modulus,marker='o',alpha=0.1, label ='fe/h =' +  str(z[i]))  

    plt.gca().invert_yaxis()   
    plt.ylim([23,14])

    plt.legend(fontsize=15) 
    plt.xlabel('g-r', fontsize=15)    
    plt.ylabel('g', fontsize=15)       
    plt.title('isochrone: age - 12 (Gyr)', fontsize=20)  

    fig_name = 'CMD_DWARFS/' + dwarf['name'].values + '_cmd.png'

    plt.savefig(fig_name[0]) 
    plt.close() 


dwarfs = pd.read_csv('dwarf_list.csv')

significance = []
significance_cut = [] 
ra_peak = []
dec_peak = []
nstars = []
pmra_peak = []
pmdec_peak = []


for i in range(len(dwarfs)):
    mask = dwarfs['name'] == dwarfs['name'][i]
    dwarf = dwarfs[mask]
    if (dwarf['nob'].values[0] == -1):
        significance.append(0)
        data_path = '/Users/elisedarragh-ford/Downloads/' + name + '/significance_cut.txt'
        sig_cut = np.loadtxt(data_path[0])
        significance_cut.append(sig_cut.item(0))
        ra_peak.append(0)
        dec_peak.append(0)
        pmra_peak.append(0)
        pmdec_peak.append(0)
        nstars.append(0)
        continue 

    nob = int(dwarf['nob'].values)
    name = (dwarf['name']).values
    data_path = '/Users/elisedarragh-ford/Downloads/' + name + '/' + name + str(nob) + '.csv'
    print(data_path)
    print(nob) 
    data = pd.read_csv(data_path[0])
    plot_cmd(data, dwarf)
    data_path = '/Users/elisedarragh-ford/Downloads/' + name + '/significance.txt'
    sig = np.loadtxt(data_path[0])
    significance.append(sig.item(nob)) 
    data_path = '/Users/elisedarragh-ford/Downloads/' + name + '/significance_cut.txt'
    sig_cut = np.loadtxt(data_path[0])
    significance_cut.append(sig_cut.item(0))
#    data_path = '/Users/elisedarragh-ford/Downloads/' + name + '/ss.txt'
#    ss = np.loadtxt(data_path[0])
    nstars.append(len(data)) 
    ra_peak.append(np.mean(data['ra']))
    dec_peak.append(np.mean(data['dec']))
    pmra_peak.append(np.mean(data['pmra']))
    pmdec_peak.append(np.mean(data['pmdec']))

    np.savetxt('ra_peak.txt', ra_peak)
    np.savetxt('dec_peak.txt', dec_peak)
    np.savetxt('pmra_peak.txt', pmra_peak)
    np.savetxt('pmdec_peak.txt', pmdec_peak)
    np.savetxt('significance_dwarfs.txt', significance)
    np.savetxt('significance_cut.txt', significance_cut)
    np.savetxt('nstars_dwarfs.txt', nstars)




