import numpy as np
import math
import Mather_photonNEP12a as NEP
import pandas as pd
import csv
import sys, os
#Installed SZpack directory
sys.path.append("/home/bolocam/erapaport/Spectro-Imaging-Forecasting-Tool/codes/SZpack.v1.1.1/python")
import SZpack as sz
import emcee
from astropy.io import fits
from scipy.interpolate import interp1d
from multiprocessing import Pool

'''
Cosmological terms from G. Sun
'''

c = 299792458.0                                         # Speed of light - [c] = m/s
h_p = 6.626068e-34                                      # Planck's constant in SI units
k_b = 1.38065e-23                                       # Boltzmann constant in SI units
MJyperSrtoSI = 1E-20
GHztoHz = 1E9
h = 4.135*10**(-15) #in eV s
k = 8.617*10**(-5) # eV/K
m = 9.109*10**(-31) # kg
TCMB = 2.725 # K
# canonical CMB

def sigB(band_details, Time, Tnoise=3.0):
    #NESB values from the NEP and band definitions
    #Mode 0 specifies photometry
    #Mode 1 specifies spectrometry
    OE = 0.5
    scan_ineff = 0.1
    nu_min = band_details['nu_minGHz']*1E9 
    nu_max = band_details['nu_maxGHz']*1E9
    if (band_details['mode'] == 0):
        nu_vec = (nu_max + nu_min)/2.0
        del_nu = nu_max - nu_min
    if (band_details['mode'] == 1):
        nu_res = band_details['nu_resGHz']*1E9
        Nse = int(np.ceil((nu_max-nu_min)/nu_res))
        nu_vec = np.linspace(nu_min, nu_max, Nse) #in Hz
    NEP_phot = band_details['NEP']
    Npx = band_details['N_pixels']
    AOnu = (c/nu_vec)**2

    #NEP_phot1 = NEP.photonNEPdifflim(nu_min, nu_max, Tnoise) #This is CMB Tnoise
    #NEP_phot2 = NEP.photonNEPdifflim(nu_min, nu_max, 10.0, aef=0.01) #Use real south pole data
    #NEP_det = 1e-18 # ATTO WATTS per square-root(hz)
    #NEP_phot = np.sqrt(NEP_phot1**2 + NEP_det**2)
    
    if (NEP_phot<1E-18):
        print('Warning photon NEP is below 1 aW/rt.Hz!')
    elif (NEP_phot>1E-15):
        print('Warning photon NEP is above 1 fW/rt.Hz!')
       
    if (band_details['mode'] == 0):
        delP = NEP_phot/np.sqrt(Time*Npx)
        sigma_B = delP/(AOnu)/(del_nu*OE*scan_ineff)
    if (band_details['mode'] == 1):
        delP = 2*int(np.ceil(nu_max/nu_res))*NEP_phot/np.sqrt(Time*Npx)
        sigma_B = delP/(AOnu)/(nu_res*OE*scan_ineff)
       
    return nu_vec, sigma_B

#CMB Anisotropy function
def dB(dt, frequency):
    temp = TCMB/(1+dt)
    x = (h_p/(k_b * temp))*frequency
    I=((2*h_p)/(c**2))*(k_b*temp/h_p)**3
    return I*(x**4*np.exp(x)/((np.exp(x)-1)**2))*dt

#SZ total integration function
def szpack_signal(frequency, tau, temperature, betac):
    x_b = (h_p/(k_b * TCMB))*frequency
    original_x_b = (h_p/(k_b * TCMB))*frequency
    sz.compute_combo_means(x_b,tau,temperature,betac,0,0,0,0)
    return x_b*13.33914078*(TCMB**3)*(original_x_b**3)*1e-20

#Classical tSZ analytical calculation
def classical_tsz(y,frequency):
    x_b = (h_p/(k_b * TCMB))*frequency
    bv = 2*k_b*TCMB*((frequency**2)/(c**2))*(x_b/(np.exp(x_b)-1))
    return y*((x_b*np.exp(x_b))/(np.exp(x_b)-1))*(x_b*((np.exp(x_b)+1)/(np.exp(x_b)-1))-4)*bv

#Optical depth to compton-y
def tau_to_y(tau, temperature):
    if (tau == 0):
        return 0
    return (tau*(temperature*11604525.0061598)*k_b) / (m*(c**2))

#Compton-y to optical depth
def y_to_tau(y, temperature):
    if (y == 0):
        return 0
    return (m*(c**2)*y) / (k_b*(temperature*11604525.0061598))

#Interpolation function
def interpolate(freq, datay, datax):
    f = interp1d(np.log(datax),np.log(datay),kind='slinear',bounds_error=False,fill_value=0)
    new_data = f(np.log(freq))
    return np.exp(new_data)

#Model used for MCMC calculation
def model(theta, anisotropies, freq):
    y, temperature, betac, amp_sides, b_sides = theta
    ksz_anis, tsz_anis, cmb_anis = anisotropies
    
    #Read SIDES average model
    df = pd.read_csv('/home/bolocam/erapaport/OLIMPO-forecasting/templates/sides.csv',header=None) 
    data = df.to_numpy()
    data = data.squeeze()
    SIDES = data*1e-20
    
    #Modify template by amp_sides, b_sides 
    sides_template = amp_sides*interpolate(freq,SIDES,np.linspace(0,1500e9,751)*b_sides)
    
    #CMB and galaxy cluster SZ template
    cmb_template = dB(cmb_anis + ksz_anis,freq)
    sz_template = szpack_signal(freq, y_to_tau(y + tsz_anis,temperature), temperature, betac)
    
    template_total = sz_template + sides_template + cmb_template
    return template_total

#Individual templates for plotting using SIDES model
def model_indv(theta, anisotropies, freq):
    y, temperature, betac, amp_sides, b_sides = theta
    ksz_anis, tsz_anis, cmb_anis = anisotropies
    
    #Read SIDES average model
    df = pd.read_csv('/home/bolocam/erapaport/OLIMPO-forecasting/templates/sides.csv',header=None) 
    data = df.to_numpy()
    data = data.squeeze()
    SIDES = data*1e-20
    
    #Modify template by amp_sides, b_sides 
    sides_template = amp_sides*interpolate(freq,SIDES,np.linspace(0,1500e9,751)*b_sides)
    
    #CMB, ksz, tsz anisotropies, and galaxy cluster SZ template
    cmb_template = dB(cmb_anis,freq)
    sz_template = szpack_signal(freq, y_to_tau(y + tsz_anis,temperature), temperature, betac)
    ksz_template = dB(ksz_anis,freq)
    tsz_template = classical_tsz(tsz_anis,freq)
    
    template_total = [sz_template,sides_template,cmb_template, ksz_template, tsz_template]
    return template_total

#Individual templates for plotting using SIDES fits data
def data_indv(theta, anisotropies, freq, long, lang):
    y, temperature, betac, amp_sides, b_sides = theta
    ksz_anis, tsz_anis, cmb_anis = anisotropies
    
    #Read SIDES fits file with emission lines
    fname = '/home/bolocam/erapaport/OLIMPO-forecasting/templates/deLooze.fits'
    hdu = fits.open(fname)
    image_data = hdu[0].data
    total_SIDES = np.zeros(751)
    #Rebinning for 3 arcmin resolution
    for col in range(6):
        for row in range(6):
            total_SIDES += image_data[:,long + row,lang + col]*1e-20
    total_SIDES = total_SIDES/36
    
    #Interpolate for specific frequencies
    sides_template = interpolate(freq,total_SIDES,np.linspace(0,1500e9,751))
    
    #CMB, ksz, tsz anisotropies, and galaxy cluster SZ template
    cmb_template = dB(cmb_anis,freq)
    sz_template = szpack_signal(freq, y_to_tau(y + tsz_anis,temperature), temperature, betac)
    ksz_template = dB(ksz_anis,freq)
    tsz_template = classical_tsz(tsz_anis,freq)
    
    template_total = [sz_template,sides_template,cmb_template, ksz_template, tsz_template]
    return template_total

def log_likelihood(theta, anisotropies, freq, data, noise):
    modeldata = model(theta,anisotropies, freq)
    return -0.5 * np.sum(((data - modeldata)/noise)**2)

#Change priors if needed
def log_prior(theta):
    y, temperature, betac, amp_sides, b_sides = theta
    
    if (y < 0 or y > 0.1):
        return -np.inf
    if (betac < -0.2 or betac > 0.2):
        return -np.inf
    if (temperature < 2.73 or temperature > 20):
        return -np.inf
    if (amp_sides < 0 or amp_sides > 2.5):
        return -np.inf
    if (b_sides < 0 or b_sides > 2.5):
        return -np.inf
    mu = 5.0
    sigma = 0.5
    #Gaussian prior on temperature with mean of mu and std. deviation of sigma
    return np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(temperature-mu)**2/sigma**2

def log_probability(theta, anisotropies,freq, data, noise):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, anisotropies, freq, data, noise)            

#Main MCMC code
def mcmc(Time, theta, anisotropies, Bands_list, long, lang, max_n, walkern, processors):
    y, temperature, betac, amp_sides, b_sides = theta
    ksz_anis, tsz_anis, cmb_anis = anisotropies

    nu_total_array = np.empty(0)
    total_sz_array = np.empty(0)
    sigma_b_array = np.empty(0)
    
    #Create list of frequencies and NESB 
    for bb in range(len(Bands_list)):
        b = Bands_list[bb]
        nu_vec_b, sigma_B_b = sigB(b, Time)
        nu_total_array = np.concatenate((nu_total_array, nu_vec_b),axis=None)
        sigma_b_array = np.concatenate((sigma_b_array, sigma_B_b),axis=None)
   
    #Get signals and foregrounds
    
    #Read SIDES fits file with emission lines
    fname = '/home/bolocam/erapaport/OLIMPO-forecasting/templates/deLooze.fits'
    hdu = fits.open(fname)
    image_data = hdu[0].data
    total_SIDES = np.zeros(751)
    #Rebinning for 3 arcmin resolution
    for col in range(6):
        for row in range(6):
            total_SIDES += image_data[:,long + row,lang + col]*1e-20
    total_SIDES = total_SIDES/36
    
    #Interpolate for specific frequencies
    sides_template = interpolate(nu_total_array,total_SIDES,np.linspace(0,1500e9,751))
    
    #CMB and galaxy cluster SZ template
    sz_template = szpack_signal(nu_total_array,y_to_tau(y + tsz_anis,temperature),temperature,betac)
    cmb_template = dB(cmb_anis + ksz_anis,nu_total_array)
    total_sz_array = sz_template + cmb_template + sides_template
        
    pos = []
    for item in theta:
        pos.append(item*(1 + 0.01*np.random.randn(walkern)))
    pos_array= np.asarray(pos)
    pos_array = pos_array.transpose()
    nwalkers, ndim = pos_array.shape

    with Pool(processors) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(anisotropies, nu_total_array, total_sz_array, sigma_b_array),pool=pool)
        for sample in sampler.sample(pos_array, iterations=max_n, progress=True):
            continue
        
    return sampler
