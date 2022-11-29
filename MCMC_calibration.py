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

electron_temperature = 5.0 #KeV

#CMB Anisotropy function
def dB(dt, frequency):
    temp = TCMB/(1+dt)
    x = (h_p/(k_b * temp))*frequency
    I=((2*h_p)/(c**2))*(k_b*temp/h_p)**3
    return I*(x**4*np.exp(x)/((np.exp(x)-1)**2))*dt

#SZ total integration function
def szpack_signal(frequency, tau, betac):
    x_b = (h_p/(k_b * TCMB))*frequency
    original_x_b = (h_p/(k_b * TCMB))*frequency
    sz.compute_combo_means(x_b,tau,electron_temperature,betac,0,0,0,0)
    return x_b*13.33914078*(TCMB**3)*(original_x_b**3)*1e-20

#Classical tSZ analytical calculation
def classical_tsz(y,frequency):
    x_b = (h_p/(k_b * TCMB))*frequency
    bv = 2*k_b*TCMB*((frequency**2)/(c**2))*(x_b/(np.exp(x_b)-1))
    return y*((x_b*np.exp(x_b))/(np.exp(x_b)-1))*(x_b*((np.exp(x_b)+1)/(np.exp(x_b)-1))-4)*bv

#Optical depth to compton-y
def tau_to_y(tau):
    if (tau == 0):
        return 0
    return (tau*(electron_temperature*11604525.0061598)*k_b) / (m*(c**2))

#Compton-y to optical depth
def y_to_tau(y):
    if (y == 0):
        return 0
    return (m*(c**2)*y) / (k_b*(electron_temperature*11604525.0061598))

#Interpolation function
def interpolate(freq, datay, datax):
    f = interp1d(np.log(datax),np.log(datay),kind='slinear',bounds_error=False,fill_value=0)
    new_data = f(np.log(freq))
    return np.exp(new_data)

#Model used for MCMC calculation
def model(theta, freq):
    y, betac, amp_sides, b_sides = theta
    
    #Read SIDES average model
    df = pd.read_csv('/data/bolocam/bolocam/erapaport/sides.csv',header=None) 
    data = df.to_numpy()
    data = data.squeeze()
    SIDES = data*1e-20
    
    #Modify template by amp_sides, b_sides 
    sides_template = amp_sides*interpolate(freq,SIDES,np.linspace(0,1500e9,751)*b_sides)
    
    #CMB and galaxy cluster SZ template
    sz_template = szpack_signal(freq, y_to_tau(y), betac)
    
    template_total = sz_template + sides_template
    return template_total

#Individual templates for plotting using SIDES model
def model_indv(theta, freq):
    y, betac, amp_sides, b_sides = theta
    
    #Read SIDES average model
    df = pd.read_csv('/data/bolocam/bolocam/erapaport/sides.csv',header=None) 
    data = df.to_numpy()
    data = data.squeeze()
    SIDES = data*1e-20
    
    #Modify template by amp_sides, b_sides 
    sides_template = amp_sides*interpolate(freq,SIDES,np.linspace(0,1500e9,751)*b_sides)
    
    #CMB, ksz, tsz anisotropies, and galaxy cluster SZ template
    sz_template = szpack_signal(freq, y_to_tau(y), betac)
    
    template_total = [sz_template,sides_template]
    return template_total

#Individual templates for plotting using SIDES fits data
def data_indv(theta, freq, long, lang):
    y, betac, amp_sides, b_sides = theta
    
    #Read SIDES fits file with emission lines
    fname = '/data/bolocam/bolocam/erapaport/continuum.fits'
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
    sz_template = szpack_signal(freq, y_to_tau(y), betac)
    
    template_total = [sz_template,sides_template]
    return template_total

def log_likelihood(theta, freq, data, noise):
    modeldata = model(theta, freq)
    return -0.5 * np.sum(((data - modeldata)/noise)**2)

#Change priors if needed
def log_prior(theta):
    y, betac, amp_sides, b_sides = theta
    
    if (y < 0 or y > 0.1):
        return -np.inf
    if (betac < -0.02 or betac > 0.02):
        return -np.inf
    if (amp_sides < 0 or amp_sides > 2.5):
        return -np.inf
    if (b_sides < 0 or b_sides > 2.5):
        return -np.inf
    return 0

def log_probability(theta,freq, data, noise):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, freq, data, noise)            

#Main MCMC code
def mcmc(theta, long, lang, max_n, walkern, processors):
    y, betac, amp_sides, b_sides = theta
    
    nu_total_array = np.array([145.0,250.0,365.0,460.0])*1e9
    sigma_b_array = np.array([1.38717811e-24,1.26101318e-24,1.91851812e-24,2.27755595e-24])

    #Get signals and foregrounds
    
    #Read SIDES fits file with emission lines
    fname = '/data/bolocam/bolocam/erapaport/continuum.fits'
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
    sz_template = szpack_signal(nu_total_array,y_to_tau(y),betac)
    total_sz_array = sz_template + sides_template
        
    pos = []
    for item in theta:
        pos.append(item*(1 + 0.01*np.random.randn(walkern)))
    pos_array= np.asarray(pos)
    pos_array = pos_array.transpose()
    nwalkers, ndim = pos_array.shape

    with Pool(processors) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(nu_total_array, total_sz_array, sigma_b_array),pool=pool)
        for sample in sampler.sample(pos_array, iterations=max_n, progress=True):
            continue
        
    return sampler
