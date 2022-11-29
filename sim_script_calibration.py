import MCMC_calibration as MCSZ
import numpy as np
import matplotlib.pyplot as plt
import time
from astropy import units as u
import csv
import pandas as pd
from astropy.io import fits

#Import CMB data
fname = '/data/bolocam/bolocam/erapaport/cmb.fits'
hdu = fits.open(fname)
image_data = hdu[1].data

#Specify parameter values

#Galaxy cluster parameters
y = 5.4e-5
electron_temperature = 5.0 #KeV
tau = MCSZ.y_to_tau(y)
peculiar_vel = 0.000001 #km/s
betac = peculiar_vel/(3e5)

#SIDES model values
amp_sides = 1
b_sides = 1

#Main MCMC call with band definitions inside
def configuration_optimize(theta, long, lang, max_n, walkern, processors_pool): 
    start_time = time.time()
    mc = MCSZ.mcmc(theta, long, lang, max_n, walkern,processors_pool)
    print("--- %s seconds ---" % (time.time() - start_time))
    return mc

#MCMC hyperparameters

chain_length = 10000
walkers = 100
discard_n = 2000
thin_n = 100
processors_pool = 30

samples = [[0,0,0,0]]
samples = np.asarray(samples)

#Random SIDES pixel
sides_long = 120
sides_lang = 120

theta = (y, betac, amp_sides, b_sides)
sampler = configuration_optimize(theta, sides_long, sides_lang, chain_length, walkers,processors_pool)
samples = np.append(samples,sampler.get_chain(discard=discard_n, flat=True, thin=thin_n),axis=0)
    
#Write simulation output, change directory/name
with open('/data/bolocam/bolocam/erapaport/mcmc_calibration_009', 'w') as f:
    write = csv.writer(f)
    write.writerows(samples)
    
