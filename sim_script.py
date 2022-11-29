import MCMC as MCSZ
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
y = 2e-6
electron_temperature = 0.5 #KeV
tau = MCSZ.y_to_tau(y,electron_temperature)
peculiar_vel = 0.000001 #km/s
betac = peculiar_vel/(3e5)

#SIDES model values
amp_sides = 1
b_sides = 1

#Define frequencies and rms

#rms_values = [0.36, 0.27, 0.70, 1.76] #uK^2 for 80 hour integration
#rms_values = [0.18, 0.135, 0.35, 0.88] #uK^2 for 320 hour integration
#rms_values = [0.36*np.sqrt(2), 0.27*np.sqrt(2), 0.70*np.sqrt(2), 1.76*np.sqrt(2)] #uK^2 for 40 hour integration
rms_values = [0.72, 0.54, 1.40, 3.52] #uK^2 for 20 hour integration


frequencies = [145,250,365,460]

#MCMC hyperparameters

chain_length = 10000
walkers = 100
realizations = 40
discard_n = 2000
thin_n = 100
processors_pool = 30

#Read previous parameters if needed
df = pd.read_csv('/data/bolocam/bolocam/erapaport/mcmc_run009_parameters',header=None) 
params = df.to_numpy()

# # Comment out if using previous parameters
# parameter_matrix = np.empty((realizations,8))

samples = [[0,0,0,0,0]]
samples = np.asarray(samples)

for i in range(realizations):

#     #Random cmb anis. value
#     cmb_long = np.random.randint(2,197)
#     cmb_lat = np.random.randint(2,197)
#     amp_cmb = image_data[cmb_long][cmb_lat] - np.mean(image_data[cmb_long-2:cmb_long+3,cmb_lat-2:cmb_lat+3])

#     #Random SIDES pixel
#     sides_long = np.random.randint(0,154)
#     sides_lat = np.random.randint(0,154)

#     #Random cmb ksz,tsz anis. values
#     amp_ksz = np.random.normal(0,6e-7)
#     amp_tsz = np.random.normal(0,6e-7)
#     parameter_matrix[i] = [y, electron_temperature, betac, sides_long, sides_lat, amp_cmb, amp_ksz, amp_tsz]
    
    #Uncomment if using previous parameters
    amp_cmb = params[i,5]
    sides_long = int(params[i,3])
    sides_lat = int(params[i,4])
    amp_ksz = params[i,6]
    amp_tsz= params[i,7]
    
    theta = (y, electron_temperature, betac, amp_sides, b_sides)
    anisotropies = (amp_ksz, amp_tsz, amp_cmb)
    start_time = time.time()
    sampler = MCSZ.mcmc(theta, anisotropies,  rms_values, frequencies, sides_long, sides_lat, chain_length, walkers, processors_pool)
    print("--- %s seconds ---" % (time.time() - start_time))
    samples = np.append(samples,sampler.get_chain(discard=discard_n, flat=True, thin=thin_n),axis=0)
    
#Write simulation output, change directory/name
with open('/data/bolocam/bolocam/erapaport/mcmc_run_037', 'w') as f:
    write = csv.writer(f)
    write.writerows(samples)
    
# #Write parameters if needed, change directory/name
# with open('/data/bolocam/bolocam/erapaport/mcmc_run012_parameters', 'w') as f:
#     write = csv.writer(f)
#     write.writerows(parameter_matrix)
