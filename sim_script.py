import MCMC as MCSZ
import numpy as np
import matplotlib.pyplot as plt
import time
from astropy import units as u
import csv
import pandas as pd
from astropy.io import fits

#Import CMB data
fname = '/home/bolocam/erapaport/OLIMPO-forecasting/templates/cmb.fits'
hdu = fits.open(fname)
image_data = hdu[1].data

#Specify parameter values

#Galaxy cluster parameters
y = 1e-4
electron_temperature = 5.0 #KeV
tau = MCSZ.y_to_tau(y,electron_temperature)
peculiar_vel = 1100 #km/s
betac = peculiar_vel/(3e5)

#SIDES model values
amp_sides = 1
b_sides = 1

#Integration time

Time = 86400*40 #40 hours

#Main MCMC call with band definitions inside
def configuration_optimize(Time, theta, anisotropies, Bands_list, long, lang, max_n, walkern, processors_pool): 
    start_time = time.time()
    mc = MCSZ.mcmc(Time, theta, anisotropies, Bands_list, long, lang, max_n, walkern,processors_pool)
    print("--- %s seconds ---" % (time.time() - start_time))
    return mc

#NEP definitions
NEP_1 = 7.3e-18
NEP_2 = 1.3e-17
NEP_3 = 7.5e-18
NEP_4 = 1.2e-17

#Pixel count definitions
pix1 = 23
pix2 = 39
pix3 = 25
pix4 = 43

#Resolution definitions (Spectrometry)
res1 = 29.0
res2 = 89.0
res3 = 49.0
res4 = 49.0

#Spectrometry bands
# Bands_list = [{'mode':1,'name':'Band 1','nu_minGHz':130,'nu_maxGHz':160,'nu_resGHz':res1,'N_pixels':pix1, 'NEP':NEP_1},\
#       {'mode':1,'name':'Band 2','nu_minGHz':190,'nu_maxGHz':280,'nu_resGHz':res2,'N_pixels':pix2, 'NEP':NEP_2},\
#       {'mode':1,'name':'Band 3','nu_minGHz':310,'nu_maxGHz':360,'nu_resGHz':res3,'N_pixels':pix3, 'NEP':NEP_3},\
#               {'mode':1,'name':'Band 4','nu_minGHz':420,'nu_maxGHz':470,'nu_resGHz':res4,'N_pixels':pix4, 'NEP':NEP_4}]

#Photometry bands
Bands_list = [{'mode':0,'name':'Band 1','nu_minGHz':133,'nu_maxGHz':159,'N_pixels':pix1,'NEP':NEP_1},\
     {'mode':0,'name':'Band 2','nu_minGHz':192,'nu_maxGHz':304,'N_pixels':pix2,'NEP':NEP_2},\
     {'mode':0,'name':'Band 3','nu_minGHz':330,'nu_maxGHz':362,'N_pixels':pix3,'NEP':NEP_3},\
             {'mode':0,'name':'Band 4','nu_minGHz':457,'nu_maxGHz':515,'N_pixels':pix4,'NEP':NEP_4}]

#MCMC hyperparameters

chain_length = 5000
walkers = 50
realizations = 40
discard_n = 500
thin_n = 50
processors_pool = 12

#Read previous parameters if needed
# df = pd.read_csv('/data/bolocam/bolocam/erapaport/mcmc_test_run001_parameters',header=None) 
# params = df.to_numpy()

#Comment out if using previous parameters
parameter_matrix = np.empty((realizations,8))

#Random cmb anis. value
cmb_long = np.random.randint(2,197)
cmb_lang = np.random.randint(2,197)
amp_cmb = image_data[cmb_long][cmb_lang] - np.mean(image_data[cmb_long-2:cmb_long+3,cmb_lang-2:cmb_lang+3])

#Random SIDES pixel
sides_long = np.random.randint(0,154)
sides_lang = np.random.randint(0,154)

#Random cmb ksz,tsz anis. values
amp_ksz = np.random.normal(0,6e-7)
amp_tsz = np.random.normal(0,6e-7)
parameter_matrix[0] = [y, electron_temperature, betac, sides_long, sides_lang, amp_cmb, amp_ksz, amp_tsz]

# Uncomment if using previous parameters
# amp_cmb = params[0,5]
# sides_long = int(params[0,3])
# sides_lang = int(params[0,4])
# amp_ksz = params[0,6]
# amp_tsz= params[0,7]

theta = (y, electron_temperature, betac, amp_sides, b_sides)
anisotropies = (amp_ksz, amp_tsz, amp_cmb)
sampler = configuration_optimize(Time, theta, anisotropies, Bands_list, sides_long, sides_lang, chain_length, walkers,processors_pool)
samples = sampler.get_chain(discard=discard_n, flat=True, thin=thin_n)

for i in range(realizations-1):
    #Comment out if using previous parameters
    parameter_matrix = np.empty((realizations,8))

    #Random cmb anis. value
    cmb_long = np.random.randint(2,197)
    cmb_lang = np.random.randint(2,197)
    amp_cmb = image_data[cmb_long][cmb_lang] - np.mean(image_data[cmb_long-2:cmb_long+3,cmb_lang-2:cmb_lang+3])

    #Random SIDES pixel
    sides_long = np.random.randint(0,154)
    sides_lang = np.random.randint(0,154)

    #Random cmb ksz,tsz anis. values
    amp_ksz = np.random.normal(0,6e-7)
    amp_tsz = np.random.normal(0,6e-7)
    parameter_matrix[i+1] = [y, electron_temperature, betac, sides_long, sides_lang, amp_cmb, amp_ksz, amp_tsz]
    
    # Uncomment if using previous parameters
    # amp_cmb = params[i+1,5]
    # sides_long = int(params[i+1,3])
    # sides_lang = int(params[i+1,4])
    # amp_ksz = params[i+1,6]
    # amp_tsz= params[i+1,7]
    
    theta = (y, electron_temperature, betac, amp_sides, b_sides)
    anisotropies = (amp_ksz, amp_tsz, amp_cmb)
    sampler = configuration_optimize(Time, theta, anisotropies, Bands_list, sides_long, sides_lang, chain_length, walkers,processors_pool)
    samples = np.append(samples,sampler.get_chain(discard=discard_n, flat=True, thin=thin_n),axis=0)
    
#Write simulation output, change directory/name
with open('/data/bolocam/bolocam/erapaport/mcmc_test_run001', 'w') as f:
    write = csv.writer(f)
    write.writerows(samples)
    
#Write parameters if needed, change directory/name
# with open('/data/bolocam/bolocam/erapaport/mcmc_test_run001_parameters', 'w') as f:
#     write = csv.writer(f)
#     write.writerows(parameter_matrix)
