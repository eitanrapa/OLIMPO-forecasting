#from numba import jit
# if this gets slow think of using numba
import numpy as np
import scipy.constants as cons
from scipy import integrate

def fancyJ1(xmin, xmax, aef):
    funnyexp1 = lambda x: aef*(x**4)/(np.exp(x)-1.0)*(1.0+aef/(np.exp(x)-1.0))
    temp = integrate.quad(funnyexp1, xmin ,  xmax)
    return temp

def fancyJ2(xmin, xmax, aef):
    funnyexp2 = lambda x: aef*(x**2)/(np.exp(x)-1.0)*(1.0+aef/(np.exp(x)-1.0))
    temp = integrate.quad(funnyexp2, xmin ,  xmax)
    return temp

def photonNEPexact(nu_min, nu_max, AOmega, Tsys, aef= 1.0, npol = 2):
    #this is the exact equation, no approximations, note AO is outisde integral
    kB = cons.Boltzmann
    h = cons.Planck
    c = cons.speed_of_light

    x_min = nu_min*h/(kB*Tsys)
    x_max = nu_max*h/(kB*Tsys)
    coeff = 2*npol*AOmega*((kB*Tsys)**5)/((c**2)*(h**3))

    fJ1 = fancyJ1(x_min, x_max, aef)
    temp = coeff*fJ1[0] # this is NEP^2

    return np.sqrt(temp)

def photonNEPold(nu_min, nu_max, nu_peak, Tsys, aef= 1.0, npol = 2):
    # input arguments are nu_min, nu_max, nu_peak, Tsys, aef= 1.0, npol = 2
    # AO/c^2 is treated as 1/nu_peak^2
    kB = cons.Boltzmann
    h = cons.Planck
    c = cons.speed_of_light

    x_min = nu_min*h/(kB*Tsys)
    x_max = nu_max*h/(kB*Tsys)
    x_peak = nu_peak*h/(kB*Tsys)

    fJ1 = fancyJ1(x_min, x_max,aef)
    temp = 2*npol*(x_peak**-2)*((kB*Tsys)**3)*(1/h)*fJ1[0] # this is NEP^2

    return np.sqrt(temp)

def photonNEPdifflim(nu_min, nu_max, Tsys, aef= 1.0, npol = 2):
    # input arguments are nu_min, nu_max, Tsys, aef= 1.0, npol = 2
    # AO/c^2 is treated as 1/nu_^2 in the integral
    kB = cons.Boltzmann
    h = cons.Planck
    c = cons.speed_of_light

    x_min = nu_min*h/(kB*Tsys)
    x_max = nu_max*h/(kB*Tsys)


    fJ2 = fancyJ2(x_min, x_max,aef)
    temp = 2*npol*((kB*Tsys)**3)*(1/h)*fJ2[0] # this is NEP^2

    return np.sqrt(temp)
