import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import bagpipes as pipes
import sys
from astropy.table import Table
from astropy.io import ascii
from astropy.io import fits
from load_data import load_data



def bin(spectrum, binn):
    """ Bins up two or three column spectral data by a specified factor. """

    binn = int(binn)
    nbins = len(spectrum) // binn
    binspec = np.zeros((nbins, spectrum.shape[1]))

    for i in range(binspec.shape[0]):
        spec_slice = spectrum[i*binn:(i+1)*binn, :]
        binspec[i, 0] = np.mean(spec_slice[:, 0])
        binspec[i, 1] = np.mean(spec_slice[:, 1])

        if spectrum.shape[1] == 3:
            binspec[i,2] = (1./float(binn)
                            *np.sqrt(np.sum(spec_slice[:, 2]**2)))

    return binspec

def Jy_to_f_lam(wave,f_nu):
    #f_nu [Jy]
    #wave [Angs]
    return 2.99792458e+21 * 10**(-26) * f_nu / wave**2

def load_spec(ID):
    """ Loads CEERS spectroscopic data from file. """

    hdulist = fits.open("/Volumes/Samsung_T5/JWST/CEERS_NIRSpec_prism/readable/ceers_" + ID + ".fits")


    nan_mask = np.isnan(Jy_to_f_lam(hdulist[1].data["WAVELENGTH"]*1e4,hdulist[1].data["FLUX"]))

    wave = hdulist[1].data["WAVELENGTH"]*1e4
    flux = Jy_to_f_lam(hdulist[1].data["WAVELENGTH"]*1e4,hdulist[1].data["FLUX"])
    flux_error = Jy_to_f_lam(hdulist[1].data["WAVELENGTH"]*1e4,hdulist[1].data["FLUX_ERROR"])

    spectrum = np.c_[wave[~nan_mask], #convert to Angstroms
                     flux[~nan_mask],
                     flux_error[~nan_mask]]

    #mask = (spectrum[:,0] < 9250.) & (spectrum[:,0] > 5250.)

    return bin(spectrum, 2)


def get_fit_instructions():
    """ Set up the desired fit_instructions dictionary. """

    dust = {}
    dust["type"] = "CF00"
    dust["eta"] = 2#(1., 3.)
    dust["Av"] = (0., 2.)
    dust["n"] = (0.3, 2.5)
    dust["n_prior"] = "Gaussian"
    dust["n_prior_mu"] = 0.7
    dust["n_prior_sigma"] = 0.3

    zmet_factor = (0.02/0.014)

    nebular = {}
    nebular["logU"] = -3#(-4., -2.)

    constant = {}
    constant["massformed"] = (5., 11.)
    constant["metallicity"] = 0.2#(0.01/zmet_factor, 3.5/zmet_factor)
    #constant["metallicity_prior"] = "log_10"
    constant["age_min"] = 0.
    constant["age_max"] = (0.001, 10.)
    constant["age_max_prior"] = "log_10"

    fit_instructions = {}
    fit_instructions["dust"] = dust
    fit_instructions["constant"] = constant
    #fit_instructions["dblplaw"] = dblplaw
    fit_instructions["nebular"] = nebular
    fit_instructions["t_bc"] = 0.01
    fit_instructions["redshift"] = (0, 12)

    hdul = fits.open("/Volumes/Samsung_T5/JWST/CEERS_NIRSpec_prism/bagpipes/jwst_nirspec_prism_disp.fits")
    fit_instructions["R_curve"] = np.c_[10000*hdul[1].data["WAVELENGTH"], hdul[1].data["R"]]

    fit_instructions["veldisp"] = (1., 1000.)   #km/s
    fit_instructions["veldisp_prior"] = "log_10"

    return fit_instructions



# Load list of objects to be fitted from catalogue.
cat = pd.read_csv("/Volumes/Samsung_T5/JWST/CEERS_NIRSpec_prism/CEERS_prism+phot_catalog.csv", delimiter=",")
cat.index = cat["MSA_ID"].astype(str).values

IDs = ["80073"]



galaxy = pipes.galaxy(IDs[0], load_spec, photometry_exists=False)
galaxy.plot()
plt.show()


fit_instructions = get_fit_instructions()
fit = pipes.fit(galaxy, fit_instructions, run="spectroscopy_8")

fit.fit(verbose=False)
