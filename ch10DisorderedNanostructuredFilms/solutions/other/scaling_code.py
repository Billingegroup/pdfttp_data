import numpy as np
import matplotlib.pyplot as plt
import math

from scipy.optimize import curve_fit

# Code to scale PDF datasets:

def scaledPDFs(e,f,g):
    '''
    e: component PDF to be scaled 1
    f: component PDF to be scaled 2
    g: target PDF
    '''
    
    # LOAD DATA
    r, e = np.loadtxt(e, skiprows=27).transpose()
    r, f = np.loadtxt(f, skiprows=27).transpose()
    r, g = np.loadtxt(g, skiprows=27).transpose()

    # DEFINE FITTING FUNCTION TO SCALE SEPARATE PDFs
    def gaussian(r,scale1,scale2):
        model =  scale1*e + scale2*f
        return model

    # GIVE INITIAL GUESSES AND RUN MINIMIZATION
    pf = [1.0,1.0]
    popt, pcov = curve_fit(gaussian, r, g, p0=pf, sigma=None)

    # PRINT REFINED SCALE FACTORS AND RW VALUE
    rw = math.sqrt(np.sum((g - gaussian(r, *popt))**2/np.sum(f**2)))
    print(popt)
    print(rw)

    # PLOT AND SHOW DATA
    FIT = gaussian(r, *popt)
    plt.plot(r, popt[0]*e +2.5, c="cyan",   lw=2.0, label="component 1")
    plt.plot(r, popt[1]*f +2.5, c="purple", lw=2.0, label="component 2")
    plt.plot(r, g,            c="blue",   lw=2.0, label="data to be fit")
    plt.plot(r, FIT,            c="red",    lw=2.0, label="model")
    plt.plot(r, g - FIT -1.5,   c="green",  lw=2.0, label="difference")
    plt.legend(frameon=False, ncol=2, fontsize=10)
    plt.ylabel("G ($\AA^{-2}$)")
    plt.xlabel("r ($\AA$)")
    plt.show()

