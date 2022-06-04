"""
Please see notes in Chapter 3 of the 'PDF to the People' book for further
explanation of the code.

This Diffpy-CMI script will carry out a structural refinement of a measured
PDF from crystalline alpha zirconium phosphate.  It is the same refinement as is done
using PDFgui in this chapter of the book, only this time using Diffpy-CMI
"""
# 1: Import relevant system packages that we will need...
import os
import numpy as np
from pyobjcryst.crystal import CreateCrystalFromCIF
from scipy.optimize import least_squares
import matplotlib as mpl
import matplotlib.pyplot as plt

# ... and the relevant CMI packages
from diffpy.srfit.pdf import PDFGenerator, DebyePDFGenerator, PDFParser
from diffpy.srfit.pdf.characteristicfunctions import sphericalCF
from diffpy.srfit.fitbase import Profile
from diffpy.srfit.fitbase import FitContribution, FitRecipe
from diffpy.srfit.fitbase import FitResults

from diffpy.Structure import Structure
from diffpy.srfit.fitbase.fithook import PlotFitHook

############### Config ##############################
# Give a file path to where your PDF (.gr) and structure (.cif) files are located.
data_file = "../../data/pdfs/aZrP.gr"
stru_file = Structure(filename="../../data/structures/alpha_ZrP.cif")
basename = "alpha-ZrP"

######## Experimental PDF Config ######################
# Specify the min, max, and step r-values of the PDF (that we want to fit over)
# also, specify the Q_max and Q_min values used to reduce the PDF.
PDF_RMIN = 1.0
PDF_RMAX = 35
PDF_RSTEP = 0.01
QMAX = 18.0
QMIN = 0.7

########PDF initialize refinable variables #############
# We explicitly specify the lattice parameters, scale,
# isotropic thermal parameters, and a correlated motion parameter.
LAT_A_I = 9.035
LAT_B_I = 5.257
LAT_C_I = 16.198
LAT_BETA_I = 111.43
SCALE_I = 0.7754
UISO_I = 0.008
DELTA1_I = 1.0
PSIZE_I = 300


# Instrumental will be fixed based on values obtained from a
# separate calibration step. These are hard-coded here.
QDAMP_I = 0.0382204955745
QBROAD_I = 0.0192369046067

# If we want to run using multiprocessors, we can switch this to 'True'.
# This requires that the 'psutil' python package installed.
RUN_PARALLEL = True

######## Functions that will carry out the refinement ##################
# We define a function 'make_recipe' to make the recipe that the fit will follow.
def make_recipe(stru_file, data_file):
    """
    Creates and returns a Fit Recipe object

    Parameters
    ----------
    stru_file : string, The full path to the structure file to load.
    data_file :  string, The full path to the PDF data to be fit.

    Returns
    ----------
    recipe :    The initialized Fit Recipe object using the datname and structure path
                provided.
    """

    
    structure = Structure(stru_file)

    # Create a Profile object for the experimental dataset and
    # tell this profile the range and mesh of points in r-space.
    profile = Profile()
    parser = PDFParser()
    parser.parseFile(data_file)
    profile.loadParsedData(parser)
    profile.setCalculationRange(xmin=PDF_RMIN, xmax = PDF_RMAX, dx = PDF_RSTEP)

    # Create a PDF Generator object for a periodic structure model.
    generator_crystal = PDFGenerator("G_crystal")
    generator_crystal.setStructure(structure, periodic=True)

    # If you have a multi-core computer (you probably do), run your refinement in parallel!
    if RUN_PARALLEL:
        try:
            import psutil
            import multiprocessing
            from multiprocessing import Pool
            syst_cores = multiprocessing.cpu_count()
            cpu_percent = psutil.cpu_percent()
            avail_cores = np.floor((100 - cpu_percent) / (100.0 / syst_cores))
            ncpu = int(np.max([1, avail_cores]))
            pool = Pool(processes=ncpu)
            generator_crystal.parallel(ncpu=ncpu, mapfunc=pool.map)
        except ImportError:
            print("\nYou don't appear to have the necessary packages for parallelization")

    # Generate PDF fit function:
    contribution = FitContribution("crystal")
    contribution.addProfileGenerator(generator_crystal)
    contribution.setProfile(profile, xname = "r")
    contribution.registerFunction(sphericalCF, name = "f")
    contribution.setEquation("scale * f * (G_crystal)")

    # Create the Fit Recipe object that holds all the details of the fit.    
    recipe = FitRecipe()
    recipe.addContribution(contribution)

    # Experimental parameters:
    generator_crystal.setQmax(QMAX)
    #generator_crystal.setQmin(0.0) # set to zero for bulk material
    generator_crystal.qdamp.value = QDAMP_I
    generator_crystal.qbroad.value = QBROAD_I
    
    # Profile parameters:
    recipe.addVar(contribution.scale, SCALE_I, tag = "scale")
    recipe.addVar(contribution.psize, PSIZE_I, tag = "scale")

    # Lattice parameters:
    phase_crystal = generator_crystal.phase
    lat = phase_crystal.getLattice()
    recipe.addVar(lat.a, value = LAT_A_I, tag = "lat")
    recipe.addVar(lat.b, value = LAT_B_I, tag = "lat")
    recipe.addVar(lat.c, value = LAT_C_I, tag = "lat")
    recipe.addVar(lat.beta, value = LAT_BETA_I, tag = "lat")

    # Thermal parameters (ADPs):    
    atoms = phase_crystal.getScatterers()
    recipe.newVar("Zr_U11",  UISO_I, tag = "adp")
    recipe.newVar("P_U11",  UISO_I, tag = "adp")
    recipe.newVar("O_U11", UISO_I, tag = "adp")

    for atom in atoms:
        if atom.element.title() == "Zr":
            recipe.constrain(atom.Uiso, "Zr_U11")
        elif atom.element.title() == "P":
            recipe.constrain(atom.Uiso, "P_U11")
        elif atom.element.title() == "O":
            recipe.constrain(atom.Uiso, "O_U11")
            
    # Correlated motion parameter:
    recipe.addVar(generator_crystal.delta1, 
                  name = "delta1_crystal", value = DELTA1_I, tag = "d1")

    return recipe

# We create a function to plot and save the results of the fit.
def plotResults(recipe):
    """Plot the results contained within a refined FitRecipe."""

    r = recipe.crystal.profile.x

    g = recipe.crystal.profile.y
    gcalc = recipe.crystal.profile.ycalc
    diffzero = -0.8 * max(g) * np.ones_like(g)
    diff = g - gcalc + diffzero

    plt.plot(r,g,'bo',label="G(r) Data")
    plt.plot(r, gcalc,'r-',label="G(r) Fit")
    plt.plot(r,diff,'g-',label="G(r) diff")
    plt.plot(r,diffzero,'k-')
    plt.xlabel("$r (\AA)$")
    plt.ylabel("$G (\AA^{-2})$")
    plt.legend(loc=1)
    plt.savefig(basename+".pdf", format="pdf")
    plt.show()
    return


def main():
    # Make the recipe
    recipe = make_recipe(stru_file, data_file)
    recipe.fithooks[0].verbose = 3

    # As before, we fix all parameters, create a list of tags and,
    # loop over them refining sequentially. In this example, we've added
    # 'psize' because we want to refine the nanoparticle size.
    recipe.fix("all")
    tags = ["lat", "scale", "psize", "adp", "d1", "all"]
    for tag in tags:
    	recipe.free(tag)
        least_squares(recipe.residual, recipe.values, x_scale="jac")

    # Save structures
    stru_file.write(basename + ".stru", "pdffit")
    profile = recipe.crystal.profile

    #import IPython.Shell; IPython.Shell.IPShellEmbed(argv=[])()
    profile.savetxt(basename + ".fit")

    # Generate and print the FitResults
    res = FitResults(recipe)
    res.printResults()

    header = "crystal_HF.\n"
    res.saveResults(basename + ".res", header=header)

    # Plot!
    plotResults(recipe)

if __name__ == "__main__":

    main()
# End of file
