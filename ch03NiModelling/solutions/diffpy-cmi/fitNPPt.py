"""
Please see notes in Chapter 3 of the 'PDF to the People' book for additonal
explanation of the code.

This Diffpy-CMI script will carry out a structural refinement of a measured
PDF from nanocrystalline platinum.  It is the same refinement as is done
using PDFgui in this chapter of the book, only this time using Diffpy-CMI.
It is required that "fitBulkNi.py" be run prior to running this example!
"""
# 1: Import relevant system packages that we will need...
from pathlib import Path
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

# ... and the relevant CMI packages
from diffpy.srfit.fitbase import FitContribution, FitRecipe, FitResults, Profile
from diffpy.srfit.pdf import PDFParser, PDFGenerator
from diffpy.structure.parsers import getParser
from diffpy.srfit.pdf.characteristicfunctions import sphericalCF
from diffpy.srfit.structure import constrainAsSpaceGroup

############### Config ##############################
# 2: Give a file path to where your PDF (.gr) and structure (.cif) files are located.
PWD = Path(__file__).parent.absolute()
DPATH = PWD.parent.parent / "data"

# 3: Give an identifying name for the refinement.
FIT_ID = "Fit_Pt_NP"

# 4: Specify the names of the input PDF and CIF files.
GR_NAME = "Pt-nanoparticles.gr"
CIF_NAME = "Pt.cif"

######## Experimental PDF Config ######################
# 5: Specify the min, max, and step r-values of the PDF (that we want to fit over)
# also, specify the Q_max and Q_min values used to reduce the PDF.
PDF_RMIN = 1.5
PDF_RMAX = 50
PDF_RSTEP = 0.01
QMAX = 25
QMIN = 0.1

########PDF initialize refinable variables #############
# 6: We explicitly specify the lattice parameters, scale,
# isotropic thermal parameters, and a correlated motion parameter.
CUBICLAT_I = 3.9
SCALE_I = 0.6
UISO_I = 0.01
DELTA2_I = 4

# 7: For the nanoparticle (NP) case, also provide an initial guess
# for the average crystallite size, in Angstrom.
PSIZE_I = 40

# 8: First, let's read the fit results from the Ni fit.
# We parse out the refined values of Q_damp and Q_broad,
# instrumental parameters which will be fixed in this fit.
# This is a bit of python input/output and regular expression
# parsing, which we won't cover here.
RESDIR = PWD / "res"
NIBASENAME = "Fit_Ni_Bulk"

RES_FILE = RESDIR / f"{NIBASENAME}.res"

if RES_FILE.exists():
    with open(RES_FILE, "r") as file:
        for line in file:
            if line.split(" ")[0] == "Calib_Qbroad":
                QBROAD_I = float(re.split(" +", line)[1])
            elif line.split(" ")[0] == "Calib_Qdamp":
                QDAMP_I = float(re.split(" +", line)[1])
else:
    print("Ni example does not appear to be run\n")
    print("The Ni example refines instrument parameters\n")
    print("The instrument parameters are necessary to run this fit\n")
    print("Please run the Ni example first\n")

# If we want to run using multiprocessors, we can switch this to 'True'.
# This requires that the 'psutil' python package installed.
RUN_PARALLEL = False


######## Functions that will carry out the refinement ##################
# 9: We define a function 'make_recipe' to make the recipe that the fit will follow.
def make_recipe(cif_path, dat_path):
    """
    Creates and returns a Fit Recipe object

    Parameters
    ----------
    cif_path :  string, The full path to the structure CIF file to load.
    dat_path :  string, The full path to the PDF data to be fit.

    Returns
    ----------
    recipe :    The initialized Fit Recipe object using the datname and structure path
                provided.
    """
    # 10: Create a CIF file parsing object, parse and load the structure, and
    # grab the space group name.
    p_cif = getParser('cif')
    stru1 = p_cif.parseFile(cif_path)
    sg = p_cif.spacegroup.short_name

    # 11: Create a Profile object for the experimental dataset and
    # tell this profile the range and mesh of points in r-space.
    profile = Profile()
    parser = PDFParser()
    parser.parseFile(dat_path)
    profile.loadParsedData(parser)
    profile.setCalculationRange(xmin=PDF_RMIN, xmax=PDF_RMAX, dx=PDF_RSTEP)

    # 12: Create a PDF Generator object for a periodic structure model.
    generator_crystal1 = PDFGenerator("G1")
    generator_crystal1.setStructure(stru1, periodic=True)

    # 13: Create a Fit Contribution object.
    contribution = FitContribution("crystal")
    contribution.addProfileGenerator(generator_crystal1)

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
            generator_crystal1.parallel(ncpu=ncpu, mapfunc=pool.map)
        except ImportError:
            print("\nYou don't appear to have the necessary packages for parallelization")

    # 14: Set the Fit Contribution profile to the Profile object.
    contribution.setProfile(profile, xname="r")

    # 15: Set an equation, based on your PDF generators. Here we add an extra layer
    # of complexity, incorporating 'f' into our equation. This new term
    # incorporates the effect of finite crystallite size damping on our PDF model.
    # In this case we use a function which models a spherical NP 'sphericalCF'.
    contribution.registerFunction(sphericalCF, name="f")
    contribution.setEquation("s1*G1*f")

    # 16: Create the Fit Recipe object that holds all the details of the fit.
    recipe = FitRecipe()
    recipe.addContribution(contribution)

    # 17: Initialize the instrument parameters, Q_damp and Q_broad, and
    # assign Q_max and Q_min.
    generator_crystal1.qdamp.value = QDAMP_I
    generator_crystal1.qbroad.value = QBROAD_I
    generator_crystal1.setQmax(QMAX)
    generator_crystal1.setQmin(QMIN)

    # 18: Add, initialize, and tag variables in the Fit Recipe object.
    # In this case we also add 'psize', which is the NP size.
    recipe.addVar(contribution.s1, SCALE_I, tag="scale")
    recipe.addVar(contribution.psize, PSIZE_I, tag="psize")

    # 19: Use the srfit function 'constrainAsSpaceGroup' to constrain
    # the lattice and ADP parameters according to the Fm-3m space group.
    spacegroupparams = constrainAsSpaceGroup(generator_crystal1.phase,
                                             sg)
    for par in spacegroupparams.latpars:
        recipe.addVar(par, value=CUBICLAT_I, fixed=False,
                      name="fcc_Lat", tag="lat")
    for par in spacegroupparams.adppars:
        recipe.addVar(par, value=UISO_I, fixed=False,
                      name="fcc_Uiso", tag="adp")

    # 20: Add delta, but not instrumental parameters to Fit Recipe.
    # The instrumental parameters will remain fixed at values obtained from
    # the Ni calibrant in our previous example. As we have not added them through
    # recipe.addVar, they cannot be refined.
    recipe.addVar(generator_crystal1.delta2,
                  name="Pt_Delta2", value=DELTA2_I, tag="d2")

    # 21: Return the Fit Recipe object to be optimized.
    return recipe

    # End of function


# 22 We create a useful function 'plot_results' for writing a plot of the fit to disk.
def plot_results(recipe, fig_name):
    """
    Creates plots of the fitted PDF and residual, and writes them to disk
    as *.pdf files.

    Parameters
    ----------
    recipe :    The optimized Fit Recipe object containing the PDF data
                we wish to plot.
    fig_name :  Path object, the full path to the figure file to create.

    Returns
    ----------
    None
    """
    if not isinstance(fig_name, Path):
        fig_name = Path(fig_name)

    plt.clf()
    plt.close('all')
    # Get an array of the r-values we fitted over.
    r = recipe.crystal.profile.x

    # Get an array of the observed PDF.
    g = recipe.crystal.profile.y

    # Get an array of the calculated PDF.
    gcalc = recipe.crystal.profile.ycalc

    # Make an array of identical shape as g which is offset from g.
    diffzero = -0.65 * max(g) * np.ones_like(g)

    # Calculate the residual (difference) array and offset it vertically.
    diff = g - gcalc + diffzero

    # Change some style detials of the plot
    mpl.rcParams.update(mpl.rcParamsDefault)
    if (PWD.parent.parent.parent / "utils" / "billinge.mplstyle").exists():
        plt.style.use(str(PWD.parent.parent.parent /
                          "utils" / "billinge.mplstyle"))

    # Create a figure and an axis on which to plot
    fig, ax1 = plt.subplots(1, 1)

    # Plot the difference offset line
    ax1.plot(r, diffzero, lw=1.0, ls="--", c="black")

    # Plot the measured data
    ax1.plot(r,
             g,
             ls="None",
             marker="o",
             ms=5,
             mew=0.2,
             mfc="None",
             label="G(r) Data")

    # Plot the calculated data
    ax1.plot(r, gcalc, lw=1.3, label="G(r) Fit")

    # Plot the difference
    ax1.plot(r, diff, lw=1.2, label="G(r) diff")

    # Let's label the axes!
    ax1.set_xlabel(r"r ($\mathrm{\AA}$)")
    ax1.set_ylabel(r"G ($\mathrm{\AA}$$^{-2}$)")

    # Tune the tick markers. We are picky!
    ax1.tick_params(axis="both",
                    which="major",
                    top=True,
                    right=True)

    # Set the boundaries on the x-axis
    ax1.set_xlim(r[0], r[-1])

    # We definitely want a legend!
    ax1.legend()

    # Let's use a tight layout. Shun wasted space!
    plt.tight_layout()

    # This is going to make a figure pop up on screen for you to view.
    # The script will pause until you close the figure!
    plt.show()

    # Let's save the figure!
    fig.savefig(fig_name.parent / f"{fig_name.name}.pdf", format="pdf")

    # End of function


# 23: We again create a 'main' function to be run when we execute the script.
def main():
    """
    This will run by default when the file is executed using
    'python file.py' in the command line.

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """

    # Make some folders to store our output files.
    resdir = PWD / "res"
    fitdir = PWD / "fit"
    figdir = PWD / "fig"

    folders = [resdir, fitdir, figdir]

    for folder in folders:
        if not folder.exists():
            folder.mkdir()

    # Establish the location of the data and a name for our fit.
    gr_path = DPATH / GR_NAME
    basename = FIT_ID
    print(basename)

    # Establish the full path of the CIF file with the structure of interest.
    stru_path = DPATH / CIF_NAME

    # 24: Call 'make_recipe' to create our fit recipe.
    recipe = make_recipe(str(stru_path),
                         str(gr_path))

    # Tell the Fit Recipe we want to write the maximum amount of
    # information to the terminal during fitting.
    recipe.fithooks[0].verbose = 3

    # 25: As before, we fix all parameters, create a list of tags and,
    # loop over them refining sequentially. In this example, we've added
    # 'psize' because we want to refine the nanoparticle size.
    recipe.fix("all")

    tags = ["lat", "scale", "psize", "adp", "d2", "all"]
    for tag in tags:
        recipe.free(tag)
        least_squares(recipe.residual, recipe.values, x_scale="jac")

    # 26 Write the fitted data to a file.
    profile = recipe.crystal.profile
    profile.savetxt(fitdir / f"{basename}.fit")

    # 27 Print the fit results to the terminal.
    res = FitResults(recipe)
    res.printResults()

    # 28 Write the fit results to a file.
    header = "crystal_HF.\n"
    res.saveResults(resdir / f"{basename}.res", header=header)

    # 29 Write a plot of the fit to a (pdf) file.
    plot_results(recipe, figdir / basename)

    # End of function


# This tells python to run the 'main' function we defined above.
if __name__ == "__main__":
    main()

# End of file
