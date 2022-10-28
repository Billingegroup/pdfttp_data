"""
Please see notes in Chapter 8 of the 'PDF to the People' book for additonal
explanation of the code.

This Diffpy-CMI script will carry out a structural refinement of a measured
PDF from SrFe2As2.  It is the same refinement as is done using PDFgui in this
chapter of the book, only this time using Diffpy-CMI.
"""
# 1: Import relevant system packages that we will need...
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

# ... and the relevant CMI packages
from diffpy.srfit.fitbase import FitContribution, FitRecipe
from diffpy.srfit.fitbase import FitResults
from diffpy.srfit.fitbase import Profile
from diffpy.srfit.structure import constrainAsSpaceGroup
from diffpy.srfit.pdf import PDFParser, PDFGenerator
from diffpy.structure.parsers import getParser
try:
    from bg_mpl_stylesheets.bg_mpl_stylesheet import bg_mpl_style
    plt.style.use(bg_mpl_style)
except ImportError:
    pass

############### Config ##############################
# 2: Give a file path to where your pdf (.gr) and (.cif) files are located.
PWD = Path(__file__).parent.absolute()
DPATH = PWD.parent.parent / "data"

# 3: Give an identifying name for the refinement.
FIT_ID_BASE = "Fit_SrFe2As2_"

# 4: Specify the names of the input PDF and cif files.
GR_NAME_BASE = "SrFe2As2_"
CIF_NAME_BASE = GR_NAME_BASE


######## Experimental PDF Config ######################
# 6: Specify the min, max, and step r-values of the PDF (that we want to fit over)
# also, specify the Q_max and Q_min values used to reduce the PDF.
# Note here, we do our fitting over a coarser r-step to speed things up.
PDF_RMIN = 1.5
PDF_RMAX = 50
PDF_RSTEP = 0.07
QMAX = 25
QMIN = 0.1

########PDF initialize refinable variables #############
# 7: In this case, initial values for the tetragonal lattice
# parameters and ADPs will be taken directly from the .cif
# structure, so we don't need to specify them here.
SCALE_I = 0.4
DELTA2_I = 1.6

# 8: Instrumental will be fixed based on values obtained from a
# separate calibration step. These are hard-coded here.
QDAMP_I = 0.0349
QBROAD_I = 0.0176

# If we want to run using multiprocessors, we can switch this to 'True'.
# This requires that the 'psutil' python package installed.
RUN_PARALLEL = False

# These are options to make the 'least_squares' function a bit
# less picky about when we have reached a converged fit
OPTI_OPTS = {'ftol': 1e-3, 'gtol': 1e-5, 'xtol': 1e-4}

# This flag turns off showing the plot between each temperature step.
# If we turn this on, the script will wait for you to close the plot
# between each temeprature step.
SHOW_PLOT = False


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
    # 9: Create a CIF file parsing object, and use it to parse out
    # relevant info and load the structure in the CIF file. This
    # includes the space group of the structure. We need this so we
    # can constrain the structure parameters later on.
    p_cif = getParser('cif')
    stru1 = p_cif.parseFile(cif_path)
    sg = p_cif.spacegroup.short_name

    # 10: Create a Profile object for the experimental dataset and
    # tell this profile the range and mesh of points in r-space.
    profile = Profile()
    parser = PDFParser()
    parser.parseFile(dat_path)
    profile.loadParsedData(parser)
    profile.setCalculationRange(xmin=PDF_RMIN, xmax=PDF_RMAX, dx=PDF_RSTEP)

    # 11: Create a PDF Generator object for a periodic structure model.
    generator_crystal1 = PDFGenerator("G1")
    generator_crystal1.setStructure(stru1, periodic=True)

    # 12: Create a Fit Contribution object.
    contribution = FitContribution("crystal")
    contribution.addProfileGenerator(generator_crystal1)

    # If you have a multi-core computer (you probably do), run your refinement in parallel!
    if RUN_PARALLEL:
        try:
            import psutil
            import multiprocessing
            from multiprocessing import Pool
        except ImportError:
            print("\nYou don't appear to have the necessary packages for parallelization")
        syst_cores = multiprocessing.cpu_count()
        cpu_percent = psutil.cpu_percent()
        avail_cores = np.floor((100 - cpu_percent) / (100.0 / syst_cores))
        ncpu = int(np.max([1, avail_cores]))
        pool = Pool(processes=ncpu)
        generator_crystal1.parallel(ncpu=ncpu, mapfunc=pool.map)

    # 13: Set the Fit Contribution profile to the Profile object.
    contribution.setProfile(profile, xname="r")

    # 14: Set an equation, based on your PDF generators. This is
    # again a simple case, with only a scale and a single PDF generator.
    contribution.setEquation("s1*G1")

    # 15: Create the Fit Recipe object that holds all the details of the fit.
    recipe = FitRecipe()
    recipe.addContribution(contribution)

    # 16: Initialize the instrument parameters, Q_damp and Q_broad, and
    # assign Q_max and Q_min.
    generator_crystal1.qdamp.value = QDAMP_I
    generator_crystal1.qbroad.value = QBROAD_I
    generator_crystal1.setQmax(QMAX)
    generator_crystal1.setQmin(QMIN)

    # 17: Add, initialize, and tag the scale variable.
    recipe.addVar(contribution.s1, SCALE_I, tag="scale")

    # 18: Use the srfit function constrainAsSpaceGroup to constrain
    # the lattice, ADP parameters, and atomic positions according to the space group.
    spacegroupparams = constrainAsSpaceGroup(generator_crystal1.phase,
                                             sg)
    for par in spacegroupparams.latpars:
        recipe.addVar(par, fixed=False, tag="lat")
    for par in spacegroupparams.adppars:
        recipe.addVar(par, fixed=False, tag="adp")
    for par in spacegroupparams.xyzpars:
        recipe.addVar(par, fixed=False, tag="xyz")

    # 19: Add delta, but not instrumental parameters to Fit Recipe.
    recipe.addVar(generator_crystal1.delta2,
                  name="Delta2", value=DELTA2_I, tag="d2")

    return recipe

    # End of function


# 20: We create a useful function 'plot_results' for writing a plot of the fit to disk.
def plot_results(recipe, fig_name):
    """
    Creates plots of the fitted PDF and residual, and writes them to disk
    as *.pdf files.

    Parameters
    ----------
    recipe :    The optimized Fit Recipe object containing the PDF data
                we wish to plot.
    fig_name :  Path object, the full path to the figure file to create..

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
    if SHOW_PLOT:
        plt.show()

    # Let's save the figure!
    fig.savefig(fig_name.parent / f"{fig_name.name}.pdf", format="pdf")

    # End of function


# 21: We again create a 'main' function to be run when we execute the script.
def main():
    """
    This will run by default when the file is executed using
    'python file.py' in the command line

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

    # 22: Search our data path for all the .gr files matching a pattern
    data_files = list(DPATH.glob(f"*{GR_NAME_BASE}*.gr"))

    # 23: Parse out the temperature from each file name.
    temps = [t.stem for t in data_files]
    temps = [t.split('_')[1].strip("K") for t in temps]
    temps = [int(t) for t in temps]

    # 24: We should sort our data files and temperatures in some meaninful way!
    temps, data_files = zip(*sorted(zip(temps, data_files), reverse=True))

    # 25: Search our data path for all the .cif files matching a pattern
    cif_files = list(DPATH.glob(f"*{CIF_NAME_BASE}*.cif"))

    # 26: Since we want to test two different structures, we will loop on each
    # .cif file we've found
    for cif in cif_files:
        # 27: Here we load in the cif file, and get a string that represents the structure.
        # Specifically, the short space group name. We need to replace any "/" with something else,
        # as this will cause issues with naming.
        p_cif = getParser('cif')
        p_cif.parseFile(str(cif))
        stru_type = p_cif.spacegroup.short_name.replace("/", "_on_")

        # 28: We make our recipe with the first temperature/PDF file on the list.
        recipe = make_recipe(str(cif), data_files[0])

        # Tell the Fit Recipe we want to write the maximum amount of
        # information to the terminal during fitting.
        recipe.fithooks[0].verbose = 3

        # 29: Inside our loop on the different structure files,
        # we will also loop over all the temperatures and PDF files.
        for file, temp in zip(data_files, temps):
            # 30: We are going to programmatically create a name for our fit,
            # Which will include the temperature as well as the structure type.
            basename = f"{FIT_ID_BASE}{stru_type}_{temp}K"
            print(basename)

            # 31: For each temperature, we will need to create a new profile,
            # set this profile in our recipe, and then tell this profile which range to
            # fit over.
            profile = Profile()
            parser = PDFParser()
            parser.parseFile(file)
            profile.loadParsedData(parser)
            recipe.crystal.setProfile(profile)
            recipe.crystal.profile.setCalculationRange(
                xmin=PDF_RMIN, xmax=PDF_RMAX, dx=PDF_RSTEP)

            # 32: As before, we fix all parameters, create a list of tags and,
            # loop over them refining sequentially.
            recipe.fix("all")
            tags = ["lat", "scale", "adp", "xyz", "d2", "all"]
            for tag in tags:
                recipe.free(tag)
                least_squares(recipe.residual, recipe.values,
                              x_scale="jac", **OPTI_OPTS)

            # 33: Write the fitted data to a file.
            profile = recipe.crystal.profile
            profile.savetxt(fitdir / f"{basename}.fit")

            # 34: Print the fit results to the terminal.
            res = FitResults(recipe)
            res.printResults()

            # 35: Write the fit results to a file.
            header = "crystal_HF.\n"
            res.saveResults(resdir / f"{basename}.res", header=header)

            # 36: Write a plot of the fit to a (pdf) file.
            plot_results(recipe, figdir / basename)

    # End of function


# This tells python to run the 'main' function we defined above.
if __name__ == "__main__":
    main()

# End of file
