"""
Please see notes in Chapter 6 of the 'PDF to the People' book for additonal
explanation of the code.

This Diffpy-CMI script will carry out a structural refinement of a measured
PDF from a sample containing two phases, silicon and nickel.  It is the
same refinement as is done using PDFgui in this chapter of the book, only
this time using Diffpy-CMI.
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
from diffpy.srfit.pdf import PDFParser, PDFGenerator
from diffpy.structure.parsers import getParser
from diffpy.srfit.structure import constrainAsSpaceGroup

############### Config ##############################
# 2: Give a file path to where your pdf (.gr) and (.cif) files are located.
PWD = Path(__file__).parent.absolute()
DPATH = PWD.parent.parent / "data"

# 3: Give an identifying name for the refinement.
FIT_ID = 'Fit_SiNi_2P'

# 4: Specify the names of the input PDF and two CIF files (one for each phase).
GR_NAME = 'sini.gr'
CIF_NAME1 = 'Si_Fd-3m.cif'
CIF_NAME2 = 'Ni_Fm-3m.cif'

######## Experimental PDF Config ######################
# 5: Specify the min, max, and step r-values of the PDF (that we want to fit over)
# also, specify the Q_max and Q_min values used to reduce the PDF.
PDF_RMIN = 1.5
PDF_RMAX = 30
PDF_RSTEP = 0.01
QMAX = 25
QMIN = 0.1

########PDF initialize refinable variables #############
# 6: In this case, initial values for the lattice parameters
# scale factors, and correlated motion parameters, while the
# ADPs for both phases will be taken directly from
# the .cif structures.
SCALE_I_SI = 0.9
DELTA1_I_SI = 1.5
DELTA1_I_NI = 1.5
DATA_SCALE_I = 0.1

# 7: Instrumental will be fixed based on values obtained from a
# separate calibration step. These are hard-coded here.
QDAMP_I = 0.05
QBROAD_I = 0.0

# If we want to run using multiprocessors, we can switch this to 'True'.
# This requires that the 'psutil' python package installed.
RUN_PARALLEL = False


######## Functions that will carry out the refinement ##################
# 8: We define a function 'make_recipe' to make the recipe that the fit will follow.
def make_recipe(cif_path1, cif_path2, dat_path):
    """
    Creates and returns a Fit Recipe object with two phases.

    Parameters
    ----------
    cif_path1 : string, The full path to the structure CIF file to load, for the first phase.
    cif_path2 : string, The full path to the structure CIF file to load, for the second phase.
    dat_path :  string, The full path to the PDF data to be fit.

    Returns
    ----------
    recipe :    The initialized Fit Recipe object using the datname and structure path
                provided.
    """
    # 9: Create two CIF file parsing objects, parse and load the structures, and
    # grab the space group names.
    p_cif1 = getParser('cif')
    p_cif2 = getParser('cif')
    stru1 = p_cif1.parseFile(cif_path1)
    stru2 = p_cif2.parseFile(cif_path2)
    sg1 = p_cif1.spacegroup.short_name
    sg2 = p_cif2.spacegroup.short_name

    # 10: Create a Profile object for the experimental dataset and
    # tell this profile the range and mesh of points in r-space.
    profile = Profile()
    parser = PDFParser()
    parser.parseFile(dat_path)
    profile.loadParsedData(parser)
    profile.setCalculationRange(xmin=PDF_RMIN, xmax=PDF_RMAX, dx=PDF_RSTEP)

    # 11a: Create a PDF Generator object for a periodic structure model
    # of phase 1.
    generator_crystal1 = PDFGenerator("G_Si")
    generator_crystal1.setStructure(stru1, periodic=True)

    # 11b: Create a PDF Generator object for a periodic structure model
    # of phase 2.
    generator_crystal2 = PDFGenerator("G_Ni")
    generator_crystal2.setStructure(stru2, periodic=True)

    # 12: Create a Fit Contribution object. This is new, as we
    # need to tell the Fit Contribution about BOTH the phase
    # represented by 'generator_crystal1' AND the phase represented
    # by 'generator_crystal2'.
    contribution = FitContribution("crystal")
    contribution.addProfileGenerator(generator_crystal1)
    contribution.addProfileGenerator(generator_crystal2)

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
        generator_crystal2.parallel(ncpu=ncpu, mapfunc=pool.map)

    # 13: Set the Fit Contribution profile to the Profile object.
    contribution.setProfile(profile, xname="r")

    # 14: Set an equation, based on your PDF generators. This is
    # a more complicated case, since we have two phases. The equation
    # here will be the sum of the contributions from each phase,
    # 'G_Si' and 'G_Ni' weighted by a refined scale term for each phase,
    # 's1_Si' and '(1 - s1_Si)'. We also include a general 's2'
    # to account for data scale.
    contribution.setEquation("s2*(s1_Si*G_Si + (1.0-s1_Si)*G_Ni)")

    # 15: Create the Fit Recipe object that holds all the details of the fit.
    recipe = FitRecipe()
    recipe.addContribution(contribution)

    # 16: Add, initialize, and tag the two scale variables.
    recipe.addVar(contribution.s1_Si, SCALE_I_SI, tag="scale")
    recipe.addVar(contribution.s2, DATA_SCALE_I, tag="scale")

    # 17:This is new, we want to ensure that the data scale parameter 's2'
    # is always positive, and the phase scale parameter 's1_Si' is always
    # bounded between zero and one, to avoid any negative PDF signals.
    # We do this by adding 'restraints'. Effectively a restrain will modify
    # our objective function such that if the parameter approaches a user
    # defined upper or lower bound, the objective function will increase,
    # driving the fit away from the boundary.
    recipe.restrain("s2",
                    lb=0.0,
                    scaled=True,
                    sig=0.00001)

    recipe.restrain("s1_Si",
                    lb=0.0,
                    ub=1.0,
                    scaled=True,
                    sig=0.00001)

    # 18a: This is a bit new. We will again use the srfit function
    # constrainAsSpaceGroup to constrain the lattice and ADP parameters
    # according to the space group of each of the two phases.
    # We loop through generators composed of PDF Generators
    # and space groups specific to EACH of the TWO candidate phases.
    # We use 'enumerate' to create an iterating index 'i' such that each
    # parameter can get it's own unique name, without colliding parameters.
    for name, generator, space_group in zip(["Si", "Ni"],
                                            [generator_crystal1,
                                                generator_crystal2],
                                            [sg1, sg2]):

        # 18b: Initialize the instrument parameters, Q_damp and Q_broad, and
        # assign Q_max and Q_min for each phase.
        generator.qdamp.value = QDAMP_I
        generator.qbroad.value = QBROAD_I
        generator.setQmax(QMAX)
        generator.setQmin(QMIN)

        # 18c: Get the symmetry equivalent parameters for each phase.
        spacegroupparams = constrainAsSpaceGroup(generator.phase,
                                                 space_group)
        # 18d: Loop over and constrain these parameters for each phase.
        # Each parameter name gets the loop index 'i' appeneded so there are not
        # parameter name collisions.
        for par in spacegroupparams.latpars:
            recipe.addVar(par,
                          name=f"{par.name}_{name}",
                          fixed=False,
                          tag='lat')
        for par in spacegroupparams.adppars:
            recipe.addVar(par,
                          name=f"{par.name}_{name}",
                          fixed=False,
                          tag='adp')

        # 19: Add delta, but not instrumental parameters to Fit Recipe.
        # One for each phase.
        recipe.addVar(generator.delta1, name=f"Delta1_{name}",
                      value=DELTA1_I_SI, tag="d1")

    # 20: Return the Fit Recipe object to be optimized.
    return recipe

    # End of function


# 21 We create a useful function 'plot_results' for writing a plot of the fit to disk.
def plot_results(recipe, fig_name):
    """
    Creates plots of the fitted PDF and residual, and writes them to disk
    as *.pdf files.

    Parameters
    ----------
    recipe :    The optimized Fit Recipe object containing the PDF data
                we wish to plot.
    fig_name :  bPath object, the full path to the figure file to create..

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
    diff = g - gcalc
    diffzero = (min(g)-np.abs(max(diff))) * \
        np.ones_like(g)

    # Calculate the residual (difference) array and offset it vertically.
    diff = g - gcalc + diffzero

    # Get the Ni and Si scale terms
    ni_scale = recipe.s2.value*(1.0-recipe.s1_Si.value)
    si_scale = recipe.s2.value*recipe.s1_Si.value

    # Get the Si model signal contribution
    si_signal = si_scale*recipe.crystal.G_Si.profile.ycalc
    si_signal += min(diff)-np.abs(max(si_signal))

    # Get the Ni model signal contribution
    ni_signal = ni_scale*recipe.crystal.G_Ni.profile.ycalc
    ni_signal += min(si_signal)-np.abs(max(ni_signal))

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

    # Plot the calculated Si signal
    ax1.plot(r, si_signal, lw=1.3, label="Silicon", color="C1")

    # Plot the calculated Ni signal
    ax1.plot(r, ni_signal, lw=1.3, label="Nickel", color="C0")

    # Plot the total calculated data
    ax1.plot(r, gcalc, lw=1.3, label="G(r) Fit")

    # Plot the difference
    ax1.plot(r, diff, lw=1.2, label="G(r) diff")

    # Let's label the axes!
    ax1.set_xlabel(r"r ($\mathrm{\AA}$)")
    ax1.set_ylabel(r"G ($\mathrm{\AA}$$^{-2}$)")
    ax1.tick_params(axis="both",
                    which="major",
                    top=True,
                    right=True)

    # Set the boundaries on the x-axis
    ax1.set_xlim(r[0], r[-1])

    # We definitely want a legend!
    ax1.legend(ncol=2)

    # Let's use a tight layout. Shun wasted space!
    plt.tight_layout()

    # This is going to make a figure pop up on screen for you to view.
    # The script will pause until you close the figure!
    plt.show()

    # Let's save the figure!
    fig.savefig(fig_name.parent / f"{fig_name.name}.pdf", format="pdf")

    # End of function


# 22: We again create a 'main' function to be run when we execute the script.
def main():
    """
    This will run by default when the file is executed using
    "python file.py" in the command line

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

    # Establish the full path of the two CIF files with the structure of interest.
    stru1_path = DPATH / CIF_NAME1
    stru2_path = DPATH / CIF_NAME2

    # 23: Call 'make_recipe' to create our fit recipe, this time providing the name
    # and location of BOTH structure files
    recipe = make_recipe(str(stru1_path),
                         str(stru2_path),
                         str(gr_path))

    # Tell the Fit Recipe we want to write the maximum amount of
    # information to the terminal during fitting.
    recipe.fithooks[0].verbose = 3

    # 24: As before, we fix all parameters, create a list of tags and,
    # loop over them refining sequentially.
    recipe.fix("all")

    tags = ["lat", "scale", "adp", "d1", "all"]
    for tag in tags:
        recipe.free(tag)
        least_squares(recipe.residual, recipe.values, x_scale="jac")

    # 25 Write the fitted data to a file.
    profile = recipe.crystal.profile
    profile.savetxt(fitdir / f"{basename}.fit")

    # 24 Print the fit results to the terminal.
    res = FitResults(recipe)
    res.printResults()

    # 25 Write the fit results to a file.
    header = "crystal_HF.\n"
    res.saveResults(resdir / f"{basename}.res", header=header)

    # 26 Write a plot of the fit to a (pdf) file.
    plot_results(recipe, figdir / basename)

    # End of function


# This tells python to run the 'main' function we defined above.
if __name__ == "__main__":
    main()

# End of file
