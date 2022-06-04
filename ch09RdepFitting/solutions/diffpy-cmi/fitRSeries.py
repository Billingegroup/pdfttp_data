"""
Please see notes in Chapter 10 of the "PDF to the People" book for additonal
explanation of the code.

This Diffpy-CMI script will carry out a structural refinement of a measured
PDF from CuIr1.76Cr0.24S4.  It is the same refinement as is done using PDFgui in this
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
from diffpy.srfit.pdf import PDFParser, PDFGenerator
from diffpy.structure.parsers import getParser
from diffpy.structure.atom import Atom
from diffpy.srfit.structure import constrainAsSpaceGroup

############### Config ##############################
# 2: Give a file path to where your pdf (.gr) and (.cif) files are located.
PWD = Path(__file__).parent.absolute()
DPATH = PWD.parent.parent / "data"

# 3: Give an identifying name for the refinement.
FIT_ID_BASE = "Fit_CuIr1.76Cr0.24S4_"

# 4: Specify the names of the input PDF and cif files.
GR_NAME = "CuIr1.76Cr0.24S4-q27r70t100-11IDC-APS.gr"
CIF_NAME = "Fd-3m.cif"


######## Experimental PDF Config ######################
# 5: Specify the min, max, and step r-values of the PDF (that we want to fit over)
# also, specify the Q_max and Q_min values used to reduce the PDF.
FULL_PDF_RMIN = 1.5
FULL_PDF_RMAX = 50.5
PDF_RSTEP = 0.01
QMAX = 27
QMIN = 0.1

# 6: We define the size of the r-range window and the number of
# fits to run for our sliding boxcar fit
SLIDING_R_WINDOW = 4.0
SLIDING_R_NUMBER = 46

# 7: We define the start r values for our sliding boxcar fits
WINDOW_R_STARTS = np.linspace(
    FULL_PDF_RMIN, FULL_PDF_RMAX - SLIDING_R_WINDOW, SLIDING_R_NUMBER)

# 8: We define the end r values for our sliding boxcar fits
WINDOW_R_STOPS = WINDOW_R_STARTS + SLIDING_R_WINDOW

# 9: We build a list of tuples, one tuple for each r_start and r_stop pair
# for our sliding box-car fits
SLIDING_WINDOW_RANGES = [(r_start, r_stop)
                         for r_start, r_stop in zip(WINDOW_R_STARTS, WINDOW_R_STOPS)]

# 10: We build a list of tuples, one tuple for each r_start and r_stop pair
# for our sliding r_max fits.
SLIDING_R_MAX_RANGES = [(FULL_PDF_RMIN, r_stop)
                        for _, r_stop in SLIDING_WINDOW_RANGES]

# 11: We want to handle 3 cases: the full fit range, sliding box-car
# and sliding r_max. Here we build a list, where each entry represents,
# one of these cases, and is itself a list of (r_max, r_min) tuples.
FIT_RANGE_LIST = [[(FULL_PDF_RMIN, FULL_PDF_RMAX)],
                  SLIDING_WINDOW_RANGES, SLIDING_R_MAX_RANGES]

# 12: We want to have a recognizable name for the 3 cases we want to handle.
# We assign a string name for each here.
FIT_NAME_LISTS = ["full", "sliding_box_car", "sliding_r_max"]

########PDF initialize refinable variables #############
# 13: In this case, initial values for the lattice
# parameters are taken directly from the .cif. We provide
# initial values for parameters here.
SCALE_I = 0.88
DELTA1_I = 2.0
UISO_I = 0.005
IR_OCC = 0.88

# 14: Instrumental will be fixed based on values obtained from a
# separate calibration step. These are hard-coded here.
QDAMP_I = 0.038
QBROAD_I = 0.018

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
# 15: We define a function 'make_recipe' to make the recipe that the fit will follow.
def make_recipe(cif_path, dat_path):
    """
    Creates and returns a Fit Recipe object

    Parameters
    ----------
    cif_path :  string, The full path to the structure CIF file to load.
    dat_path :  string, The full path to the PDF data to be fit.

    Returns
    ----------
    fitrecipe : The initialized Fit Recipe object using the datname and structure
                provided
    """
    # 16: Create a CIF file parsing object, and use it to parse out
    # relevant info and load the structure in the CIF file. This
    # includes the space group of the structure. We need this so we
    # can constrain the structure parameters later on.
    p_cif = getParser('cif')
    stru1 = p_cif.parseFile(cif_path)
    sg = p_cif.spacegroup.short_name
    stru1.anisotropy = False

    # 17: Here we have something new. The cif file we loaded is for CuIr2S4,
    # But our sample contains Cr on the Ir site.
    # we need to add some atoms, so we loop over all atoms in the structure,
    # and if the atom element matches "Ir" we add a Cr at the same
    # coordinates.
    for atom in stru1:
        if "Ir" in atom.element:
            stru1.addNewAtom(Atom("Cr", xyz=atom.xyz))

    # 18: The file we loaded had a fully occupied Ir site, but now we've added Cr on
    # this site. We need to alter the occupancy of these two sites according to the 'IR_OCC'
    # parameter we defined above.
    for atom in stru1:
        if 'Ir' in atom.element:
            atom.occupancy = IR_OCC
        if 'Cr' in atom.element:
            atom.occupancy = 1.0 - IR_OCC

    # 19: Create a Profile object for the experimental dataset and
    # tell this profile the range and mesh of points in r-space.
    profile = Profile()
    parser = PDFParser()
    parser.parseFile(dat_path)
    profile.loadParsedData(parser)
    profile.setCalculationRange(
        xmin=FULL_PDF_RMIN, xmax=FULL_PDF_RMAX, dx=PDF_RSTEP)

    # 20: Create a PDF Generator object for a periodic structure model.
    generator_crystal1 = PDFGenerator("G1")
    generator_crystal1.setStructure(stru1, periodic=True)

    # 21: Create a Fit Contribution object.
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

    # 22: Set the Fit Contribution profile to the Profile object.
    contribution.setProfile(profile, xname="r")

    # 23: Set an equation, based on your PDF generators. This is
    # again a simple case, with only a scale and a single PDF generator.
    contribution.setEquation("s1*G1")

    # 24: Create the Fit Recipe object that holds all the details of the fit.
    recipe = FitRecipe()
    recipe.addContribution(contribution)

    # 25: Initialize the instrument parameters, Q_damp and Q_broad, and
    # assign Q_max and Q_min.
    generator_crystal1.qdamp.value = QDAMP_I
    generator_crystal1.qbroad.value = QBROAD_I
    generator_crystal1.setQmax(QMAX)
    generator_crystal1.setQmin(QMIN)

    # 26: Add, initialize, and tag the scale variable.
    recipe.addVar(contribution.s1, SCALE_I, tag="scale")

    # 27: Use the srfit function constrainAsSpaceGroup to constrain
    # the lattice and ADP parameters according to the I4/mmm space group setting.
    spacegroupparams = constrainAsSpaceGroup(generator_crystal1.phase,
                                             sg,
                                             constrainadps=False)

    for par in spacegroupparams.latpars:
        recipe.addVar(par, fixed=False, tag="lat")
    for par in spacegroupparams.xyzpars:
        if par.constrained:
            par.constrained = False
        recipe.addVar(par, fixed=False, tag="xyz")

    # 28: We create the variables of isotropic ADP and assign the initial value to them,
    # specified above.
    ir_uiso = recipe.newVar("Ir_Uiso", value=UISO_I, tag="adp")

    cu_uiso = recipe.newVar("Cu_Uiso", value=UISO_I, tag="adp")

    s_uiso = recipe.newVar("S_Uiso", value=UISO_I, tag="adp")

    # 29: For all atoms in the structure model, we constrain their Uiso according to
    # their species. Cr and Ir get the same U_iso parameter.
    atoms = generator_crystal1.phase.getScatterers()
    for atom in atoms:
        if atom.element in ['Ir', 'Cr']:
            recipe.constrain(atom.Uiso, ir_uiso)
        elif atom.element == 'S':
            recipe.constrain(atom.Uiso, s_uiso)
        elif atom.element == 'Cu':
            recipe.constrain(atom.Uiso, cu_uiso)

    # 30: Set value of delta1, but we do not add it to the fit recipe,
    # as it will stay fixed.
    generator_crystal1.delta1.value = DELTA1_I

    return recipe
    # End of function


# 31: We create a useful function 'plot_results' for writing a plot of the fit to disk.
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


# 32: We again create a 'main' function to be run when we execute the script.
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

    # Establish the full path of the CIF file with the structure of interest.
    stru_path = DPATH / CIF_NAME

    # 33: Call 'make_recipe' to create our fit recipe.
    recipe = make_recipe(str(stru_path),
                         str(gr_path))

    # Tell the Fit Recipe we want to write the maximum amount of
    # information to the terminal during fitting.
    recipe.fithooks[0].verbose = 3

    # 34: We loop over the fit names and the associated list of (r_starts, r_stops)
    # to perform all our different types of fitting.
    for fit_name, fit_ranges in zip(FIT_NAME_LISTS, FIT_RANGE_LIST):
        # 35: We loop over each (r_starts, r_stops) tuple in the list of these
        for r_start, r_stop in fit_ranges:
            # 36: We create a bespoke fit name including the chemistry,
            # and the r_start and r_stop of the fit.
            basename = f"{FIT_ID_BASE}r_{r_start}-{r_stop}"
            print(basename)

            # 37: We change the r-range of our fit.
            recipe.crystal.profile.setCalculationRange(xmin=r_start,
                                                       xmax=r_stop)

            # 38: We want to handle two cases, the full r-range, where we
            # will vary more parameters, and the sliding window/r_rmax,
            # where we will vary a smaller subset of the parameters.
            # we are in the case labelled 'full' we will define one set of parameters.
            if fit_name == "full":
                tags = ["lat", "scale", "xyz", "adp"]

            # 39: If we are not in the 'full' case we will define a smaller, subset of
            # parameters.
            else:
                tags = ["lat", "xyz", "adp"]

            # 40: As before, we fix all parameters and loop over our list of tags refining sequentially.
            recipe.fix("all")
            for tag in tags:
                recipe.free(tag)
                least_squares(recipe.residual, recipe.values,
                              x_scale="jac", **OPTI_OPTS)

            # 41: Write the fitted data to a file.
            profile = recipe.crystal.profile
            profile.savetxt(fitdir / f"{basename}.fit")

            # 42: Print the fit results to the terminal.
            res = FitResults(recipe)
            res.printResults()

            # 43: Write the fit results to a file.
            header = "crystal_HF.\n"
            res.saveResults(resdir / f"{basename}.res", header=header)

            # 44: Write a plot of the fit to a (pdf) file.
            plot_results(recipe, figdir / basename)

    # End of function


if __name__ == "__main__":
    main()

# End of file
