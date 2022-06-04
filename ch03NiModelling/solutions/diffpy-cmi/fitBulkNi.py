"""
Please see notes in Chapter 3 of the 'PDF to the People' book for additonal
explanation of the code.

This Diffpy-CMI script will carry out a structural refinement of a measured
PDF from nickel.  It is the same refinement as is done using PDFgui in this
chapter of the book, only this time using Diffpy-CMI
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
# 2: Give a file path to where your PDF (.gr) and structure (.cif) files are located.
# In this case it is two directories up, in a folder called 'data'.
# First we store the absolute directory of this script, then two directories above this,
# with the directory 'data' appended
PWD = Path(__file__).parent.absolute()
DPATH = PWD.parent.parent / "data"

# 3: Give an identifying name for the refinement, similar
# to what you would name a fit tree in PDFGui.
FIT_ID = "Fit_Ni_Bulk"

# 4: Specify the names of the input PDF and CIF file.
GR_NAME = "Ni.gr"
CIF_NAME = "Ni.cif"

######## Experimental PDF Config ######################
# 5: Specify the min, max, and step r-values of the PDF (that we want to fit over)
# also, specify the Qmax and Qmin values used to reduce the PDF.
PDF_RMIN = 1.5
PDF_RMAX = 50
PDF_RSTEP = 0.01
QMAX = 25
QMIN = 0.1

######## PDF Initialize refinable variables #############
# 6: We explicitly specify initial values lattice parameter, scale, and
# isotropic thermal parameters, as well as a correlated
# motion parameter, in this case delta_2.
CUBICLAT_I = 3.52
SCALE_I = 0.4
UISO_I = 0.005
DELTA2_I = 2

# 7: We will give initial values for the instrumental parameters, but because
# this is a calibrant, we will also refine these variables.
QDAMP_I = 0.04
QBROAD_I = 0.02

# If we want to run using multiprocessors, we can switch this to 'True'.
# This requires that the 'psutil' python package installed.
RUN_PARALLEL = True


######## Functions that will carry out the refinement ##################
# 8: We define a function 'make_recipe' to make the recipe that the fit will follow.
# This Fit Recipe object contains the PDF data, information on all the structure(s)
# we will use to fit, and all relevant details necessary to run the fit.
def make_recipe(cif_path, dat_path):
    """
    Creates and returns a Fit Recipe object

    Parameters
    ----------
    cif_path :  string, The full path to the structure CIF file to load.
    dat_path :  string, The full path to the PDF data to be fit.

    Returns
    ----------
    recipe :    The initialized Fit Recipe object using the dat_path and structure path
                provided.
    """
    # 9: Create a CIF file parsing object, and use it to parse out
    # relevant info and load the structure in the CIF file. This
    # includes the space group of the structure. We need this so we
    # can constrain the structure parameters later on.
    p_cif = getParser('cif')
    stru1 = p_cif.parseFile(cif_path)
    sg = p_cif.spacegroup.short_name

    # 10: Create a Profile object for the experimental dataset.
    # This handles all details about the dataset.
    # We also tell this profile the range and mesh of points in r-space.
    # The 'PDFParser' function should parse out the appropriate Q_min and
    # Q_max from the *.gr file, if the information is present.
    profile = Profile()
    parser = PDFParser()
    parser.parseFile(dat_path)
    profile.loadParsedData(parser)
    profile.setCalculationRange(xmin=PDF_RMIN, xmax=PDF_RMAX, dx=PDF_RSTEP)

    # 11: Create a PDF Generator object for a periodic structure model.
    # Here we name it arbitrarily 'G1' and we give it the structure object.
    # This Generator will later compute the model PDF for the structure
    # object we provide it here.
    generator_crystal1 = PDFGenerator("G1")
    generator_crystal1.setStructure(stru1, periodic=True)

    # 12: Create a Fit Contribution object, and arbitrarily name it 'crystal'.
    # We then give the PDF Generator object we created just above
    # to this Fit Contribution object. The Fit Contribution holds
    # the equation used to fit the PDF.
    contribution = FitContribution("crystal")
    contribution.addProfileGenerator(generator_crystal1)

    # If you have a multi-core computer (you probably do),
    # run your refinement in parallel!
    # Here we just make sure not to overload your CPUs.
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

    # 13: Set the experimental profile, within the Fit Contribution object,
    # to the Profile object we created earlier.
    contribution.setProfile(profile, xname="r")

    # 14: Set an equation, within the Fit Contribution, based on your PDF
    # Generators. Here we simply have one Generator, 'G1', and a scale variable,
    # 's1'. Using this structure is a very flexible way of adding additional
    # Generators (ie. multiple structural phases), experimental Profiles,
    # PDF characteristic functions (ie. shape envelopes), and more.
    contribution.setEquation("s1*G1")

    # 15: Create the Fit Recipe object that holds all the details of the fit,
    # defined in previous lines above. We give the Fit Recipe the Fit
    # Contribution we created earlier.
    recipe = FitRecipe()
    recipe.addContribution(contribution)

    # 16: Initialize the instrument parameters, Q_damp and Q_broad, and
    # assign Q_max and Q_min, all part of the PDF Generator object.
    # It's possible that the 'PDFParse' function we used above
    # already parsed out ths information, but in case it didn't, we set it
    # explicitly again here.
    # All parameter objects can have their value assigned using the
    # below '.value = ' syntax.
    recipe.crystal.G1.qdamp.value = QDAMP_I
    recipe.crystal.G1.qbroad.value = QBROAD_I
    recipe.crystal.G1.setQmax(QMAX)
    recipe.crystal.G1.setQmin(QMIN)

    # 17: Add a variable to the Fit Recipe object, initialize the variables
    # with some value, and tag it with an aribtrary string. Here we add the scale
    # parameter from the Fit Contribution. The '.addVar' method can be
    # used to add variables to the Fit Recipe.
    recipe.addVar(contribution.s1, SCALE_I, tag="scale")

    # 18: Configure some additional fit variables pertaining to symmetry.
    # We can use the srfit function 'constrainAsSpaceGroup' to constrain
    # the lattice and ADP parameters according to the Fm-3m space group.
    # First we establish the relevant parameters, then we loop through
    # the parameters and activate and tag them. We must explicitly set the
    # ADP parameters using 'value=' because CIF had no ADP data.
    spacegroupparams = constrainAsSpaceGroup(generator_crystal1.phase,
                                             sg)
    for par in spacegroupparams.latpars:
        recipe.addVar(par,
                      value=CUBICLAT_I,
                      fixed=False,
                      name="fcc_Lat",
                      tag="lat")
    for par in spacegroupparams.adppars:
        recipe.addVar(par,
                      value=UISO_I,
                      fixed=False,
                      name="fcc_ADP",
                      tag="adp")

    # 19: Add delta and instrumental parameters to Fit Recipe.
    # These parameters are contained as part of the PDF Generator object
    # and initialized with values as defined in the opening of the script.
    # We give them unique names, and tag them with relevant strings.
    recipe.addVar(generator_crystal1.delta2,
                  name="Ni_Delta2",
                  value=DELTA2_I,
                  tag="d2")

    recipe.addVar(generator_crystal1.qdamp,
                  fixed=False,
                  name="Calib_Qdamp",
                  value=QDAMP_I,
                  tag="inst")

    recipe.addVar(generator_crystal1.qbroad,
                  fixed=False,
                  name="Calib_Qbroad",
                  value=QBROAD_I,
                  tag="inst")

    # 20: Return the Fit Recipe object to be optimized.
    return recipe

    # End of function


# 21: We create a useful function 'plot_results' for writing a plot of the fit to disk.
# We won't go into detail here as much of this is non-CMI specific
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
    plt.show()

    # Let's save the figure!
    fig.savefig(fig_name.parent / f"{fig_name.name}.pdf", format="pdf")

    # End of function


# 22: By Convention, this main function is where we do most of our work, and it
# is the bit of code which will be run when we issue 'python file.py' from a terminal.
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

    # 23: Now we call our 'make_recipe' function created above, giving
    # strings which points to the relevant CIF file and PDF data file.
    recipe = make_recipe(str(stru_path),
                         str(gr_path))

    # Tell the Fit Recipe we want to write the maximum amount of
    # information to the terminal during fitting.
    # Passing '2' or '1' prints intermediate info, while '0' prints no info.
    recipe.fithooks[0].verbose = 3

    # 24: During the optimization, fix and free parameters sequentially
    # as you would PDFgui. This leads to more stability in the refinement.
    # This can be done with 'recipe.fix' and 'recipe.free' and we can use
    # either a single parameter name or any of the tags we assigned when creating
    # the fit recipe. We first fix all variables. The tag 'all' incorporates every parameter.
    # We then create a list of 'tags' which we want free sequentially, we
    # loop over them freeing each during a loop, and then fit using the
    # SciPy function 'least_squares'. 'least_squares' takes as its arguments
    # the function to be optimized, here 'recipe.residual',
    # as well as initial values for the fitted parameters, provided by
    # 'recipe.values'. The 'x_scale="jac"' argument is optional
    # and provides for a bit more stability.
    recipe.fix("all")
    tags = ["lat", "scale", "adp", "d2", "all"]
    for tag in tags:
        recipe.free(tag)
        least_squares(recipe.residual, recipe.values, x_scale="jac")

    # 25: We use the 'savetxt' method of the profile to write a text file
    # containing the measured and fitted PDF to disk.
    # The file is named based on the basename we created earlier, and
    # written to the 'fitdir' directory.
    profile = recipe.crystal.profile
    profile.savetxt(fitdir / f"{basename}.fit")

    # 26: We use the 'FitResults' method to parse out the results from
    # the optimized Fit Recipe, and 'printResults' to print them
    # to the terminal.
    res = FitResults(recipe)
    res.printResults()

    # 27: We use the 'saveResults' method of 'FitResults' to write a text file
    # containing the fitted parameters and fit quality indices to disk.
    # The file is named based on the basename we created earlier, and
    # written to the 'resdir' directory.
    header = "crystal_HF.\n"
    res.saveResults(resdir / f"{basename}.res", header=header)

    # 28: We use the 'plot_results' method we created earlier to write a pdf file
    # containing the measured and fitted PDF with residual to disk.
    # The file is named based on the 'basename' we created earlier, and
    # written to the 'figdir' directory.
    plot_results(recipe, figdir / basename)

    # End of function


# This tells python to run the 'main' function we defined above.
if __name__ == "__main__":
    main()

# End of file
