"""
Please see notes in Chapter 11 of the 'PDF to the People' book for additonal
explanation of the code.

This Diffpy-CMI script will carry out a structural refinement of a measured
PDF from several TiO2 thin films.  It is the same refinement as is done using PDFgui in this
chapter of the book, only this time using Diffpy-CMI.
"""
# 1: Import relevant system packages that we will need...
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
import re

# ... and the relevant CMI packages
from diffpy.srfit.fitbase import FitContribution, FitRecipe
from diffpy.srfit.fitbase import FitResults
from diffpy.srfit.fitbase import Profile
from diffpy.srfit.pdf import PDFParser, PDFGenerator
from diffpy.structure.parsers import getParser
from diffpy.srfit.structure import constrainAsSpaceGroup
from diffpy.srfit.pdf.characteristicfunctions import sphericalCF

############### Config ##############################
# 2: Give a file path to where your pdf (.gr) and (.cif) files are located.
PWD = Path(__file__).parent.absolute()
DPATH = PWD.parent.parent / "data"


######## Experimental PDF Config ######################
# 3: Specify the min, max, and step r-values of the PDF (that we want to fit over).
PDF_RMIN = 1.5
PDF_RMAX = 30
PDF_RSTEP = 0.01

########PDF initialize refinable variables #############
# 4: We provide initial values for some parameters
SCALE_I = 0.6
DELTA1_I = 0.5
PSIZE_I = 10
SCALE_PHASE_I = 0.5
DATA_SCALE_I = 0.4
U_ISO_I = 0.001

# 5: Instrumental will be fixed based on values obtained from a
# separate calibration step. These are hard-coded here.
QDAMP = 0.0437
QBROAD = 0.0170

# If we want to run using multiprocessors, we can switch this to 'True'.
# This requires that the 'psutil' python package installed.
RUN_PARALLEL = False
OPTI_OPTS = {'ftol': 1e-6, 'gtol': 1e-6, 'xtol': 1e-6}
SHOW_PLOT = False

# 6: Specify strings for the different film types, strings for the associated data sets,
# and associated structure files.
FILM_TYPES = ["conventional", "microwave", "ITO"]
GR_BASE_NAMES = ["tio2-ito-glass", "tio2_minus-ito-glass", "ito_minus-glass"]
CIF_BASE_NAMES = ["TiO2", "TiO2", "ITO"]


######## Functions that will carry out the refinement ##################
# 7: Make the recipe that the fit will follow.
# Here we create a function to make a recipe designed to fit our
# data using one phase. We include the new argument 'nano' to allow us to switch between
# nanocrystalline and bulk samples.
def make_recipe_one_phase(cif_path, dat_path, nano=True):
    """
    Creates and returns a Fit Recipe object using just one phase.

    Parameters
    ----------
    cif_path :  string, The location and filename of the structure XYZ file to load
    dat_path :  string, The full path to the PDF data to be fit.
    nano :      bool, Indicate whether to consider the phase an nanocrystalline,
                      and as such include a spherical damping function.

    Returns
    ----------
    recipe :    The initialized Fit Recipe object using the datname and structure path
                provided.
    """
    # 8: Create a CIF file parsing object, parse and load the structure, and
    # grab the space group name.
    p_cif = getParser('cif')
    stru1 = p_cif.parseFile(cif_path)
    sg = p_cif.spacegroup.short_name

    # 9: Create a Profile object for the experimental dataset and
    # tell this profile the range and mesh of points in r-space.
    profile = Profile()
    parser = PDFParser()
    parser.parseFile(dat_path)
    profile.loadParsedData(parser)
    profile.setCalculationRange(xmin=PDF_RMIN, xmax=PDF_RMAX, dx=PDF_RSTEP)

    # 10: Create a PDF Generator object for a periodic structure model.
    generator_crystal1 = PDFGenerator("G1")
    generator_crystal1.setStructure(stru1, periodic=True)

    # 11: Create a Fit Contribution object.
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

    # 12: Set the Fit Contribution profile to the Profile object.
    contribution.setProfile(profile, xname="r")

    # 13: If we are considering a nanocrystalline sample, adopt one behavior
    if nano:
        # 14: Set an equation, based on your PDF generators. Here we
        # incorporate damping to our PDF to model the effect of finite crystallite size.
        # In this case we use a function which models a spherical NP.
        contribution.registerFunction(sphericalCF, name="f")
        contribution.setEquation("s1*G1*f")
    # 14: If the sample is not nano, adopt another behavior
    else:
        contribution.setEquation("s1*G1")

    # 15: Create the Fit Recipe object that holds all the details of the fit.
    recipe = FitRecipe()
    recipe.addContribution(contribution)

    # 16: Initialize the instrument parameters, Q_damp and Q_broad.
    generator_crystal1.qdamp.value = QDAMP
    generator_crystal1.qbroad.value = QBROAD

    # 17: Add, initialize, and tag variables in the Fit Recipe object.
    # In this case we also add psize, which is the NP size, based onour
    # 'nano' argument.
    if nano:
        recipe.addVar(contribution.psize, PSIZE_I, tag="psize")
    recipe.addVar(generator_crystal1.delta1,
                  name="Delta1", value=DELTA1_I, tag="d1")
    recipe.addVar(contribution.s1, SCALE_I, tag="scale")

    # 18: We explicitly want to tell our structure to not allow anisotropy.
    # We pass in the boolean 'False' to the anisotropy attribute of the structure.
    stru1.anisotropy = False

    # 19: Use the srfit function constrainAsSpaceGroup to constrain
    # the just lattice according to the space group.
    # Note we do not include ADPs  or atomic coordinates in the following.
    spacegroupparams = constrainAsSpaceGroup(generator_crystal1.phase,
                                             sg,
                                             constrainadps=False)

    for par in spacegroupparams.latpars:
        recipe.addVar(par, fixed=False, tag="lat")

    # 20: Get a list of all elements in our structure.  Here we need to
    # strip off non alpha characters, as our input file contains oxidation states.
    els = list(recipe.crystal.G1.phase.stru.composition.keys())
    els = [re.compile('[^a-zA-Z]').sub('', el) for el in els]

    # 21: We create the variables of isotropic ADP for each element.
    # Sn is on a shared site, so it does not get its own parameter.
    for el in els:
        if "Sn" in el:
            continue
        else:
            recipe.newVar(f"{el}_Uiso", value=U_ISO_I, tag="adp")

    # 22: Loop on all atoms in the structure and constrain their
    # ADP to our new paramters. Sn get's constrained to the In ADP.
    for atom in recipe.crystal.G1.phase.atoms:
        for site in els:
            if site in atom.atom.label and "Sn" not in site:
                recipe.constrain(atom.Uiso, f"{site}_Uiso")
            elif site in atom.atom.label and "Sn" in site:
                recipe.constrain(atom.Uiso, f"In_Uiso")

    recipe.restrain("s1",
                    lb=0.01,
                    scaled=True,
                    sig=0.00001)

    if nano:
        recipe.restrain("psize",
                        lb=0.0,
                        ub=200.0,
                        scaled=True,
                        sig=0.00001)

    recipe.restrain("Delta1",
                    lb=0.0,
                    scaled=True,
                    sig=0.00001)

    # 23: Return the Fit Recipe object to be optimized
    return recipe

    # End of function


# 24: Make the recipe that the fit will follow.
# Here we create a second function to make a recipe designed to fit our
# data using two phases.
def make_recipe_two_phase(cif_path1, cif_path2, dat_path):
    """
    Creates and returns a Fit Recipe object using two phases.

    Parameters
    ----------
    cif_path1 :  string, The location and filename of the first structure cif file to load
    cif_path2 :  string, The location and filename of the second structure cif file to load
    dat_path :   string, The full path to the PDF data to be fit.

    Returns
    ----------
    recipe :    The initialized Fit Recipe object using the dat_path and structure paths
                provided.
    """
    # 25: Create a CIF file parsing object, parse and load the structure, and
    # grab the space group name for the first structure.
    p_cif1 = getParser('cif')
    stru1 = p_cif1.parseFile(cif_path1)
    sg1 = p_cif1.spacegroup.short_name

    # 26: Create a CIF file parsing object, parse and load the structure, and
    # grab the space group name for the second structure.
    p_cif2 = getParser('cif')
    stru2 = p_cif2.parseFile(cif_path2)
    sg2 = p_cif2.spacegroup.short_name

    # 27: Create a Profile object for the experimental dataset and
    # tell this profile the range and mesh of points in r-space.
    profile = Profile()
    parser = PDFParser()
    parser.parseFile(dat_path)
    profile.loadParsedData(parser)
    profile.setCalculationRange(xmin=PDF_RMIN, xmax=PDF_RMAX, dx=PDF_RSTEP)

    # 28: Create a PDF Generator object for a periodic structure model
    # of phase 1.
    generator_crystal1 = PDFGenerator("G1")
    generator_crystal1.setStructure(stru1, periodic=True)

    # 29: Create a PDF Generator object for a periodic structure model
    # of phase 2.
    generator_crystal2 = PDFGenerator("G2")
    generator_crystal2.setStructure(stru2, periodic=True)

    # 30: Make convenient lists of the generators and space group names
    generators = [generator_crystal1, generator_crystal2]
    sgs = [sg1, sg2]

    # 31: Create a Fit Contribution objects, one for each phase. This is new, as we
    # need to tell the Fit Contribution about BOTH the phase
    # represented by "generator_crystal1" AND the phase represented
    # by "generator_crystal2".
    contribution = FitContribution("crystal")
    contribution.addProfileGenerator(generator_crystal1)
    contribution.addProfileGenerator(generator_crystal2)

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

    # 32: Set the Fit Contribution profile to the Profile object.
    contribution.setProfile(profile, xname="r")

    # 33: Set an equation, based on your PDF generators. Here we incorporate
    # the effect of finite crystallite size by damping our PDF model.
    # In this case we use a function which models a spherical NP.
    # We use a different instance of the function for each phase.
    # This means we need to explicitly specify the names of the arguments
    # to 'sphericalCF' to avoid naming collisions.
    contribution.registerFunction(
        sphericalCF, name="f1", argnames=['r', 'psize_1'])
    contribution.registerFunction(
        sphericalCF, name="f2", argnames=['r', 'psize_2'])
    contribution.setEquation("data_scale*(s1*G1*f1 + (1.0-s1)*G2*f2)")

    # 34: Create the Fit Recipe object that holds all the details of the fit.
    recipe = FitRecipe()
    recipe.addContribution(contribution)

    # 35: Add, initialize, and tag both scale parameters, the crystal size parameters,
    # and a correlated motion parameter. We will use one correlated motion parameter across
    # both phases.
    recipe.addVar(contribution.s1, SCALE_PHASE_I, tag="phase_scale")
    recipe.addVar(contribution.data_scale, DATA_SCALE_I, tag="scale")
    recipe.addVar(contribution.psize_1, PSIZE_I, tag="psize")
    recipe.addVar(contribution.psize_2, PSIZE_I, tag="psize")
    delta1 = recipe.newVar("Delta1", value=DELTA1_I, tag="d1")

    # 36: restrain our new parameters.
    recipe.restrain("data_scale",
                    lb=0.01,
                    scaled=True,
                    sig=0.00001)

    recipe.restrain("s1",
                    lb=0.00,
                    ub=1.0,
                    scaled=True,
                    sig=0.00001)

    recipe.restrain("psize_1",
                    lb=0.0,
                    ub=200.0,
                    scaled=True,
                    sig=0.00001)

    recipe.restrain("psize_2",
                    lb=0.0,
                    ub=200.0,
                    scaled=True,
                    sig=0.00001)

    recipe.restrain("Delta1",
                    lb=0.0,
                    scaled=True,
                    sig=0.00001)

    # 37: Initialize the instrument parameters, Q_damp and Q_broad.
    generator_crystal1.qdamp.value = QDAMP
    generator_crystal1.qbroad.value = QBROAD

    # 38: Get a list of all elements in our structure.  Here we need to
    # strip off non alpha characters, as our input file contains oxidation states.
    els = list(recipe.crystal.G1.phase.stru.composition.keys())
    els = [re.compile('[^a-zA-Z]').sub('', el) for el in els]

    # 39: We create the variables of isotropic ADP for each element.
    # Sn is on a shared site, so it does not get its own parameter.
    for el in els:
        if "Sn" in el:
            continue
        else:
            recipe.newVar(f"{el}_Uiso", tag="adp")

    # 40: Now, we loop over our list of generators and space groups
    for ii, (generator, sg) in enumerate(zip(generators, sgs)):
        # 41: constrain the delta1 parameter of each generator according to our parameter
        recipe.constrain(generator.delta1, delta1)

        # 42: turn off anisotropy
        generator.phase.stru.anisotropy = False

        # 43: Use the srfit function constrainAsSpaceGroup to constrain
        # the just lattice according to the space group.
        # Note we do not include ADPs  or atomic coordinates in the following.
        spacegroupparams = constrainAsSpaceGroup(generator.phase,
                                                 sg,
                                                 constrainadps=False)

        for par in spacegroupparams.latpars:
            recipe.addVar(
                par, name=f"{par.name}_phase_{ii}", fixed=False, tag="lat")

        # 44: Loop on all atoms in the structure and constrain their
        # ADP to our new paramters. Sn get's constrained to the In ADP.
        atoms = generator.phase.getScatterers()
        for atom in atoms:
            for site in els:
                if site in atom.atom.label and "Sn" not in site:
                    recipe.constrain(atom.Uiso, f"{site}_Uiso")
                elif site in atom.atom.label and "Sn" in site:
                    recipe.constrain(atom.Uiso, f"In_Uiso")

    # 45: Return the Fit Recipe object to be optimized
    return recipe

    # End of function


# We create a useful function 'plot_results' for writing a plot of the fit to disk.
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


# 46: We again create a 'main' function to be run when we execute the script.
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

    # 47: Search our data path for all the .gr files matching a pattern
    # and parse out an identifying string from each file name.
    data_files = list(DPATH.glob(f"./**/*.gr"))
    data_ids = [ff.stem for ff in data_files]

    # 48: Search our data path for all the .cif files matching a pattern
    # and parse out two identifying string from each file name.
    stru_files = list(DPATH.glob(f"./**/*.cif"))
    stru_names = [ff.stem.split("-")[-1] for ff in stru_files]
    stru_chems = [ff.stem.split("-")[0] for ff in stru_files]

    # 49: Loop on the film types and associated identifying strings for the  relevant
    # PDF datafiles and structure files.
    for film_type, gr_base_name, cif_base_name in zip(FILM_TYPES, GR_BASE_NAMES, CIF_BASE_NAMES):

        # 50: Loop on all PDF datafiles and the associated identifying strings.
        for data_file, data_id in zip(data_files, data_ids):

            # 51: If the PDF datafile is not relevant for the type of film we are considering
            # we skip to the next instance of the loop.
            if gr_base_name not in data_id or "Comp" in data_id:
                continue

            # 52: If we are in the microwave film case, we want to proceed before looping on
            # structure files.
            elif film_type == "microwave":

                # 53: Find all relevant structure files.
                rel_structs = [stru_file for stru_file, stru_chem in zip(
                    stru_files, stru_chems) if cif_base_name in stru_chem]

                # 54: We want to consider all TiO2 two-phase permutations where anatase is the first
                # phase, so we need to find the anatase cif.
                anatase_struct = [
                    rel_struct for rel_struct in rel_structs if "anatase" in rel_struct.name][0]

                # 55: Loop on all relevant structures
                for rel_struct in rel_structs:

                    # 56: Build our two phase fit recipe.
                    recipe = make_recipe_two_phase(str(anatase_struct),
                                                   str(rel_struct),
                                                   str(data_file))

                    # 57: Create a unique name for our fit.
                    basename = f"{film_type}_{data_file.stem}_{anatase_struct.stem}_{rel_struct.stem}"
                    print(basename)

                    # Tell the Fit Recipe we want to write the maximum amount of
                    # information to the terminal during fitting.
                    recipe.fithooks[0].verbose = 3

                    # 58: As before, we fix all parameters, create a list of tags and,
                    # loop over them refining sequentially.
                    recipe.fix("all")
                    tags = ["scale", "lat", "psize",
                            "phase_scale", "adp", "d1", "all"]
                    for tag in tags:
                        recipe.free(tag)
                        least_squares(recipe.residual,
                                      recipe.values)

                    # 59: Write the fitted data to a file.
                    profile = recipe.crystal.profile
                    profile.savetxt(fitdir / f"{basename}.fit")

                    # 60: Print the fit results to the terminal.
                    res = FitResults(recipe)
                    res.printResults()

                    # 61: Write the fit results to a file.
                    header = "crystal_HF.\n"
                    res.saveResults(resdir / f"{basename}.res", header=header)

                    # 62: Write a plot of the fit to a (pdf) file.
                    plot_results(recipe, figdir / basename)

            # 63: Now we loop over all structure files to consider one-phase fits.
            for stru_file, stru_name, stru_chem in zip(stru_files, stru_names, stru_chems):

                # 64: If the structure file is not relevant for the type of film we are considering
                # we skip to the next instance of the loop.
                if cif_base_name not in stru_chem:
                    continue

                # 65: In the following we consider all the different PDF datafile, structure files, and
                # film type permutations of relevance.
                # if we encounter a relevant permutation, we create a fit recipe and define
                # a list of relevant parameters
                elif (film_type == "conventional" and "450" in data_id and "anatase" in stru_name) or \
                     (film_type == "conventional" and "250" in data_id) or \
                     (film_type == "microwave"):
                    recipe = make_recipe_one_phase(str(stru_file),
                                                   str(data_file))
                    tags = ["scale", "lat", "psize", "adp", "d1", "all"]

                # 66: When we consider the ITO substrate, we do not want to include
                # nanocrystalline PDF damping.
                elif film_type == "ITO" and "tio2" not in data_id:
                    recipe = make_recipe_one_phase(str(stru_file),
                                                   str(data_file),
                                                   nano=False)
                    tags = ["scale", "lat", "adp", "d1", "all"]

                # 67: If we find no relevant permutations, we skip to the next iteration
                # of the loop.
                else:
                    continue

                # 68:
                basename = f"{film_type}_{data_file.stem}_{stru_file.stem}"
                print(basename)

                # Tell the Fit Recipe we want to write the maximum amount of
                # information to the terminal during fitting.
                recipe.fithooks[0].verbose = 3

                # 69: As before, we fix all parameters, create a list of tags and,
                # loop over them refining sequentially. We've added our new 'occs'
                # tag to the list.
                recipe.fix("all")
                for tag in tags:
                    recipe.free(tag)
                    least_squares(recipe.residual,
                                  recipe.values)

                # 70: Write the fitted data to a file.
                profile = recipe.crystal.profile
                profile.savetxt(fitdir / f"{basename}.fit")

                # 71: Print the fit results to the terminal.
                res = FitResults(recipe)
                res.printResults()

                # 72: Write the fit results to a file.
                header = "crystal_HF.\n"
                res.saveResults(resdir / f"{basename}.res", header=header)

                # 73: Write a plot of the fit to a (pdf) file.
                plot_results(recipe, figdir / basename)

    # End of function


# This tells python to run the 'main' function we defined above.
if __name__ == "__main__":
    main()

# End of file
