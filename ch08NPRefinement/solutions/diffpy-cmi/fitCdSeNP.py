"""
Please see notes in Chapter 9 of the 'PDF to the People' book for additonal
explanation of the code.

This Diffpy-CMI script will carry out a structural refinement of various measured
PDFs from CdSe nanoparticles.  It is the same refinements as are done using PDFgui in this
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
from diffpy.srfit.structure import constrainAsSpaceGroup
from diffpy.srfit.pdf.characteristicfunctions import sphericalCF

############### Config ##############################
# 2: Give a file path to where your pdf (.gr) and (.cif) files are located.
PWD = Path(__file__).parent.absolute()
DPATH = PWD.parent.parent / "data"

# 3: Give an identifying name for the refinement.
FIT_ID_BASE = 'Fit_CdSe'

# 4: Specify the names of the input PDF file.
GR_NAME_SEARCH_STRING = "CdSe"

# 5: Specify the filename for the structure we'd like to use.
STRU_FILE_SEARCH_STRING = GR_NAME_SEARCH_STRING


######## Experimental PDF Config ######################
# 6: Specify the min, max, and step r-values of the PDF (that we want to fit over)
# also, specify the Q_max and Q_min values used to reduce the PDF.
PDF_RMIN = 0.5
PDF_RMAX = 40
PDF_RSTEP = 0.01
QMAX = 19
QMIN = 1.0

########PDF initialize refinable variables #############
# 7: In this case, initial values for the lattice parameters
# and ADPs, for both phases, will be taken directly from
# the .cif structures.
SCALE_I = 0.1
DELTA2_I = 5
PSIZE_I = 40
UISO_I = 0.01
SCALE_PHASE_I = 0.50
DATA_SCALE_I = SCALE_I


# 8: Instrumental will be fixed based on values obtained from a
# separate calibration step. These are hard-coded here.
QDAMP = 0.058
QBROAD = 0.0

RUN_PARALLEL = False
OPTI_OPTS = {'ftol': 1e-3, 'gtol': 1e-5, 'xtol': 1e-4}
SHOW_PLOT = False


######## Functions that will carry out the refinement ##################
# 9: Make the recipe that the fit will follow.
# Here we create a function to make a recipe designed to fit our
# data using one phase.
def make_recipe_one_phase(cif_path, dat_path, adp_iso=True):
    """
    Creates and returns a Fit Recipe object using just one phase.

    Parameters
    ----------
    cif_path :  string, The location and filename of the structure XYZ file to load
    dat_path :  string, The full path to the PDF data to be fit.
    adp_iso :   bool, specifiying if ADPs are isotropic (True) or anisotropic (False).

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
    # of complexity, incorporating "f" int our equation. This new term
    # incorporates damping to our PDF to model the effect of finite crystallite size.
    # In this case we use a function which models a spherical NP.
    contribution.registerFunction(sphericalCF, name="f")
    contribution.setEquation("s1*G1*f")

    # 16: Create the Fit Recipe object that holds all the details of the fit.
    recipe = FitRecipe()
    recipe.addContribution(contribution)

    # 17: Initialize the instrument parameters, Q_damp and Q_broad, and
    # assign Q_max and Q_min.
    generator_crystal1.qdamp.value = QDAMP
    generator_crystal1.qbroad.value = QBROAD
    generator_crystal1.setQmax(QMAX)
    generator_crystal1.setQmin(QMIN)

    # 18: Add, initialize, and tag variables in the Fit Recipe object.
    # In this case we also add psize, which is the NP size.
    recipe.addVar(contribution.s1, SCALE_I, tag="scale")
    recipe.addVar(contribution.psize, PSIZE_I, tag="psize")
    recipe.addVar(generator_crystal1.delta2,
                  name="Delta2", value=DELTA2_I, tag="d2")

    # 19: Now, we want to have one behavior if we desire isotropic ADPs
    # and another behavior if we desire anistropic ADPs. To achive this we
    # use our 'adp_iso' argument and an 'if..else' statement.
    if adp_iso:
        # 20: We explicitly want to tell our structure to not allow anisotropy.
        # We pass in the boolean 'False' to the anisotropy attribute of the structure.
        stru1.anisotropy = False
        # 21: Use the srfit function constrainAsSpaceGroup to constrain
        # the lattice and atomic positions according to the space group.
        # Note we do not include ADPs in the following.
        spacegroupparams = constrainAsSpaceGroup(generator_crystal1.phase,
                                                 sg,
                                                 constrainadps=False)

        for par in spacegroupparams.latpars:
            recipe.addVar(par, fixed=False, tag="lat")
        for par in spacegroupparams.xyzpars:
            recipe.addVar(par, fixed=False, tag="xyz")

        # 22: We create the variables of isotropic ADP and assign the initial value to them,
        # specified above. In this portion of the 'if' statement, we use isotropic
        # ADP for all atoms
        cd_uiso = recipe.newVar("Cd_Uiso", value=UISO_I, tag="adp")
        se_uiso = recipe.newVar("Se_Uiso", value=UISO_I, tag="adp")

        # 23: For all atoms in the structure model, we constrain their Uiso according to
        # their species.
        atoms = generator_crystal1.phase.getScatterers()
        for atom in atoms:
            if atom.element == 'Cd':
                recipe.constrain(atom.Uiso, cd_uiso)
            elif atom.element == 'Se':
                recipe.constrain(atom.Uiso, se_uiso)

    # 24: Now, we want to have one behavior if we desire isotropic ADPs
    # and another behavior if we desire anistropic ADPs. To achive this we
    # use our 'adp_iso' argument and an 'if..else' statement.
    else:
        # 25: We explicitly want to tell our structure to allow anisotropy.
        # We pass in the boolean 'True' to the anisotropy attribute of the structure.
        stru1.anisotropy = True

        # 26: Use the srfit function constrainAsSpaceGroup to constrain
        # the lattice and atomic positions according to the space group.
        # Note we do include ADPs in the following.
        spacegroupparams = constrainAsSpaceGroup(generator_crystal1.phase,
                                                 sg)

        for par in spacegroupparams.latpars:
            recipe.addVar(par, fixed=False, tag="lat")
        for par in spacegroupparams.adppars:
            recipe.addVar(par, fixed=False, tag="adp")
        for par in spacegroupparams.xyzpars:
            recipe.addVar(par, fixed=False, tag="xyz")

    # 27: Return the Fit Recipe object to be optimized
    return recipe

    # End of function


# 28: Make the recipe that the fit will follow.
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
    # 29: Create a CIF file parsing object, parse and load the structure, and
    # grab the space group name for the first structure.
    p_cif1 = getParser('cif')
    stru1 = p_cif1.parseFile(cif_path1)
    sg1 = p_cif1.spacegroup.short_name

    # 30: Create a CIF file parsing object, parse and load the structure, and
    # grab the space group name for the second structure.
    p_cif2 = getParser('cif')
    stru2 = p_cif2.parseFile(cif_path2)
    sg2 = p_cif2.spacegroup.short_name

    # 31: Create a Profile object for the experimental dataset and
    # tell this profile the range and mesh of points in r-space.
    profile = Profile()
    parser = PDFParser()
    parser.parseFile(dat_path)
    profile.loadParsedData(parser)
    profile.setCalculationRange(xmin=PDF_RMIN, xmax=PDF_RMAX, dx=PDF_RSTEP)

    # 32: Create a PDF Generator object for a periodic structure model
    # of phase 1.
    generator_crystal1 = PDFGenerator("G1")
    generator_crystal1.setStructure(stru1, periodic=True)

    # 33: Create a PDF Generator object for a periodic structure model
    # of phase 2.
    generator_crystal2 = PDFGenerator("G2")
    generator_crystal2.setStructure(stru2, periodic=True)

    # 33: Make convenient lists of the generators and space group names
    generators = [generator_crystal1, generator_crystal2]
    sgs = [sg1, sg2]

    # 34: Create a Fit Contribution objects, one for each phase. This is new, as we
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

    # 35: Set the Fit Contribution profile to the Profile object.
    contribution.setProfile(profile, xname="r")

    # 36: Set an equation, based on your PDF generators. Here we add an extra layer
    # of complexity, incorporating "f" int our equation. This new term
    # incorporates damping to our PDF to model the effect of finite crystallite size.
    # In this case we use a function which models a spherical NP.
    # We use the same function for each phase.
    contribution.registerFunction(sphericalCF, name="f")
    contribution.setEquation("data_scale*(s1*G1*f + (1.0-s1)*G2*f)")

    # 37: Create the Fit Recipe object that holds all the details of the fit.
    recipe = FitRecipe()
    recipe.addContribution(contribution)

    # 38: Add, initialize, and tag both scale parameters, the crystal size parameter,
    # and a correlated motion parameter. We will use one correlated motion parameter across
    # both phases.
    recipe.addVar(contribution.s1, SCALE_PHASE_I, tag="phase_scale")
    recipe.addVar(contribution.data_scale, DATA_SCALE_I, tag="scale")
    recipe.addVar(contribution.psize, PSIZE_I, tag="psize")
    delta2 = recipe.newVar("Delta2", value=DELTA2_I, tag="d2")

    # 39: restrain our new parameters.
    recipe.restrain("data_scale",
                    lb=0.0,
                    scaled=True,
                    sig=0.00001)

    recipe.restrain("s1",
                    lb=0.0,
                    ub=1.0,
                    scaled=True,
                    sig=0.00001)

    recipe.restrain("psize",
                    lb=0.0,
                    scaled=True,
                    sig=0.00001)

    # 40: Initialize the instrument parameters, Q_damp and Q_broad, and
    # assign Q_max and Q_min.
    generator_crystal1.qdamp.value = QDAMP
    generator_crystal1.qbroad.value = QBROAD
    generator_crystal1.setQmax(QMAX)
    generator_crystal1.setQmin(QMIN)

    # 41: We create the variables of ADP and assign the initial value to them,
    # specified above. In this example, we use isotropic ADP for all atoms.
    # We will use these across both phases
    cd_uiso = recipe.newVar("Cd_Uiso", value=UISO_I, tag="adp")
    se_uiso = recipe.newVar("Se_Uiso", value=UISO_I, tag="adp")

    # 42: Now, we loop over our list of generators and space groups
    for generator, sg in zip(generators, sgs):
        # 43: The minus sign '-' will cause problems in parameter naming, so
        # we should remove it.
        sg_clean = sg.replace("-", "")

        # 44: constrain the delta2 parameter of each generator according to our parameter
        recipe.constrain(generator.delta2, delta2)

        # 45: turn off anisotropy
        generator.phase.stru.anisotropy = False

        # 46: Use the srfit function constrainAsSpaceGroup to constrain
        # the lattice, ADP parameters, and atomic positions according to the space group.
        spacegroupparams = constrainAsSpaceGroup(generator.phase,
                                                 sg,
                                                 constrainadps=False)

        for par in spacegroupparams.latpars:
            recipe.addVar(
                par, name=f"{par.name}_phase_{sg_clean}", fixed=False, tag="lat")
        for par in spacegroupparams.xyzpars:
            recipe.addVar(
                par, name=f"{par.name}_phase_{sg_clean}", fixed=False, tag="xyz")

        # 47: We do not use the ADP portion of spacegroupparams, instead we add isotropic ADPs
        # We use one set across both phases.
        atoms = generator.phase.getScatterers()
        for atom in atoms:
            if atom.element == 'Cd':
                recipe.constrain(atom.Uiso, cd_uiso)
            elif atom.element == 'Se':
                recipe.constrain(atom.Uiso, se_uiso)

    # 48: Return the Fit Recipe object to be optimized
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


# 49: We again create a 'main' function to be run when we execute the script.
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

    # 50: Search our data path for all the .gr files matching a pattern
    # and parse out an identifying string from each file name.
    data_files = list(DPATH.glob(f"*{GR_NAME_SEARCH_STRING}*.gr"))
    data_ids = [ff.stem.split("-")[-1] for ff in data_files]

    # 51: Search our data path for all the .cif files matching a pattern
    # and parse out an identifying string from each file name.
    stru_files = list(DPATH.glob(f"*{STRU_FILE_SEARCH_STRING}*.cif"))
    stru_names = [ff.stem.split("-")[-1] for ff in stru_files]

    # 52: We want to handle both isotropic and anisotropic ADP constraints,
    # so we create a list of strings to keep track of which case we are
    # working on
    adp_names = ["isotropic", "anisotropic"]

    # 53: We want to loop on all the PDF data files we found, and their
    # associated identifying string.
    for data_file, data_id in zip(data_files, data_ids):

        # 54: We are going to programmatically create a name for our fit,
        # which will include the root base name we chose at the beginning of the
        # script, as well as identifying string parsed out from the PDF data file.
        basename = f"{FIT_ID_BASE}_{data_id}_two_phase"
        print(basename)

        # 55: We call our two phase make recipe function to build our recipe.
        # We do this outside our loop on structure files, because we need to
        # include both our structure files here.
        recipe = make_recipe_two_phase(str(stru_files[0]),
                                       str(stru_files[1]),
                                       str(data_file))

        # 56: As before, we fix all parameters, create a list of tags and,
        # loop over them refining sequentially.
        recipe.fix("all")
        recipe.fithooks[0].verbose = 3
        tags = ["scale", "psize", "lat", "adp", "d2", "phase_scale", "all"]
        for tag in tags:
            recipe.free(tag)
            least_squares(recipe.residual, recipe.values,
                          x_scale="jac", **OPTI_OPTS)

        # 57: Write the fitted data to a file.
        profile = recipe.crystal.profile
        profile.savetxt(fitdir / f"{basename}.fit")

        # 58: Print the fit results to the terminal.
        res = FitResults(recipe)
        res.printResults()

        # 59: Write the fit results to a file.
        header = "crystal_HF.\n"
        res.saveResults(resdir / f"{basename}.res", header=header)

        # 60: Write a plot of the fit to a (pdf) file.
        plot_results(recipe, figdir / basename)

        # 61: Now we loop on every structure file we found as well as
        # the identifying string we parsed from each file name.
        for stru_file, stru_name in zip(stru_files, stru_names):
            # 62: We loop over all type pf ADP symmetry constraints.
            for adp_symm in adp_names:

                # 63: We are going to programmatically create a name for our fit,
                # which will include the root base name we chose at the beginning of the
                # script, the identifying string parsed out from the PDF data file,
                # the structure we are using, and whether we are workin in with isotropic or
                # anisotropic constraints on our ADPs.
                basename = f"{FIT_ID_BASE}_{data_id}_{stru_name}_{adp_symm}_ADPs"
                print(basename)

                # 64: We call our single phase make recipe function to build our recipe.
                # We do this inside our loop on structure files, so we we test each structure
                # file seperately. We also have a simple logic evaluation to decide
                # what boolean to pass to our new 'adp_symm' argument.
                recipe = make_recipe_one_phase(str(stru_file),
                                               str(data_file),
                                               True if adp_symm == "isotropic" else False)

                # 65: In some cases, the symmetry of the space group means that isotropic and
                # anisotropic constraints on our ADPs yield essentially equivalent models.
                # here, we check to see if all the ADP parameters in our fit are isotropic,
                # and if we are attempting to use anisotropic, we skip the fitting.
                recipe.fix("all")
                recipe.free("adp")
                if np.all(["iso" in par for par in recipe.getNames()]) and adp_symm == "anisotropic":
                    continue

                # 66: As before, we fix all parameters, create a list of tags and,
                # loop over them refining sequentially.
                recipe.fix("all")
                recipe.fithooks[0].verbose = 3
                tags = ["scale", "psize", "lat", "adp", "d2", "all"]
                for tag in tags:
                    recipe.free(tag)
                    least_squares(recipe.residual, recipe.values,
                                  x_scale="jac", **OPTI_OPTS)

                # 67: Write the fitted data to a file.
                profile = recipe.crystal.profile
                profile.savetxt(fitdir / f"{basename}.fit")

                # 68: Print the fit results to the terminal.
                res = FitResults(recipe)
                res.printResults()

                # 69: Write the fit results to a file.
                header = "crystal_HF.\n"
                res.saveResults(resdir / f"{basename}.res", header=header)

                # 70: Write a plot of the fit to a (pdf) file.
                plot_results(recipe, figdir / basename)

    # End of function


# This tells python to run the 'main' function we defined above.
if __name__ == "__main__":
    main()

# End of file
