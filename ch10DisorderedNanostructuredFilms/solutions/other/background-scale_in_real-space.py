import numpy as np
import matplotlib.pyplot as plt
import math

from scipy.optimize import curve_fit

from scaling_code import scaledPDFs

# Try to find scaling of non-background-subtracted datasets to best match the background subtracted dataset.

a = "../../data/microwave_film/pdfs/ito-glass_noBKGsub.gr"
b = "../../data/microwave_film/pdfs/tio2-ito-glass_noBKGsub.gr"
c = "../../data/microwave_film/pdfs/tio2_minus-ito-glass.gr"

scaledPDFs(a,b,c)
