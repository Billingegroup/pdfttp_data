import numpy as np
import matplotlib.pyplot as plt
import math

from scipy.optimize import curve_fit

from scaling_code import scaledPDFs

# Fit extracted TiO2 and ITO PDFs to the PDF of ITO+TiO2.

a = "../../data/microwave_film/pdfs/tio2_minus-ito-glass.gr"
b = "../../data/microwave_film/pdfs/ito-glass_TiO2Comp.gr"
c = "../../data/microwave_film/pdfs/tio2-ito_minus-glass_TiO2Comp.gr"

scaledPDFs(a,b,c)
