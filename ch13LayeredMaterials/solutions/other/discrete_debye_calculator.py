import numpy as np
import matplotlib.pyplot as plt

from diffpy.Structure import Structure, loadStructure
from diffpy.srreal.pdfcalculator import DebyePDFCalculator

# Import file containing discrete layer structure
structure = loadStructure(filename="../../data/structures/discrete_alpha_layer.xyz")

# Define an array containing all the chemical species in the structure
formula = ["Zr","P","O"]

# Define a second array with corresponding Biso values
Biso = [0.5,0.5,0.5]
    
# Specify values for further experimental parameters
cfg = { 'qmax' : 18.0,
        'qmin': 1.0,
        'rmin' : 0.01,
        'rmax' : 50.0,
        'qdamp': 0.04,
        'qbroad': 0.02,
        'delta2': 1.0}
    
# Add all atoms to our structure instance and specify the Biso value
for i in [structure]:
    k = Structure(i)
    for n,m in zip(formula,Biso):
        k[k.element == n].Bisoequiv = m

# Calculate the PDF from the structure instance
pc = DebyePDFCalculator(**cfg)
layerPDF =  pc(k)

# Import the experimental PDF
dataset = np.loadtxt("../../data/pdfs/H-Zr_2-1_PP_cutFirstPeak.gr", skiprows=27).T

# Plot the comparison of simulated and experimental PDFS
plt.plot(layerPDF[0], layerPDF[1], label="PDF for a single layer")
plt.plot(dataset[0],(dataset[1])*3.0, label="experimental PDF")
plt.xlim(0.0,45.0)

plt.xlabel("$r (\AA)$")
plt.ylabel("$G (\AA^{-2})$")

plt.show()
