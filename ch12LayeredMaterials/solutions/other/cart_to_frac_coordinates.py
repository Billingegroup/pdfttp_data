import numpy as np
from math import *

# Define values for new cell with expanded c 
new_a = 9.076
new_b = 5.298
new_c = 200.
new_alpha = 90.
new_beta = 111.5
new_gamma = 90.

# Define a function to convert the provided cartesian to new fractional coordinates
def cart_to_frac(objSel,a,b,c,alpha,beta,gamma):
    
    """
    cart_to_frac converts atomic cartesian coordinates from an .xyz file in objSel
    to fractional coordinates for a specified cell with lattice parameters a, b, c,
    alpha, beta, and gamma.
    """

    # strip atom names
    floats = np.genfromtxt(objSel,usecols=(0),dtype=None)
    atoms = floats[1:]
    
    # extract xyz coordinates
    cart_coord = np.loadtxt(objSel,skiprows=2,usecols=(1,2,3))

    # convert angles to radians
    alpha = (pi / 180.0) * alpha
    beta  = (pi / 180.0) * beta
    gamma = (pi / 180.0) * gamma
 
    # calculate cell Volume/(abc)
    v = sqrt(1 -cos(alpha)*cos(alpha) - cos(beta)*cos(beta) - cos(gamma)*cos(gamma) + 2*cos(alpha)*cos(beta)*cos(gamma))
 
    # cartesian to fractional coordinate transform matrix
    tmat = np.matrix( [
      [ 1.0 / a, -cos(gamma)/(a*sin(gamma)), (cos(alpha)*cos(gamma)-cos(beta)) / (a*v*sin(gamma))  ],
      [ 0.0,     1.0 / b*sin(gamma),         (cos(beta) *cos(gamma)-cos(alpha))/ (b*v*sin(gamma))  ],
      [ 0.0,     0.0,                        sin(gamma) / (c*v)                                    ] 
      ] )
 
    # apply transformation
    r = tmat * cart_coord.T
    a = np.array(r)

    # print results
    for i in range(len(atoms)):
        print(atoms[i]+str(i)),
        print(atoms[i]),
        print(a[0][i]),
        print(a[1][i]),
        print(a[2][i])


cart_to_frac("../../data/structures/unit_cell_layer.xyz",new_a, new_b, new_c, new_alpha, new_beta, new_gamma)
