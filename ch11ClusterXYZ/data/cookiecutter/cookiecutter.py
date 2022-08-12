#!/usr/bin/env python
"""Make a cookie out of a diffpy.Structure object."""

__id__ = "$Id: makeEllipsoid.py 3032 2009-04-08 19:15:37Z juhas $"

import math
from itertools import starmap, product, izip

import numpy

from diffpy.Structure import Atom, Lattice, Structure

_toler = 1e-8

class CookieCutter(object):
    """Class for cutting a cookie out of a diffpy.Structure object.

    Attributes
    origin      --  The origin of the cookie cutter in Cartesian coordinates
                    in the world frame (numpy.array). It is up to the cookie
                    cutter to define what point relative to the cookie cutter
                    is the origin.
    orient      --  The Z-X-Z euler angles (numpy.array) that give the
                    orientation of the cookie cutter (about its origin) in the
                    world frame. The first angle (phi) defines the rotation
                    about the z axis that takes x to x'. The second angle
                    (theta) is about the x' axis and takes z to z'. The third
                    angle (psi) defines the rotation about the z' axis.

    Properties

    x, y, z         --  The origin coordinates.
    phi, theta, psi --  The Euler angles.
    maxrads         --  The maximum radii along the cardinal directions, read
                        only, same as _maxRads method.

    Class Attributes
    toler       --  Tolerance of measurements, 1e-8.
    pars        --  The names of the shape parameters that are specific to a
                    subclass. This must be overloaded.

    """

    toler = _toler
    pars = []

    x = property( lambda self: self.origin[0],
                  lambda self, v: self.origin.__setitem__(0, v))
    y = property( lambda self: self.origin[1],
                  lambda self, v: self.origin.__setitem__(1, v))
    z = property( lambda self: self.origin[2],
                  lambda self, v: self.origin.__setitem__(2, v))
    phi = property( lambda self: self.orient[2],
                  lambda self, v: self.orient.__setitem__(2, v))
    theta = property( lambda self: self.orient[1],
                  lambda self, v: self.orient.__setitem__(1, v))
    psi = property( lambda self: self.orient[0],
                  lambda self, v: self.orient.__setitem__(0, v))
    maxrads = property(lambda self: self._maxRads())

    def __init__(self):
        self.origin = numpy.zeros(3)
        self.orient = numpy.zeros(3)
        return

    def isIn(self, point):
        """Tell whether a point (in Cartesian) is within the coookie cutter.

        The point is in the frame of the cookie cutter. This can be achieved
        with the putInFrame method.

        """
        raise NotImplementedError("Overload me!")

    def putInFrame(self, point, m = None):
        """Transform a point (in Cartesian) to the frame of the cookie cutter.

        point   --  The point to transform
        m       --  The rotation matrix from the Euler angles. If this is None
                    (default), then it will be automatically generated.

        Returns the point translated into the cookie cutter frame.

        """
        p = numpy.subtract(point, self.origin)
        if m is None: m = _euler(*self.orient)
        # Multiply on the right for the inverse operation:
        # (p * M).T = M.T * p.T = M.I * p.T
        p = numpy.dot(m, p)
        return p

    def putInWorld(self, point, m = None):
        """Transform a point (in Cartesian) from the frame of the cookie cutter.

        point   --  The point to transform
        m       --  The rotation matrix from the Euler angles. If this is None
                    (default), then it will be automatically generated.

        Returns the point translated into the world frame from the cookie
        cutter frame.

        """
        p = numpy.subtract(point, self.origin)
        if m is None: m = _euler(*self.orient)
        p = numpy.dot(p, m)
        return p


    def _maxRads(self, point):
        """The maximum radii of the cookie cutter in world frame.

        Returns a tuple of the maximum distances from the origin of the cookie
        cutter to its perimiter, translated to the world frame.

        This is used to cut out the cookie. Be sure to overload this method.

        """
        raise NotImplementedError("Overload me!")

    def cut(self, stru):
        """Make a cookie from a diffpy.Structure.
        """
        cookie = Structure(lattice = stru.lattice)
        m = _euler(*self.orient)
        # Get the extent of the cell in all directions.
        d = self.maxrads
        abc = stru.lattice.abcABG()[:3]
        mind = self.origin - d - abc
        maxd = self.origin + d + abc
        mindf = stru.lattice.fractional(mind)
        maxdf = stru.lattice.fractional(maxd)
        mindfi = map(int, numpy.floor(mindf))
        maxdfi = map(int, 1 + numpy.ceil(maxdf))
        bounds = zip(mindfi, maxdfi)
        ranges = starmap(xrange, bounds)
        for i, j, k in product(*ranges):
            for a in stru:
                xyzf = a.xyz + [i, j, k]
                xyzc = stru.lattice.cartesian(xyzf)
                xyzcookie = self.putInFrame(xyzc, m)
                if self.isIn(xyzcookie):
                    cookie.append(Atom(a, xyzf))

        return cookie


# End class CookieCutter

class SphericalCookieCutter(CookieCutter): #inherit
    """Cookie cutter for a sphere.

    Attributes
    origin      --  The origin of the cookie cutter in Cartesian coordinates
                    in the world frame (numpy.array). The in cookie cutter
                    coordinates, the origin is at the center of the sphere.
    orient      --  The Z-X-Z euler angles (numpy.array) that give the
                    orientation of the cookie cutter (about its origin) in the
                    world frame. The first angle (phi) defines the rotation
                    about the z axis that takes x to x'. The second angle
                    (theta) is about the x' axis and takes z to z'. The third
                    angle (psi) defines the rotation about the z' axis.
    radius      --  The radius of the sphere.

    Properties

    x, y, z         --  The origin coordinates.
    phi, theta, psi --  The Euler angles.

    """

    pars = ["radius"]

    def __init__(self):
        CookieCutter.__init__(self)
        self.radius = 10
        return

    def isIn(self, point):
        """Tell whether a point (in Cartesian) is within the coookie cutter.

        Points on the boundary are considered within the sphere.

        """
        r = numpy.linalg.norm(point)
        return r <= self.radius + self.__class__.toler

    def _maxRads(self):
        return [2 * self.radius]*3

# End class SphericalCookieCutter

class CylindricalCookieCutter(CookieCutter):
    """Cookie cutter for a right-cylinder.

    Attributes
    origin      --  The origin of the cookie cutter in Cartesian coordinates
                    in the world frame (numpy.array). The in cookie cutter
                    coordinates, the origin is at the center of the cylinder.
    orient      --  The Z-X-Z euler angles (numpy.array) that give the
                    orientation of the cookie cutter (about its origin) in the
                    world frame. The first angle (phi) defines the rotation
                    about the z axis that takes x to x'. The second angle
                    (theta) is about the x' axis and takes z to z'. The third
                    angle (psi) defines the rotation about the z' axis.
    radius      --  The radius of the cylinder.
    height      --  Height of the cylinder.

    Properties

    x, y, z         --  The origin coordinates.
    phi, theta, psi --  The Euler angles.

    """

    pars = ["radius", "height"]

    def __init__(self):
        CookieCutter.__init__(self)
        self.radius = 10
        self.height = 10
        return

    def isIn(self, point):
        """Tell whether a point (in Cartesian) is within the cookie cutter.

        Points on the boundary are considered inside.

        """
        p = point
        inxy = (p[0]**2 + p[1]**2)**0.5 <= self.radius + self.__class__.toler
        inz = numpy.fabs(p[2]) <= self.height / 2 + self.__class__.toler
        return inxy and inz

    def _maxRads(self):
        dx = [self.radius, 0, 0]
        dy = [0, self.radius, 0]
        dz = [0, 0, self.height/2]
        m = _euler(*self.orient)
        wdx = numpy.fabs(numpy.dot(dx, m))
        wdy = numpy.fabs(numpy.dot(dy, m))
        wdz = numpy.fabs(numpy.dot(dz, m))
        return wdx + wdy + wdz

# End class CylindricalCookieCutter

class RohomboidCookieCutter(CookieCutter):
    """Cookie cutter for a 4-sized prism.

    Attributes
    origin      --  The origin of the cookie cutter in Cartesian coordinates
                    in the world frame (numpy.array). In the cookie cutter
                    frame, the origin is in the corner, you know.
    orient      --  The Z-X-Z euler angles (numpy.array) that give the
                    orientation of the cookie cutter (about its origin) in the
                    world frame. The first angle (phi) defines the rotation
                    about the z axis that takes x to x'. The second angle
                    (theta) is about the x' axis and takes z to z'. The third
                    angle (psi) defines the rotation about the z' axis.
    radius      --  The radius of the cylinder.
    height      --  Height of the cylinder.
    _lattice    --  A diffpy.Structure.Lattice instance that helps with the
                    geometry. It is safe to modify this directly or swap it
                    out.

    Properties

    x, y, z         --  The origin coordinates.
    phi, theta, psi --  The Euler angles.
    ap, bp, cp      --  The side lengths of the prism (default 1). These are
                        tied to the _lattice attributes of the same name.
    alphap,
    betap,
    gammap          --  The angles (degrees) of the prism, defined in the usual
                        sense (default 90). These are tied to the _lattice
                        attributes of the same name.


    """

    pars = ["ap", "bp", "cp", "alphap", "betap", "gammap"]

    ap = property(lambda self: self._lattice.a,
            lambda self, v: self._lattice.setLatPar(a = v))
    bp = property(lambda self: self._lattice.b,
            lambda self, v: self._lattice.setLatPar(b = v))
    cp = property(lambda self: self._lattice.c,
            lambda self, v: self._lattice.setLatPar(c = v))
    alphap = property(lambda self: self._lattice.alpha,
            lambda self, v: self._lattice.setLatPar(alpha = v))
    betap = property(lambda self: self._lattice.beta,
            lambda self, v: self._lattice.setLatPar(beta = v))
    gammap = property(lambda self: self._lattice.gamma,
            lambda self, v: self._lattice.setLatPar(gamma = v))

    def __init__(self):
        CookieCutter.__init__(self)
        self._lattice = Lattice()
        return

    def isIn(self, point):
        """Tell whether a point (in Cartesian) is within the coookie cutter.

        Points on the boundary are considered within the sphere.

        """
        pf = self._lattice.fractional(point)
        retval = numpy.logical_and(pf >= 0, pf <= 1).all()
        return retval

    def _maxRads(self):
        dx = self._lattice.cartesian([1, 0, 0])
        dy = self._lattice.cartesian([0, 1, 0])
        dz = self._lattice.cartesian([0, 0, 1])
        m = _euler(*self.orient)
        wdx = numpy.fabs(numpy.dot(dx, m))
        wdy = numpy.fabs(numpy.dot(dy, m))
        wdz = numpy.fabs(numpy.dot(dz, m))
        return wdx + wdy + wdz

# End class RohomboidCookieCutter

class CookieDecorator(object):
    """A class for decorating a cookie.

    This is still a work in progress.

    Attributes
    cc      --  A CookieCutter instance. This class may use the origin and
                orient attributes of the CookieCutter.

    Class Attributes
    toler   --  Tolerance of measurements, 1e-8.
    pars    --  The names of the shape parameters that are specific to a
                subclass. This must be overloaded.

    """
    toler = _toler
    pars = []

    def __init__(self, cc):
        self.cc = cc
        return

    def decorate(self, cookie):
        """Decorate the cookie.

        This returns a new, decorated cookie.

        """
        raise NotImplementedError("Overload me!")

# Enc class CookieDecorator

class CylindricalQuadraticExpansionDecorator(CookieDecorator):
    """Decorator that expands atoms along the radius of a cylinder.

    This is meant to be used with a cookie cutter with cylindrical symmetry.

    Attributes
    cc      --  A CookieCutter instance.
    eradius --  The expansion coeffient (default 0) in the radial direction.
    eheight --  The expansion coeffient (default 0) along the z-axis.

    The expansion factor is (1 + e * x), where e is eradius or eheight, and x
    is either the radius or height ordinate of the position. This is a bigger
    expansion at larger distance from the origin.

    Class Attributes
    toler   --  Tolerance of measurements, 1e-8.
    pars    --  The names of the shape parameters.

    """
    pars = ["eradius", "eheight"]

    def __init__(self, cc):
        CookieDecorator.__init__(self, cc)
        self.eradius = 0.0
        self.eheight = 0.0
        return

    def decorate(self, stru):
        """Expand cookie by factor 1 + zeta in radial direction."""
        cookie = Structure(lattice = stru.lattice)
        er = self.eradius
        eh = self.eheight
        m = _euler(*self.cc.orient)
        for a in stru:
            xyzf = a.xyz
            xyzc = stru.lattice.cartesian(xyzf)
            xyzccookie = self.cc.putInFrame(xyzc, m)
            r = (xyzccookie[0]**2 + xyzccookie[1]**2)**0.5
            h = xyzccookie[2]
            xyzccookie *= [1 + er * r, 1 + er * r, 1 + eh * h]
            xyzccookiew = self.cc.putInWorld(xyzccookie, m)
            xyzfcookie = stru.lattice.fractional(xyzccookiew)
            cookie.append(Atom(a, xyzfcookie))

        return cookie

# End class CylindricalQuadraticExpansionDecorator

class CoordinationShiftDecorator(CookieDecorator):
    """Decorator that shifts undercoordinated atoms.

    This shifts atoms with specified coordination towards or away from the
    origin.

    Attributes
    cc      --  A CookieCutter instance.
    eps     --  The adjustment factor.
    element --  The type of atom to apply this to.
    coord   --  The cooridnation numbers to apply the eps to.
    cutoff  --  Cutoff for coordination

    The shift factor is (1 + eps).

    Class Attributes
    toler   --  Tolerance of measurements, 1e-8.
    pars    --  The names of the shape parameters.

    """
    pars = ["eps"]

    def __init__(self, cc):
        CookieDecorator.__init__(self, cc)
        self.element = "Z"
        self.coord = [0]
        self.eps = 0
        self.cutoff = 0
        return

    def decorate(self, stru):
        """Expand coordinated atoms by factor 1 + eps."""
        atoms = [a for a in stru if a.element.title() == self.element.title()]
        nn = []
        for a in atoms:
            nbrs = [b for b in stru if b is not a and
                    stru.lattice.dist(a.xyz, b.xyz) <= self.cutoff]
            nn.append(nbrs)
        nncount = map(len, nn)
        for a, n in zip(atoms, nncount):
            if n in self.coord:
                a.xyz = a.xyz * (1 + self.eps)
        return stru

# End class CoordinationShiftDecorator

class CoordinationSmearDecorator(CookieDecorator):
    """Decorator that moves undercoordinated atoms.

    Attributes
    cc      --  A CookieCutter instance.
    smear   --  The smear factor.
    element --  The type of atom to apply this to.
    coord   --  The cooridnation number to apply the smear to.
    cutoff  --  Cutoff for coordination

    Class Attributes
    toler   --  Tolerance of measurements, 1e-8.
    pars    --  The names of the shape parameters.

    """
    pars = ["smear"]

    def __init__(self, cc):
        CookieDecorator.__init__(self, cc)
        self.element = "Z"
        self.coord = 0
        self.smear = 0
        self.cutoff = 0
        return

    def decorate(self, stru):
        """Expand coordinated atoms by factor 1 + smear."""
        atoms = [a for a in stru if a.element.title() == self.element.title()]
        nn = []
        for a in atoms:
            nbrs = [b for b in stru if b is not a and
                    stru.lattice.dist(a.xyz, b.xyz) <= self.cutoff]
            nn.append(nbrs)
        nncount = map(len, nn)
        for a, n in zip(atoms, nncount):
            if n == self.coord:
                a.U11 += self.smear
                a.U22 += self.smear
                a.U33 += self.smear
                a.Uisoequiv += self.smear
        return stru

# End class CoordinationSmearDecorator

from diffpy.srfit.pdf import DebyePDFGenerator
from diffpy.srfit.fitbase.parameter import ParameterAdapter
from diffpy.srreal.srreal_ext import nosymmetry

class CookiePDFGenerator(DebyePDFGenerator):
    """SrFit generator object for cookies!

    cc      --  A CookieCutter instance
    declist --  A list of CookieDecorator instances

    """

    def __init__(self, name = "pdf"):
        """Initialize the generator.

        """
        DebyePDFGenerator.__init__(self, name)
        self.declist = []
        return

    def setCookieCutter(self, cc):
        """Set the cookie cutter.

        This turns the origin and orientation of the cookie into paramters as
        well as the shape parameters. Watch out for name conflicts.

        """
        self.cc = cc
        pars = ["x", "y", "z", "phi", "theta", "psi"]
        pars += cc.__class__.pars
        for pname in pars:
            self.addParameter(
                ParameterAdapter(pname, self.cc, attr = pname)
                )
        return

    def addCookieDecorator(self, dec, names = None):
        """Add a cookie decorator.

        The cookie decorators will be applied after the cookiecutter in the
        order they were added to this generator. If names is provided, it will
        rename the parameters from the cookie decorator for use in the
        generator.

        """
        self.declist.append(dec)
        pars = dec.__class__.pars
        if names is None:
            names = pars
        for pname, aname in zip(names, pars):
            self.addParameter(
                ParameterAdapter(pname, dec, attr = aname)
                    )
        return

    def __call__(self, r):
        """Calculate the PDF.

        This ProfileGenerator will be used in a fit equation that will be
        optimized to fit some data.  By the time this function is evaluated,
        the crystal has been updated by the optimizer via the ObjCrystParSet
        created in setCrystal. Thus, we need only call pdf with the internal
        structure object.

        """
        if r is not self._lastr:
            self._prepare(r)

        # Get the cookie
        cookie = self.cc.cut(self._phase.stru)
        for dec in self.declist:
            cookie = dec.decorate(cookie)
        rcalc, y = self._calc(nosymmetry(cookie))

        if numpy.isnan(y).any():
            y = numpy.zeros_like(r)
        else:
            y = numpy.interp(r, rcalc, y)
        return y

# End class CookiePDFGenerator

from diffpy.srfit.pdf import PDFContribution as BasePDFContribution
class PDFContribution(BasePDFContribution):

    def addStructure(self, name, stru, periodic = True):
        # Based on periodic, create the proper generator.
        if periodic:
            from diffpy.srfit.pdf.pdfgenerator import PDFGenerator
            gen = PDFGenerator(name)
        else:
            gen = CookiePDFGenerator(name)

        # Set up the generator
        gen.setStructure(stru, "phase", periodic)
        self._setupGenerator(gen)

        return gen.phase

    def addPhase(self, name, parset, periodic = True):
        # Based on periodic, create the proper generator.
        if periodic:
            from diffpy.srfit.pdf.pdfgenerator import PDFGenerator
            gen = PDFGenerator(name)
        else:
            from diffpy.srfit.pdf.debyepdfgenerator import DebyePDFGenerator
            gen = CookiePDFGenerator(name)

        # Set up the generator
        gen.setPhase(parset, periodic)
        self._setupGenerator(gen)

        return gen.phase



def _euler(phi, theta, psi):
    """Euler rotation of a point in the world frame in Z-X-Z formalism """
    cph = math.cos(phi)
    cth = math.cos(theta)
    cps = math.cos(psi)
    sph = math.sin(phi)
    sth = math.sin(theta)
    sps = math.sin(psi)

    m = numpy.array([
      [cps * cph - cth * sph * sps, cps * sph + cth * cph * sps, sps * sth],
      [-sps * cph - cth * sph * cps, -sps * sph + cth * cph * cps, cps * sth],
      [sth * sph, -sth * cph, cth]
      ], dtype = float)

    return m

def _test1():
    pi = numpy.pi
    # Note that we are using a row vector, so we either multiply on the left,
    # or transpose the matrix and multiply on the right.
    a = numpy.array([1, 0, 0], dtype = float)
    m1 = _euler(pi/2, 0, 0)
    a1 = numpy.dot(a, m1)
    assert( numpy.allclose(a1, [0, 1, 0]) )
    a2 = numpy.dot(m1.T, a)
    assert( numpy.allclose(a2, [0, 1, 0]) )

    a = [0, 0, 1]
    m2 = _euler(0, pi/2, 0)
    a1 = numpy.dot(a, m2)
    assert( numpy.allclose(a1, [0, -1, 0]) )

    a = [1, 2, 3]
    m2 = _euler(0, pi/2, pi/2)
    a1 = numpy.dot(a, m2)
    assert( numpy.allclose(a1, [-2, -3, 1]) )

    a = [1, 2, 3]
    m2 = _euler(pi/2, pi/2, 0)
    a1 = numpy.dot(a, m2)
    assert( numpy.allclose(a1, [3, 1, 2]) )

    return

def _test2():
    pi = numpy.pi
    cc = CookieCutter()
    # Represents a cookie cutter where the y-axis is along the z-direction of
    # the world frame.
    cc.orient = numpy.array([0, pi/2, 0])
    # This point is along the z-direction in the world frame.
    a = numpy.array([0, 0, 1], dtype = float)
    # In the frame of the cookie cutter, it will be along the y-axis
    a1 = cc.putInFrame(a)
    assert( numpy.allclose(a1, [0, 1, 0]) )
    # In the world frame, this point is along the  y-axis.
    a = numpy.array([0, 1, 0], dtype = float)
    # It will be along the -z-axis of the cookie cutter frame
    a1 = cc.putInFrame(a)
    assert( numpy.allclose(a1, [0, 0, -1]) )

    # Put the x-axis of the cookie cutter in the [1/2**0.5, 1/2**0.5, 0]
    # direction
    cc.phi = pi/4
    cc.theta = cc.psi = 0
    # A point along the y-axis of the world frame
    a = numpy.array([0, 1, 0], dtype = float)
    # Will be along the [1/2**0.5, 1/2**0.5, 0] direction of the cookie cutter
    # frame.
    a1 = cc.putInFrame(a)
    assert( numpy.allclose(a1, [1/2**0.5, 1/2**0.5, 0]) )

    return

def _test3():
    cc = SphericalCookieCutter()
    cc.radius = 1
    a = [0, 0, 0]
    assert( cc.isIn(a) )
    a = [1, 0, 0]
    assert( cc.isIn(a) )
    a = [0, 1, 0]
    assert( cc.isIn(a) )
    a = [0, 0, 1]
    assert( cc.isIn(a) )
    a = [1/3**0.5]*3
    assert( cc.isIn(a) )
    a = [0, 0, 1.1]
    assert( not cc.isIn(a) )

    cc.origin = numpy.array([0, 0, 1])
    a = [0, 0, 0]
    assert( cc.isIn(cc.putInFrame(a)) )
    a = [1, 0, 0]
    assert( not cc.isIn(cc.putInFrame(a)) )
    a = [0, 1, 0]
    assert( not cc.isIn(cc.putInFrame(a)) )
    a = [0, 0, 1]
    assert( cc.isIn(cc.putInFrame(a)) )
    a = [1/3**0.5]*3
    assert( cc.isIn(cc.putInFrame(a)) )
    a = [0, 0, 1.1]
    assert( cc.isIn(cc.putInFrame(a)) )

    return

def _test4():
    cc = CylindricalCookieCutter()
    stru = Structure(filename = "Bruque.cif")  # cif file to start from
    cc.radius = 20  # specify the morphology of the cylinder, radius/length etc.
    cc.height = 8

    gen = CookiePDFGenerator("G") #calculate PDF for the cookie using DebyePDFGenerator~
    gen.setStructure(stru)
    gen.setQmax(22) # Qmax, Qmin for computing PDF
    gen.setQmin(1)

    gen.setCookieCutter(cc)
    cookie = cc.cut(stru)  # this would return the cutted cookie in xyz format!!
    #cookie.write("cylinder_rh_5_2.stru", "pdffit") # write into xyz file in PDFfit/PDFgui
    format
    cookie.write("cylinder_20_8_bruque.xyz", "xyz") # xyz format

    r = numpy.arange(0, 40, 0.01)
    gr = gen(r)
    numpy.savetxt("cylinder_20_8_bruque.txt", zip(r, gr))
    import pylab
    pylab.plot(r, gr)
    pylab.show()
    return

def _test5():
    cc = RohomboidCookieCutter()
    stru = Structure(filename = "Phenyl.cif")
    cc.ap = 30
    cc.bp = 30
    cc.cp = 8
    cc.alphap = cc.gammap = 90
    cc.betap = 101.333
    cc.z = -4 #
    print cc.maxrads

    gen = CookiePDFGenerator("G")
    gen.setStructure(stru)
    gen.setQmax(22)
    gen.setQmin(1)
    gen.setCookieCutter(cc)
    cookie = cc.cut(stru)
    #cookie.write("temp.stru", "pdffit")
    cookie.write("rohom_30_30_8_100_phenyl.xyz", "xyz")
    r = numpy.arange(0, 30, 0.01)
    gr = gen(r)

    import pylab
    pylab.plot(r, gr)
    pylab.show()
    return


def _test6():
    cc = SphericalCookieCutter()
    stru = Structure(filename = "CoOO.cif")  # cif file to start from
    cc.radius = 20  # specify the morphology of the cylinder, radius/length etc.
    #cc.height = 2

    gen = CookiePDFGenerator("G") #calculate PDF for the cookie using DebyePDFGenerator~
    gen.setStructure(stru)
    gen.setQmax(20) # Qmax, Qmin for computing PDF
    gen.setQmin(1)
    gen.setCookieCutter(cc)
    cookie = cc.cut(stru)  # this would return the cutted cookie in xyz format!!
    #cookie.write("cylinder_rh_5_2.stru", "pdffit") # write into xyz file in PDFfit/PDFgui
    format
    cookie.write("spherical_r_20.xyz", "xyz") # xyz format

    r = numpy.arange(0, 30, 0.1)
    gr = gen(r)
    numpy.savetxt("sphere_r_20.txt", zip(r, gr))
    import pylab
    pylab.plot(r, gr)
    pylab.show()
    return


def _alltests():
    _test1()
    _test2()
    _test3()
    return

if __name__ == "__main__":
    _alltests()


