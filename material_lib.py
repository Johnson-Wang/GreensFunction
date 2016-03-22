__author__ = 'xwangan'
from scipy.constants import hbar, N_A, eV
import numpy as np
import scipy as sp
epsilon=1e-4

def green_membrane_couple(q, omega, mat1, mat2, mat3, direc=0):
    "q in 1e9 and omega in THz unit in 1 e-24"
    green1 = mat1.get_green_function(q, omega, direc=direc)
    green3 = mat3.get_green_function(q, omega, direc=direc)
    K1 = mat2.bindfc[mat1.symbol][direc]
    K3 = mat2.bindfc[mat3.symbol][direc]
    PI1 = K1 / (1 - K1 * green1)
    PI3 = K3 / (1 - K3 * green3)
    if direc == 0:
        return 1 / (mat2.rho * omega ** 2 - mat2.kappa * q ** 4 - PI1 - PI3)
    elif direc == 1:
        return 1 / (mat2.rho * omega ** 2 - (mat2.lamb + 2 * mat2.mu) * q ** 2 - PI1 - PI3)
    elif direc == 2:
        return 1 / (mat2.rho * omega ** 2 - mat2.mu * q ** 2  - PI1 - PI3)

class Material3D():
    def __init__(self,
                 a=None, # lattice constant
                 rho=None, # 3D density (g/cm^3)
                 is_isotropic=False, # if True, only c11 and c33 are necessary
                 c11=None, #  elastic constants (GPa)
                 c12=None,
                 c13=None,
                 c33=None,
                 c44=None,
                 symbol=None):
        self.a = a
        self.rho = rho / 1e6 # convert unit to 1e9 kg/m^3
        self.is_isotropic = is_isotropic
        self.c11 = c11
        self.c12 = c12
        self.c13 = c13
        self.c33 = c33
        self.c44 = c44
        self.symbol=symbol
        if is_isotropic:
            if self.c33 is None:
                self.c33 = c11
            if self.c12 is None:
                self.c12 = c11 - 2 * c44
            if self.c13 is None:
                self.c13 = self.c12
        self.c11 /= 1e6; self.c33 /= 1e6; self.c44 /= 1e6; self.c12 /= 1e6; self.c13 /= 1e6
        #convert unit of elastic constants to 1e15 Pa

    def get_green_function(self, qpoint, omega, direc=0):
        "q in nm^-1 and omega in THz unit in 1e-24"
        c11, c12, c13, c33, c44 = self.c11, self.c12, self.c13, self.c33, self.c44
        rho1 = self.rho
        B = c11/c44*(omega**2*self.rho/c11-qpoint**2)+c44/c33*(omega**2*self.rho/c44-qpoint**2)+\
            (c13+c44)**2/(c33*c44)*qpoint**2 # 10^18
        c = c11/c33*(omega**2*self.rho/c11-qpoint**2)*(omega**2*self.rho/c44-qpoint**2) # 10^36
        p1 = -sp.sqrt(0.5*B+0.5*sp.sqrt(B**2-4*c)+1j*epsilon) #10 ^ 9
        p2 = - sp.sqrt(0.5*B-0.5*sp.sqrt(B**2-4*c)+1j*epsilon)  # 10 ^ 9
        p3 = - sp.sqrt(omega**2*self.rho/c44-(c11-c12)/2/c44*qpoint**2+1j*epsilon)
        f1 = (c13+c44)*qpoint*p1/(self.rho*omega**2-c11*qpoint**2-c44*p1**2)  # 1
        f2 = (c13+c44)*qpoint*p2/(self.rho*omega**2-c11*qpoint**2-c44*p2**2) # 1
        M=c44*(p1*f1+qpoint)*(c13*qpoint*f2+c33*p2)-c44*(p2*f2+qpoint)*(c13*qpoint*f1+c33*p1) #10 ^ 48
        if direc==0:
            # return sp.where(np.abs(M) == 0, 1/epsilon, 1j*c44*(f1*p1-f2*p2) / M)
            return 1j*c44*(f1*p1-f2*p2) / M
        elif direc==2:
            return 1j / (c44 / p3)
        elif direc==1:
            return 1j*c33*(f1*p2-f2*p1) / M

class Material2D():
    def __init__(self,
                 a=None, # lattice constants (nm)
                 rho=None, # 2D density (kg/m^2)
                 mu=None,  # lame constant \mu (eV)
                 lamb=None, # lame constant \lambda (eV/A^2)
                 kappa=None, # bending energy (eV/A^2)
                 symbol=None):
        self.a = a
        self.rho = rho
        self.mu = mu * eV * 1e20 / 1e6 #   unit in 1e6
        self.kappa=kappa * eV * 1e12  # kappa is the bend elasticity of graphene
        self.lamb = lamb * 1e20 / 1e6 #  unit in 1e6
        self.symbol=symbol
        self.bindfc = {}

    def set_bindfc(self, symbol, fc1, fc2=None, fc3=None):
        """symbol: the interacting material with graphene; fc1: the interacting force constant resembling c33 in a continuum model  (1e20 N/m^3);
        fc2: the interaction force constant resembling c13 in a continuum model; fc3: c44 in a continuum model.
        """
        if self.symbol == "graphene":
            if fc2 is None:
                fc2 = fc1 / 3.65 * 1.5 # analogous to graphite
            if fc3 is None:
                fc3 = fc1 / 3.65 * 0.035 # analogous to graphite
        self.bindfc[symbol] = np.array([fc1, fc2, fc3], dtype="double") * 1e20 / 1e24 # N/m^-3 (Interaction strength) unit in 1e24

    def get_bindfc(self, symbol):
        return self.bindfc[symbol]

graphene = Material2D(a=0.228, rho=7.612e-7, kappa=1.1, lamb=2., mu=10., symbol="graphene")
# Al2O3
#http://www.mt-berlin.com/frames_cryst/descriptions/sapphire.htm
Al2O3 = Material3D(a=0.4, rho=4, c11=496, c12=164, c13=115, c33=498, c44=148, symbol="Al2O3")
graphene.set_bindfc("Al2O3", 1.1)
#graphite
#Blakslee, O. L., et al. "Elastic constants of compression?annealed pyrolytic graphite." Journal of Applied Physics 41.8 (1970): 3373-3382.
graphite = Material3D(a=0.33, rho=2.266, c11=1060., c12=180., c13=15, c33=36.5, c44=0.35, symbol="graphite")
graphene.set_bindfc("graphite", 2.5)
# SiO2
SiO2 = Material3D(a=0.25, rho=2.2, is_isotropic=True, c11=78., c44=31., symbol="SiO2")
graphene.set_bindfc("SiO2", 1.82)
#hBN
#http://www.ioffe.ru/SVA/NSM/Semicond/BN/mechanic.html
hBN = Material3D(a=0.326, rho=2.28, c11=810., c12=169., c13=0, c33=27, c44=7.7, symbol="hBN")
graphene.set_bindfc("hBN", 1.82)
# Ti
##Tromans, D. Elastic anisotropy of HCP metal crystals and polycrystals. Int. J. Res. Rev. Appl. Sci 6, 462 -483 (2011)
Ti = Material3D(a=0.3, rho=4.5, c11=160, c12=90, c13=66, c33=181.0, c44=46.5, symbol="Ti")
# graphene.set_bindfc("Ti", 0.4)
graphene.set_bindfc("Ti", 0.5)

#Au
#Y. Hiki & A.V. Granato, "Anharmonicity in Noble Metals; Higher Order Elastic Constants." Phys. Rev.  144, 411 (1966)
Au = Material3D(a=0.3, rho=19.32, is_isotropic=True, c11=192.9, c12=163.8, c44=41.5, symbol="Au")
# Au = Material3D(a=0.3, rho=19.32, is_isotropic=True, c11=192.9, c44=41.5, symbol="Au")
graphene.set_bindfc("Au", 0.44)
#Al
# FCC Al crystal
#Kamm G N and Alers G A 1964 J. Appl. Phys. 35 327
Al = Material3D(a=0.4046, rho=2.7, is_isotropic=True, c11=106.9, c12=60.8, c44=28.2, symbol="Al")
graphene.set_bindfc("Al", 0.27)
#Cu
Cu = Material3D(a=0.3597, rho=8.96, c11=169.8, c12=122.6, c13=122.6, c33=169.8, c44=75.3, symbol="Cu")
graphene.set_bindfc("Cu", 1.82)
