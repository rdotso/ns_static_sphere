import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt

import decimal
from decimal import Decimal
import math
import csv

from typing import List, Tuple

decimal.getcontext().prec = 32

###
#Universal or helpful Constants in cgs
c = Decimal(2.99792458*(10**10))
G = Decimal(6.6743*(10**-8))
M_sun = Decimal(1.989*(10**33))
###
  

def get_EoS() -> npt.NDArray:
    """ Returns ndarray[Decimal] """
    with open('sly.dat') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        EoS: List[List[Decimal]] = [[Decimal(row[0]), Decimal(row[1])*G/c**2, Decimal(row[2])*G/c**2] 
                                    for row in reader]
    return np.asarray(EoS)


def buildNS(P0: float, r0: float, step: int) -> Tuple[List[Decimal], List[Decimal], List[Decimal]]:
    EoS = get_EoS()
    
    P_limit = Decimal(P0) * Decimal(10)**Decimal(-11)
    r_arr = [Decimal(r0)]
    P_arr = [Decimal(P0)]
    M_arr = [dM(Decimal(r0), Decimal(P0), EoS)]
    while P_arr[-1] >= P_limit:
        tmp = iterate(r_arr[-1], P_arr[-1], M_arr[-1], EoS, step)
        r_arr.append(tmp[0])
        P_arr.append(tmp[1])
        M_arr.append(tmp[2])
    
    print(len(r_arr))
    #print(P)
    #print(M)
    return r_arr, P_arr, M_arr
    
    
def iterate(r: Decimal, P: Decimal, M: Decimal, EoS: npt.NDArray, step: int) -> Tuple[Decimal, Decimal, Decimal]:
    #print("Iterate")
    #print(r)
    if r % (step*(10**3)) == 0:
        print(f'r: {int(r)}, p: {float(P)}, M: {float(M)}')
    r_new = r + step
    #print("r_new[-1]: " + str(r_new[-1]))
    P_new = P + (step * dP(r_new, P, M, EoS))
    #print("P[-1]: "+ str(P[-1]))
    M_new = M + (step * dM(r_new, P, EoS))
    return (r_new, P_new, M_new)


def dP(r: Decimal, P: Decimal, M: Decimal, EoS: npt.NDArray) -> Decimal:
    #print("r: " + str(r))
    #print("M: " + str(M))
    #print("dP: " + str((((4*np.pi*P*(r**3)+M)*(rhoFunc(P,EoS)+(P)))/((2*M-r)*r))))
    return (Decimal(4) * Decimal(math.pi) * P * (r**Decimal(3))+M) * (rhoFunc(P, EoS) + P) / ((2*M-r)*r)

def dM(r: Decimal, P: Decimal, EoS: npt.NDArray) -> Decimal:
    return Decimal(4) * Decimal(math.pi) * (r**Decimal(2)) * rhoFunc(P, EoS)

def rhoFunc(P: Decimal, EoS: npt.NDArray) -> Decimal:
    #lP is the index of the greatest pressure listed in the EoS less than p
    lP = np.argwhere(EoS[:,1] < P)[-1]
    #print(lP)
    return EoS[lP,2] + ((EoS[lP+1,2] - EoS[lP,2]) / (EoS[lP+1,1] - EoS[lP,1]) * (P - EoS[lP,1]))


def Mass_Radius(p_min: float, p_max: float, number: int) -> Tuple[List[Decimal], List[Decimal]]:
    Radii = []
    Masses = []
    Pressures = np.logspace(math.log10(p_min),math.log10(p_max),num=number)
    for i in Pressures:
        print(f"Pressure: {str(i)}") 
        NS = buildNS(i, 10**2, 10**2)
        Radii.append(NS[0][-1]/(10**5))
        Masses.append(NS[2][-1] * (((Decimal(c)**Decimal(2)) / Decimal(G) ) / Decimal(M_sun)))
    return Radii, Masses


#Runs:
def run_NS():
    NS = np.asarray(buildNS(5*(10**-16), 10**2, 10**2), dtype=object)
    plt.loglog(NS[0],NS[1])
    plt.savefig('loglog')

def run_Mass_Radius():
    Radii, Masses =  Mass_Radius(2.5*(10**-16), 1.25*(10**-13), 100)
    plt.plot(np.asarray(Radii, dtype=object), np.asarray(Masses, dtype=object))
    plt.xlabel("Radius (Km)")
    plt.ylabel("Mass (M☉)")
    plt.savefig('masses')

#run_NS()
run_Mass_Radius()