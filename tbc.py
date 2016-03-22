from scipy.constants import k as kb
import matplotlib.pyplot as plt
from material_lib import *
import csv
epsilon=1e-48


#*************************************************#
m1 = Ti
m2 = graphene
m3 = Ti
#*************************************************#

def cv(omega, temp):
    "hbar * omega * \partial(n) / \partial(T) unit: J/K"
    x = hbar / kb * omega / temp * 1e12 # unit : 1
    if x < 1e-5:
        PI = 1.0 / (1 + x + x ** 2)
    elif x > 1e2:
        PI = 0.
    else:
        PI = x / (np.exp(x) - 1)
    return kb * PI * (PI + x)


ts = np.linspace(50, 350, num=50)
omegas = np.linspace(0.03, 100, num=200)
dPIdT = np.zeros_like(omegas)
directions=[0]
trans_fG = np.zeros((len(omegas), len(directions)), dtype="double")
hs = np.zeros((len(ts), len(directions)), dtype="double")
for i, omega in enumerate(omegas):
    for j, direct in enumerate(directions):
        K1 = m2.bindfc[m1.symbol][direct]
        K3 = m2.bindfc[m3.symbol][direct]
        def transmission_fG(q):
            xi1 =  m1.get_green_function(q, omega, direct)
            xi2 = green_membrane_couple(q, omega, m1, m2, m3, direct)
            xi3 = m3.get_green_function(q, omega, direct)
            Gamma1 = - 2 * K1 ** 2 * np.imag(xi1 / (1 - K1 * xi1))
            Gamma3 = - 2 * K3 ** 2 * np.imag(xi3 / (1 - K3 * xi3))
            pp = Gamma1 * Gamma3 * np.abs(xi2) ** 2
            return pp * 2 * np.pi * q # 2 * pi * q comes from the double integral
        qs = np.linspace(1e-3, np.pi / max(m1.a, m3.a, m3.a), 4000)
        trans_fG[i,j] = np.trapz(transmission_fG(qs), qs)


for i, T in enumerate(ts):
    dPIdT[:] = 0
    for j, omega in enumerate(omegas):
        dPIdT[j] = cv(omega, T)
    for k, direc in enumerate(directions):
        hs[i, k] = np.trapz(dPIdT * trans_fG[:,k], x=omegas) * 1e12 * 1e18 / (2 * np.pi) ** 3

omegas /= 2 * np.pi
trans_fG /= np.pi * (np.pi / max(m1.a, m3.a, m3.a)) ** 2
plt.plot(omegas, trans_fG)


with open("T-F-%s-%s-%s.csv"%(m1.symbol, m2.symbol, m3.symbol), "wb") as f:
    writer = csv.writer(f)
    writer.writerow(["Frequency (THz)", 'Transmission'])
    writer.writerows(np.hstack((omegas[:, np.newaxis], trans_fG)))

plt.figure()
plt.plot(ts,hs * 2)
for t, h in zip(ts, hs):
    print t, "%e " * len(directions) % tuple(2 * h)
plt.show()