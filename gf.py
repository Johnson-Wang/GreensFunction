__author__ = 'xwangan'
from scipy.constants import k as kb
import matplotlib.pyplot as plt
from material_lib import *
TradtomeV = 1/(2*np.pi) * 4.13567
#*************************************************#
m1 = SiO2
m2 = graphene
m3 = SiO2
#*************************************************#

def dfdT(omega, temp):
    " \partial(n) / \partial(T) unit: J/K"
    x = hbar / kb * omega / temp * 1e12 # unit : 1
    if x < 1e-5:
        PI = 1.0 / (1 + x + x ** 2)
    elif x > 1e2:
        PI = 0.
    else:
        PI = x / (np.exp(x) - 1)
    return (PI + x) / temp

def write_data_to_tecplot_2D(X, Y, Data, xtitle=None, ytitle=None, titles=None, filename=None):
    """Write the data into a file in tecplot format.
    Args:
        X: (numpy.array)
            The x axis values (either broadcasted or not)
        Y: (numpy.array)
            The y axis values (either broadcast or not)
        xtitle: str
            The title for X data
        ytitle: str
            The title for Y data
        Data: (numpy.array)
            The data to ouput
        titles: list or str
            The title for each data to be exported
        filename: str
            The name of the output (not including ".dat")
    """
    if len(X.shape) != 1:
        X = X[0]
    if len(Y.shape) != 1:
        Y = Y[:,0]
    if xtitle == None:
        xtitle = "X"
    if ytitle == None:
        ytitle = "Y"
    if len(Data.shape) == 2:
        Data = Data[np.newaxis]
    assert Data.shape[1:] == (len(Y), len(X))
    if titles is not None:
        if type(titles) == type(""):
            titles = [titles]
        else:
            assert len(titles) == len(Data)
    else:
        titles = ["V%d"%(i+1) for i in range(len(Data))]
    if filename is None:
        filename = "export"
    export_filename = filename + ".dat"
    with open(export_filename, "w") as f:
        f.write("title = \" %s\"\n"%filename)
        f.write("variables =\"%s\", \"%s\"" %(xtitle, ytitle))
        for i, ttl in enumerate(titles):
            f.write(",\"%s\""%ttl)
        f.write("\n")
        f.write("zone I=%d, J=%d, F=POINT\n" %(len(X), len(Y)))
        for j, yvalue in enumerate(Y):
            for i, xvalue in enumerate(X):
                f.write("%15.10f %15.10f"%(xvalue, yvalue))
                f.write(" %15.10f"* len(Data) % tuple(Data[:, j, i]))
                f.write("\n")
    return

# cR = brenth(xi_substrate_denominator, 0.5 * cT, 0.97 * cT)
# print derivative(xi_substrate_denominator_complex, cR, dx=1e-6)
qs = np.linspace(0.1,  np.pi / m2.a, num=300)
# qs = np.linspace(1e-3, min(omega / cT, np.pi / a1, np.pi / a0), 400)
# omega_fR = np.zeros_like(qs)
# omega_sR = np.zeros_like(qs) /  (2 * np.pi)
omega_f = qs * qs * np.sqrt(m2.kappa / m2.rho)
omegas = np.linspace(0.03, omega_f.max(), num=400) # omega in Trad
# omega_fG = np.sqrt( omega_f ** 2 + Kz / rho0)
# omega_R = qs * cR / (2 * np.pi)
# omega_sT = qs * cT
mode = "dos"
directions=[0] # 0: zz; 1: yy; 2: xx
# if mode == "thermal expansion":
#     unit = 1e9 ** 4 * 1e12 * 1e-24 * hbar
#     # omegas = np.linspace(0.01,200, num=200)
#     omegas = np.linspace(1, 100, num=200)
#     # qs_fR = get_fR_wave_vector(omegas)
#     te = np.zeros_like(omegas)
#     for i, o in enumerate(omegas):
#         q_fR = qs_fR[i]
#         def denominator(q):
#             return green_membrane_couple_denominator(q, o)
#         deriv = derivative(denominator, q_fR, 1e-6)
#         def integral(q):
#             return q ** 3 * np.imag(green_membrane_couple_single(q, o))
#         qs = np.linspace(0.0001, min(o / cT, np.pi / a1), num=600)
#         te[i] = np.trapz(integral(qs), qs)
#         te[i] += np.pi * q_fR ** 3 * green_membrane_couple_numerator(q_fR, o) / deriv
#         # te[i] = quad(integral, 0.001, min(o/cT, np.pi/a1))[0]
#         te[i] *= dfdT(o, temp=300) / (2 * np.pi) ** 2 * unit
#     print np.trapz(te, omegas)

if mode == "transmission":
    # plt.plot(qs, omega_fG/ (2 * np.pi), color="w", linewidth=1)
    # omega_sT = qs * np.sqrt(m1.c11 / m1.rho)
    # plt.plot(qs, omega_sT / (2 * np.pi), color="c", linewidth=1)
    # plt.plot(qs, qs * cL / (2 * np.pi), color="c", linewidth=1)
    # plt.ylim((0, omegas.max()/ (2 * np.pi)))
    qs, omegas = np.meshgrid(qs, omegas)
    pp = np.zeros_like(qs)
    for d in directions:
        K1 = m2.bindfc[m1.symbol][d]
        xi1 = m1.get_green_function(qs, omegas, d)
        K3 = m2.bindfc[m3.symbol][d]
        xi3 = m3.get_green_function(qs, omegas, d)
        xi2 = green_membrane_couple(qs, omegas, m1, m2, m3, d)
        Gamma1 = - 2 * K1 ** 2 * np.imag(xi1 / (1 - K1 * xi1))
        Gamma3 = - 2 * K3 ** 2 * np.imag(xi3 / (1 - K3 * xi3))
        pp += Gamma1 * Gamma3 * np.abs(xi2) ** 2
    omegas /= 2 * np.pi # convert from Trad to THz
    filename = "Transmission-%s-%s-%s"%(m1.symbol, m2.symbol, m3.symbol)
    write_data_to_tecplot_2D(X=qs,
                             Y=omegas,
                             Data=pp,
                             xtitle="Wavevector(nm^-1)",
                             ytitle="Frequency (THz)", titles="Transmission",
                             filename=filename)
    plt.pcolor(qs, omegas, pp)
    plt.colorbar()
    plt.show()


if mode == "dos":
    # plt.plot(qs, omega_fR * TradtomeV)
    # omega_sT = qs * np.sqrt(m1.c44 / m1.rho)
    # plt.plot(qs, omega_sT *TradtomeV, color="c", linewidth=1)
    qs, omegas = np.meshgrid(qs, omegas)
    dos = np.zeros_like(qs)

    for d in directions:
        K1 = m2.bindfc[m1.symbol][d]; K3 = m2.bindfc[m3.symbol][d]
        omega0 = np.sqrt(K1/m2.rho + K3 / m2.rho)
        gamma0 = K1 / np.sqrt(m1.c33 * m1.rho) + K3 / np.sqrt(m3.c33 * m3.rho)
        unit = np.pi * omega0 ** 2  / (2 * gamma0)
        xi2 = green_membrane_couple(qs, omegas, m1, m2, m3, d)
        dos += np.imag(xi2) * (-2 * m2.rho * omegas / np.pi)
    filename = "dos-%s-%s-%s"\
                                      %(m1.symbol, m2.symbol, m3.symbol)
    omegas *= TradtomeV
    write_data_to_tecplot_2D(X=qs,
                             Y=omegas,
                             Data=dos,
                             xtitle="Wavevector(nm^-1)",
                             ytitle="Frequency (meV)", titles="DOS",
                             filename=filename)
    plt.pcolor(qs, omegas, dos, vmax=0.1)
    plt.colorbar()
    plt.show()