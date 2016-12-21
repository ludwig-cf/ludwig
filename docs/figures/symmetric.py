import numpy
import matplotlib.pyplot as plt

def main():

    """
    Schemeatic figure to illustrate symmetric free energy

    Actually semi-schematic as the plots are based on the
    \phi^4 form discussed by e.g., Chaikin and Lubensky
    in the section on Nucleation and spinodal decomposition.
    """

    make_schematic()
    plt.savefig('preview.svg')

def make_schematic():
    """
    There are two panels (ax1 and ax2)
    """

    phi = numpy.linspace(start = -10.0, stop = +10.0)
    f1 = potential(phi, 2.0)
    f2 = potential(phi, 0.0)
    f3 = potential(phi, -1.0)
    f4 = potential(phi, -2.0)

    phistar = numpy.linspace(start = -20.0, stop = +20.0)
    tco = coexistance(phistar)
    tsp = spinodal(phistar)

    fig = plt.figure(figsize = (9.0, 3.0))
    ax1 = plt.subplot(1, 2, 1)

    ax1.spines['left'].set_position('center')
    ax1.spines['right'].set_color('none')
    ax1.spines['bottom'].set_position('center')
    ax1.spines['top'].set_color('none')
    ax1.xaxis.set_ticks([])
    ax1.yaxis.set_ticks([])

    ax1.set_ylim(-25, 10)
    ax1.plot(phi, f1, label = "T > Tc")
    ax1.plot(phi, f2, label = "T = Tc")
    ax1.plot(phi, f3, label = "T < Tc")
    ax1.plot(phi, f4)
    ax1.text(9, -6, "$\phi$", fontsize = 18)
    ax1.text(0.2, 10, "$V(\phi)$", fontsize = 18)

    ax2 = plt.subplot(1, 2, 2)

    ax2.spines['left'].set_position('center')
    ax2.spines['right'].set_color('none')
    ax2.spines['top'].set_color('none')
    ax2.xaxis.set_ticks([])

    ax2.yaxis.set_ticks([])
    ax2.yaxis.set_ticklabels([])

    ax2.set_xlim(-30, +30)
    ax2.set_ylim(-38, +20)

    ax2.plot(phistar, tco)
    ax2.plot(phistar, tsp, "--")
    ax2.text(27, -35, "$\phi$", fontsize = 18)
    ax2.text(1, 19, "$T$", fontsize = 18)
    ax2.text(1, 2, "$T_c$", fontsize = 14)

    ax2.annotate('Coexistence line', xy=(-15, -20), xycoords='data',
                 xytext=(-50, 60), textcoords='offset points',
                 arrowprops=dict(arrowstyle="->")
                )
    ax2.annotate('Spinodal line', xy=(10, -25), xycoords='data',
                 xytext=(10, 60), textcoords='offset points',
                 arrowprops=dict(arrowstyle="->")
                )
    ax2.text(-24, 12, "Mixed phase",
        bbox=dict(boxstyle="round",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8)))
    ax2.text(-11, -34, "Separated phase",
        bbox=dict(boxstyle="round",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8)))



def potential(phi, t, a = 1.0, tc = 1.0):
    """
    V(phi) = (1/2) A(T-T_c)phi^2 + (1/4)Uphi^4
    """
    f = numpy.zeros(phi.size)
    r = a*(t - tc)
    u = a*0.1
    f[:] = 0.5*r*phi[:]**2 + 0.25*u*phi[:]**4
    return f

def coexistance(phi, tc = 1.0):
    """
    Coexistance point - minima of V(phi)
    """
    t = numpy.zeros(phi.size)
    a = 1.0
    u = a*0.1
    t[:] = tc - u*phi[:]**2/a
    return t
    
def spinodal(phi, tc = 1.0):
    """
    Spinodal point - inflection in V(phi)
    """
    t = numpy.zeros(phi.size)
    a = 1.0
    u = a*0.1
    t[:] = tc - 3.0*u*phi[:]**2/a
    return t


if __name__ == "__main__":
    main()

