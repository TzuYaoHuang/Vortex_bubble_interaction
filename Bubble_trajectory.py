import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from scipy.linalg import lu_factor, lu_solve
from scipy import interpolate, stats, integrate
import time
import datetime
import os
import sys
from scipy.special import struve, jv

# based on the MatLab file from SwarajNanda
# following MKS system

mpl.use('Qt5Agg')

def LambOssenVortex(x, paramLO):

    viscousRadius, cavitateRadius, sigmaLO, betaLO, gammaInf = paramLO

    r = np.linalg.norm(x)
    theta = np.arctan2(x[1], x[0])

    uTheta = gammaInf / (2*np.pi*r) * \
        (1 - betaLO * np.exp(-sigmaLO * r**2/viscousRadius**2))

    omegaZ = gammaInf / np.pi * sigmaLO/(cavitateRadius**2 + sigmaLO * viscousRadius**2) * \
        np.exp(-sigmaLO * (r**2 - cavitateRadius**2)/viscousRadius**2)

    DurDt = -uTheta**2/r

    u = np.array((-uTheta*np.sin(theta), uTheta*np.cos(theta), 0))
    omega = np.array((0, 0, omegaZ))
    # http://pleasemakeanote.blogspot.com/2008/08/derivation-of-navier-stokes-equations_17.html
    DuDt = np.array((DurDt*np.cos(theta), DurDt*np.sin(theta), 0))

    return u, omega, DuDt


def BubbleMotion(y, t, paramLO, paramOther):
    x = y[0:3]
    ub = y[3:6]

    u, omega, DuDt = LambOssenVortex(x, paramLO)
    iniBubbleRadius, liquidViscosity, gasDensity, liquidDensity = paramOther
    Cam = 0.5
    g = np.array((0, 0, -9.81))

    rBubble = iniBubbleRadius
    VBubble = 4/3 * np.pi * rBubble**3
    ABubble = 4/3 * np.pi * rBubble**3

    mEff = (gasDensity + Cam*liquidDensity) * VBubble
    ReBubble = 2*rBubble*np.linalg.norm(u-ub)/liquidViscosity
    alpha = np.linalg.norm(omega)*rBubble/np.linalg.norm(u-ub)

    Cl = 4*alpha/3
    Cd = (24/ReBubble) * (1 + 0.197*ReBubble**0.63 + 2.6e-4*ReBubble**1.38)

    FPres = (1+Cam) * liquidDensity * VBubble * DuDt
    FGrav = (gasDensity - liquidDensity) * VBubble * g
    FDrag = 0.5 * Cd * liquidDensity * ABubble * (u-ub) * np.linalg.norm(u-ub)
    FLift = Cl * liquidDensity * VBubble * np.cross(u-ub, omega)

    FTotal = FPres + FGrav + FDrag + FLift

    dydt = np.hstack((ub, FTotal/mEff))

    return dydt


viscousRadius = 2.0e-3
cavitateRadius = 0
maxTangenVel = 3          # always at viscous radius

iniBubbleRadius = 200e-6
liquidViscosity = 1e-6
gasDensity = 1.2
liquidDensity = 1000

paramOther = (iniBubbleRadius, liquidViscosity, gasDensity, liquidDensity)

sigmaLO = 1.256431
betaLO = \
    cavitateRadius**2 / (cavitateRadius**2 + sigmaLO * viscousRadius**2) * \
    np.exp(sigmaLO * (cavitateRadius/viscousRadius)**2)
gammaInf = maxTangenVel * 2 * np.pi * viscousRadius / \
    (1 - cavitateRadius**2 / (cavitateRadius**2 + sigmaLO * viscousRadius**2) *
     np.exp(-sigmaLO * (viscousRadius**2 - cavitateRadius**2)/viscousRadius**2))

paramLO = (viscousRadius, cavitateRadius, sigmaLO, betaLO, gammaInf)

dt = 5e-5
N = 1000000
timeSpan = dt * np.linspace(0, N*dt, N+1, endpoint=True)
y0 = np.array((2e-3, 0, 0, 0, 2.9, 0))

sol = np.zeros((N+1,6))
sol[0,:] = y0

runWhich = False

if runWhich:
    for i in range(1,N+1):
        x = sol[i-1,0:3]
        ub = sol[i-1,3:6]

        u, omega, DuDt = LambOssenVortex(x, paramLO)
        # iniBubbleRadius, liquidViscosity, gasDensity, liquidDensity = paramOther
        Cam = 0.5
        g = np.array((0, 0, -9.81))

        rBubble = iniBubbleRadius
        VBubble = 4/3 * np.pi * rBubble**3
        ABubble = 4/3 * np.pi * rBubble**3

        mEff = (gasDensity + Cam*liquidDensity) * VBubble
        ReBubble = 2*rBubble*np.linalg.norm(u-ub)/liquidViscosity
        alpha = np.linalg.norm(omega)*rBubble/np.linalg.norm(u-ub)

        Cl = 4*alpha/3
        Cd = (24/ReBubble) * (1 + 0.197*ReBubble**0.63 + 2.6e-4*ReBubble**1.38)

        FPres = (1+Cam) * liquidDensity * VBubble * DuDt*0
        FGrav = (gasDensity - liquidDensity) * VBubble * g
        FDrag = 0.5 * Cd * liquidDensity * ABubble * (u-ub) * np.linalg.norm(u-ub)
        FLift = Cl * liquidDensity * VBubble * np.cross(u-ub, omega)

        FTotal = FPres + FGrav + FDrag + FLift

        dydt = np.hstack((ub, FTotal/mEff))

        sol[i,:] = sol[i-1,:] + dydt*dt       

if not runWhich:
    sol = integrate.odeint(BubbleMotion, y0, timeSpan, args=(paramLO, paramOther))

aaaaaa = 0    # could be a break point

plt.plot(sol[:,0], sol[:,1])
plt.show()


