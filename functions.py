#!/usr/bin/env python3
# @author patrickdiehl@lsu.edu
# @author serge.prudhomme@polymtl.ca
# @date 10/05/2019
import numpy as np
import sys 
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.size"] = 12
matplotlib.rcParams['text.latex.preamble'] = [
    r'\usepackage{xfrac}']

def u(x,eps):
    return x-((np.exp(-(1-x)/eps)-np.exp(-1/eps))/(1-np.exp(-1/eps)))

x = np.linspace(0,1,100)

plt.plot(x,u(x,0.1),label=r"$\epsilon=0.1$",color="black",linestyle="-")
plt.plot(x,u(x,0.01),label=r"$\epsilon=0.01$",color="black",linestyle=":")
plt.xlabel(r"$x$")
plt.ylabel(r"$u(x)$")
plt.legend()
plt.grid()

plt.savefig("functions.pdf",bbox_inches='tight')

plt.clf()

def kx(x):
    k=1
    return k/(4*(x/delta)-(3-2*np.log(x/delta))*x*x/delta/delta)

def kx2(x):
    k=1
    return k/(4*((1-x)/delta)-(3-2*np.log((1-x)/delta))*(1-x)*(1-x)/delta/delta)

n = np.power(2,4)
h=1./n
delta=2*h

xp = np.linspace(0.01,2*h,2)
plt.plot(xp,kx(xp),label="$\overline{\kappa}(x),\, x \in ]0,\delta]$",c="black")
xp = np.linspace(1-2*h,0.99,2)
plt.plot(xp,kx2(xp),label="$\overline{\kappa}(x),\, x \in [1-\delta,1]$",c="black",ls="-.")
plt.grid()
plt.xlabel("$x$")
plt.ylabel(r"$\overline{\kappa}(x)$")
plt.legend()
plt.title("Correction Factor for the EDM method")
plt.savefig("EDM-correction.pdf",bbox_inches='tight')

