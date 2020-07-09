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
plt.xlabel("x")
plt.ylabel(r"$u(x)$")
plt.legend()
plt.grid()

plt.savefig("functions.pdf",bbox_inches='tight')

plt.clf()

def kx(x,delta):
    return 1 / (4*(x/delta)-(3-2*np.log(x/delta))*x*x/delta/delta)

def kx2(x,delta):
    return 1 / (4*((1-x)/delta)-(3-2*np.log((1-x)/delta))*(1-x)*(1-x)/delta/delta)

n = np.power(2,4)
h=1./ (n+1) 
delta=2*h

xp = np.array([h,2*h])
print(kx(xp,delta))
plt.scatter(xp,kx(xp,delta),label="$\sfrac{\overline{\kappa}(x)}{\kappa},\, x \in (0,\delta)$",c="black")
xp2 = np.array([1-2*h,1-h])
plt.scatter(xp2,kx2(xp2,delta),label="$\sfrac{\overline{\kappa}(x)}{\kappa},\, x \in (1-\delta,1)$",c="black",marker="s")
plt.grid()
plt.xlabel("x")
plt.ylabel(r"$\sfrac{\overline{\kappa}(x)}{\kappa}$")
plt.legend(loc=8)
plt.xlim(0,1)
plt.ylim(0,1.25)
plt.annotate("$x_1$", (xp[0], 1.05))
plt.annotate("$x_2$", (xp[1], 0.95))
plt.annotate("$x_{n-1}$", (xp2[1]-0.012, 1.05))
plt.annotate("$x_{n-2}$", (xp2[0]-0.012, 0.95))
plt.title("Correction Factor for the EDM I method")
plt.savefig("EDM-correction.pdf",bbox_inches='tight')

