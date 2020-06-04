#!/usr/bin/env python3
# @author patrickdiehl@lsu.edu
# @author serge.prudhomme@polymtl.ca
# @date 10/05/2019
import numpy as np
import sys 
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["text.usetex"] = True
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

plt.savefig("functions.pdf")