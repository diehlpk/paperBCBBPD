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

eps=0.1

#############################################################################
# Solve the system
#############################################################################

def solve(M,f):
    return np.linalg.solve(M,f)


#############################################################################
# Exact solution 
#############################################################################

def exactSolution(x):
    return x-((np.exp(-(1-x)/eps)-np.exp(-1/eps))/(1-np.exp(-1/eps)))

#############################################################################
# Loading
#############################################################################

def force(x):
    return np.exp((x-1)/eps)/(eps*eps*(1-np.exp(-1/eps)))

#############################################################################
# Assemble the stiffness matrix for the varibale horizon model (VHM)
#############################################################################

def VHM(n,h):
        
    MVHM = np.zeros([n,n])

    MVHM[0][0] = 1.
    MVHM[1][0] = -8.
    MVHM[1][1] = 16.
    MVHM[1][2] = -8.

    for i in range(2,n-2):
        MVHM[i][i-2] = -1.
        MVHM[i][i-1] = -4.
        MVHM[i][i] = 10.
        MVHM[i][i+1] = -4.
        MVHM[i][i+2] = -1.


    MVHM[n-2][n-1] = -8.
    MVHM[n-2][n-2] = 16.
    MVHM[n-2][n-3] = -8.

    MVHM[n-1][n-1] = 1
    
    MVHM *= 1./(8.*h*h)
    
    return  MVHM

#############################################################################
#Computation
#############################################################################
print("n,h,VHM")
markers = ['s','o','x','.']

for i in range(2,6):
    n = np.power(2,i)
    h = 1./n
    nodes = n+1
    
    x = np.linspace(0,1.,nodes)
    load = force(x) 
    load[len(x)-1] = 0


    u=np.linalg.solve(VHM(nodes,h),load)
    print(str(n)+","+str(h)+","+str(max(abs((u[1:len(u)-2]-exactSolution(x)[1:len(u)-2])/exactSolution(x)[1:len(u)-2]))))
    plt.scatter(x,u-exactSolution(x),label="$\delta$="+str(2*h), marker=markers[i-2], c="black")


plt.title(r"Example with $\epsilon=$"+str(eps)+" Solution using VHM")
plt.xlabel("x")
plt.ylabel('Error in displacement')
plt.xscale('linear')
plt.yscale('linear')
plt.grid()
plt.legend()
plt.savefig("VHM-Error-eps-"+str(eps)+"Solution.pdf",bbox_inches='tight')