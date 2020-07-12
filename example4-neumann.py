#!/usr/bin/env python3
# @author patrickdiehl@lsu.edu
# @author serge.prudhomme@polymtl.ca
# @date 10/05/2019
import numpy as np
import sys 
import matplotlib.pyplot as plt
import matplotlib
from cycler import cycler

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
# Boundary value
#############################################################################

def boundary(x):
    return 1-np.exp((x-1)/eps)/(eps*(1-np.exp(-1/eps)))

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

    MVHM[n-1][n-1] = 12.*h
    MVHM[n-1][n-2] = -16.*h
    MVHM[n-1][n-3] = 4.*h
    
    MVHM *= 1./(8.*h*h)
    
    return  MVHM

#############################################################################
# Assemble the stiffness matrix for the local linear elasticity model (LLEM)
#############################################################################

def LLEM(n,h):
    MLLEM = np.zeros([n,n])

    MLLEM[0][0] = 1

    for i in range(1,n-1):
        MLLEM[i][i-1] = -2
        MLLEM[i][i] = 4
        MLLEM[i][i+1] = -2

    MLLEM[n-1][n-1] = 3*h
    MLLEM[n-1][n-2] = -4*h
    MLLEM[n-1][n-3] = h

    MLLEM *= 1./(2.*h*h)
    
    return  MLLEM

#############################################################################
#Computation
#############################################################################
print("n,h,VHM")
markers = ['s','o','x','.']
size= [2,4,8,10]

for i in range(5,10):
    n = np.power(2,i)
    h = 1./n
    nodes = n+1
    
    x = np.linspace(0,1.,nodes)
    load = force(x) 
    load[len(x)-1] = boundary(1)

 
    u=np.linalg.solve(VHM(nodes,h),load)
    uLLEM=np.linalg.solve(LLEM(nodes,h),load)

    print(str(n)+","+str(h)+","+str(max(((u[1:len(u)-2]-exactSolution(x)[1:len(u)-2])/exactSolution(x)[1:len(u)-2])))+","+str(max(((uLLEM[1:len(u)-2]-exactSolution(x)[1:len(u)-2])/exactSolution(x)[1:len(u)-2]))))
    if i > 6:
        plt.plot(x,u-exactSolution(x),label="$\delta=\sfrac{1}{2^{"+str(int(n/2))+"}}$", marker=markers[i-6], c="black",markevery=size[i-6],ls='')


plt.title(r"Example with $\epsilon=$"+str(eps)+" Solution using VHM")
plt.xlabel("x")
plt.ylabel('Error in displacement')
plt.xscale('linear')
plt.yscale('linear')
plt.grid()
plt.legend()
plt.savefig("VHM-Error-eps-"+str(eps)+"-neumann-Solution.pdf",bbox_inches='tight')


plt.cla()
monochrome = (cycler('color', ['k']) * cycler('linestyle',['--',':','-.']))
ax = plt.gca()
ax.set_prop_cycle(monochrome)

for i in [5,6,7]:
    n = np.power(2,i)
    h = 1./n
    nodes = n+1
    
    x = np.linspace(0,1.,nodes)
    load = force(x) 
    load[len(x)-1] = boundary(1)

    u=np.linalg.solve(VHM(nodes,h),load)
    plt.plot(x,u,label="$\delta=\sfrac{1}{2^{"+str(int(n/2))+"}}$")

plt.plot(x,exactSolution(x),label=r"$\underline{u}(x)$",color="black",ls="-")

plt.xlabel("x")
plt.ylabel('Displacement $u$')
plt.xscale('linear')
plt.yscale('linear')
plt.title(r"Convergence Study with $\epsilon=$"+str(eps)+" Solution using VHM")
plt.grid()
plt.legend()
plt.savefig("VHM-convergence-eps-"+str(eps)+"-neumann-Solution.pdf",bbox_inches='tight')


plt.cla()
monochrome = (cycler('color', ['k']) * cycler('linestyle',['--',':','-.']))
ax = plt.gca()
ax.set_prop_cycle(monochrome)

for i in [5,6,7]:
    n = np.power(2,i)
    h = 1./n
    nodes = n+1
    
    x = np.linspace(0,1.,nodes)
    load = force(x) 
    load[len(x)-1] = boundary(1)

    u=np.linalg.solve(LLEM(nodes,h),load)
    plt.plot(x,u,label="$\delta=\sfrac{1}{2^{"+str(int(n/2))+"}}$")

plt.plot(x,exactSolution(x),label=r"$\underline{u}(x)$",color="black",ls="-")

plt.xlabel("x")
plt.ylabel('Displacement $u$')
plt.xscale('linear')
plt.yscale('linear')
plt.title(r"Convergence Study with $\epsilon=$"+str(eps)+" Solution using VHM")
plt.grid()
plt.legend()
plt.savefig("LLEM-convergence-eps-"+str(eps)+"-neumann-Solution.pdf",bbox_inches='tight')