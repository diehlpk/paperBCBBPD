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


example = "Quartic"

#############################################################################
# EDM correction for the left-hand side
#############################################################################

def kx(x):
    k=1
    return k/(4*(x/delta)-(3-2*np.log(x/delta))*x*x/delta/delta)

#############################################################################
# EDM correction for the right-hand side
#############################################################################

def kx2(x):
    k=1
    return k/(4*((1-x)/delta)-(3-2*np.log((1-x)/delta))*(1-x)*(1-x)/delta/delta)

#############################################################################
# Solve the system
#############################################################################

def solve(M,f):
    return np.linalg.solve(M,f)

#############################################################################
# Exact solution 
#############################################################################

def exactSolution(x):
    
    if example == "Cubic":
        return x*(3.-x)*(3.+x)/6.
    elif example == "Quartic":
        return x*(16.-x*x*x)/12.
    elif example == "Quadratic":
        return x*(4.-x)/2.
    elif example == "Linear":
        return x
    
    else:
        print("Error: Either provide linear or quadratic")
        sys.exit()
        
    
#############################################################################
# Loading
#############################################################################

def f(x):
    
    if example == "Cubic":
        return x
    elif example == "Quartic":
        return x*x
    elif example == "Quadratic":
        return 1
    elif example == "Linear":
        return 0
    else:
        print("Error: Either provide linear or quadratic")
        sys.exit()


#############################################################################
# Assemble the stiffness matrix for the extended domain model (EMD)
#############################################################################

def EDM(n,h):
    
    MEDM = np.zeros([n,n])

    MEDM[0][0] = 1.

    MEDM[1][0] = -6.
    MEDM[1][1] = 11.
    MEDM[1][2] = -4.
    MEDM[1][3] = -1.
    
    for i in range(2,n-2):
        MEDM[i][i-2] = -1.
        MEDM[i][i-1] = -4.
        MEDM[i][i] = 10.
        MEDM[i][i+1] = -4.
        MEDM[i][i+2] = -1.


    MEDM[n-2][n-1] = -6.
    MEDM[n-2][n-2] = 11.
    MEDM[n-2][n-3] = -4.
    MEDM[n-2][n-4] = -1.

    MEDM[n-1][n-1] = 12.*h
    MEDM[n-1][n-2] = -16.*h
    MEDM[n-1][n-3] = 4.*h

    MEDM *= 1./(8.*h*h)

    return MEDM

#############################################################################
# Assemble the stiffness matrix for the corrected extended domain model (EMD)
#############################################################################

def EDMCORRECTION(n,h):
    
    MEDM = np.zeros([n,n])

    MEDM[0][0] = 1. 

    MEDM[1][0] = -6. * kx(x[1]) / 8 / h / h
    MEDM[1][1] = 11. * kx(x[1]) / 8 / h / h
    MEDM[1][2] = -4. * kx(x[1]) / 8 / h / h
    MEDM[1][3] = -1. * kx(x[1]) / 8 / h / h
    
    MEDM[2][0] = -1. * kx(x[2]) / 8 / h / h
    MEDM[2][1] = -4. * kx(x[2]) / 8 / h / h
    MEDM[2][2] = 10. * kx(x[2])   / 8 / h / h
    MEDM[2][3] = -4. * kx(x[2]) / 8 / h / h
    MEDM[2][4] = -1. * kx(x[2]) / 8 / h / h
    
    
    for i in range(3,n-3):
        MEDM[i][i-2] = -1.  / 8 / h / h
        MEDM[i][i-1] = -4.  / 8 / h / h
        MEDM[i][i] = 10.    / 8 / h / h
        MEDM[i][i+1] = -4. / 8 / h / h
        MEDM[i][i+2] = -1. / 8 / h / h

        
    MEDM[n-3][n-5] = -1. * kx2(x[n-3])  / 8 / h / h
    MEDM[n-3][n-4] = -4. * kx2(x[n-3])  / 8 / h / h
    MEDM[n-3][n-3] = 10. * kx2(x[n-3])  / 8 / h / h
    MEDM[n-3][n-2] = -4. * kx2(x[n-3]) / 8 / h / h
    MEDM[n-3][n-1] = -1. * kx2(x[n-3]) / 8 / h / h
        

    MEDM[n-2][n-1] = -6. * kx2(x[n-2])  / 8 / h / h
    MEDM[n-2][n-2] = 11. * kx2(x[n-2]) / 8 / h / h
    MEDM[n-2][n-3] = -4. * kx2(x[n-2]) / 8 / h / h
    MEDM[n-2][n-4] = -1. * kx2(x[n-2])  / 8 / h / h

    MEDM[n-1][n-1] = 3   / 2 / h 
    MEDM[n-1][n-2] = -4 / 2 / h   
    MEDM[n-1][n-3] = 1  / 2 / h 
    
    return MEDM

#############################################################################
# Assemble the stiffness matrix for the corrected extended domain model (EMD)
#############################################################################

def EDMCORRECTION2(n,h):
    
    factor = 8/7
    
    MEDM = np.zeros([n,n])

    MEDM[0][0] = 1.

    MEDM[1][0] = -6. * factor
    MEDM[1][1] = 11. * factor
    MEDM[1][2] = -4. * factor
    MEDM[1][3] = -1. * factor
    
    for i in range(2,n-2):
        MEDM[i][i-2] = -1.
        MEDM[i][i-1] = -4.
        MEDM[i][i] = 10.
        MEDM[i][i+1] = -4.
        MEDM[i][i+2] = -1.


    MEDM[n-2][n-1] = -6. * factor
    MEDM[n-2][n-2] = 11. * factor
    MEDM[n-2][n-3] = -4. * factor
    MEDM[n-2][n-4] = -1. * factor

    MEDM[n-1][n-1] = 12.*h
    MEDM[n-1][n-2] = -16.*h
    MEDM[n-1][n-3] = 4.*h

    MEDM *= 1./(8.*h*h)

    return MEDM

#############################################################################
# Loading for the LLEM and VHM 
#############################################################################

def force(n,h):
    
    force = np.zeros(n)
   
    for i in range(1,n-1):
        force[i] = f(i * h)
    
    force[n-1] = 1
    
    return force

#############################################################################
# Compute the maximum relative error for the LLEM, the EDM, and the VHM
#############################################################################

def error(n,h,u):
    e = []
    
    for i in range(1,n):
        e.append(abs((exactSolution(i*h)-u[i])/exactSolution(i*h)))

    return max(e)


#############################################################################
# Compute error at each grid point
#############################################################################

def errorplot(n,h,u):
    ep = []    
    for i in range(0,n):
        ep.append(exactSolution(i*h)-u[i])

    return ep


#############################################################################
#Computation
#############################################################################

markers = ['s','o','x','.']
print("n,delta,EDM,EDMCorrection1,EDMCorrection2")

for i in range(3,7):
    n = np.power(2,i)
    h = 1./n
    nodes = n+1
    delta = 2*h
    
    x = np.linspace(0,1.,nodes)
    
    uEDM = solve(EDM(nodes,h),force(nodes,h))
    uEDMCORRECTION = solve(EDMCORRECTION(nodes,h),force(nodes,h))
    uEDMCORRECTION2 = solve(EDMCORRECTION2(nodes,h),force(nodes,h))
    plt.scatter(x,uEDMCORRECTION-exactSolution(x),color='black',marker=markers[i-3],label="$\delta="+str(delta)+"$")
    print(str(n)+","+str(2*h)+","+str(error(nodes,h,uEDM))+","+str(error(nodes,h,uEDMCORRECTION))+","+str(error(nodes,h,uEDMCORRECTION2)))

plt.grid()
plt.xlabel("x")
plt.ylabel("Error in displacement")
plt.title("Example with Quartic Solution using corrected EDM I")
plt.legend()
plt.savefig("EDM-Corrected-Error-"+example+"Solution.pdf",bbox_inches='tight')
    
plt.cla()

for i in range(3,7):
    n = np.power(2,i)
    h = 1./n
    nodes = n+1
    delta = 2*h
    
    x = np.linspace(0,1.,nodes)
    
    uEDMCORRECTION2 = solve(EDMCORRECTION2(nodes,h),force(nodes,h))
    plt.scatter(x,uEDMCORRECTION2-exactSolution(x),color='black',marker=markers[i-3],label="$\delta="+str(delta)+"$")

plt.grid()
plt.xlabel("x")
plt.ylabel("Error in displacement")
plt.title("Example with Quartic Solution using corrected EDM II")
plt.legend()
plt.savefig("EDM-Corrected-Error-2-"+example+"Solution.pdf",bbox_inches='tight')


plt.cla()

for i in range(4,5):
    n = np.power(2,i)
    h = 1./n
    nodes = n+1
    delta = 2*h
    
    x = np.linspace(0,1.,nodes)
    
    uEDM = solve(EDM(nodes,h),force(nodes,h))
    uEDMCORRECTION = solve(EDMCORRECTION(nodes,h),force(nodes,h))
    uEDMCORRECTION2 = solve(EDMCORRECTION2(nodes,h),force(nodes,h))
    plt.scatter(x,abs(uEDM-exactSolution(x)),color='black',marker=markers[0],label="EDM")
    plt.scatter(x,abs(uEDMCORRECTION-exactSolution(x)),color='black',marker=markers[1],label="EDM I")
    plt.scatter(x,abs(uEDMCORRECTION2-exactSolution(x)),color='black',marker=markers[2],label="EDM 2")

plt.grid()
plt.xlabel("x")
plt.ylabel("Absolute error in displacement")
plt.title("Accuracy Study with Quartic Solution for $\delta=0.125$")
plt.legend()
plt.savefig("EDM-Corrected-Comparison-"+example+"Solution.pdf",bbox_inches='tight')
