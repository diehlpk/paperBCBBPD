#!/usr/bin/env python3
# @author patrickdiehl@lsu.edu
# @author serge.prudhomme@polymtl.ca
# @date 10/05/2019
import numpy as np
import sys 
import matplotlib.pyplot as plt

example = "Quartic"

#############################################################################
#  kappa(x) Equation 10 in the paper
#############################################################################

def s(x,delta):
    if x < delta:
        return (2*delta*delta)/(delta*delta+x*x)
    if x > 1 - delta:
        return (2*delta*delta)/(delta*delta+(1-x)*(1-x))
    else:
        return 1


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
# Assemble the stiffness matrix for our surface correction
#############################################################################
def SCM(n,h,delta):

    M = np.zeros([n,n])

    M[0][0]=1 * s(x[0],delta) / h  / h
    M[1][0] = -1. * s(x[1],delta)  / h / h 
    M[1][1] = 2.  * s(x[1],delta) /  h / h  
    M[1][2] = -1. * s(x[1],delta)  / h / h 


    for i in range(2,n-2):
        M[i][i-2] = -(1./8.) * s(x[i],delta)  / h / h 
        M[i][i-1] = -(1./2.) * s(x[i],delta)  / h / h 
        M[i][i] = (5./4.) * s(x[i],delta)  / h / h 
        M[i][i+1] = -(1./2.) * s(x[i],delta)  / h / h 
        M[i][i+2] = -(1./8.) * s(x[i],delta)  / h / h 
    
    M[n-2][n-1] = -1. * s(x[n-2],delta)  / h / h 
    M[n-2][n-2] = 2. * s(x[n-2],delta)  / h / h 
    M[n-2][n-3] = -1. * s(x[n-2],delta)  / h / h 

    M[n-1][n-1] = 3. *  s(x[n-1],delta)    / h  / 4. 
    M[n-1][n-2] = -4. * s(x[n-1],delta) / h   / 4.  
    M[n-1][n-3] = 1. * s(x[n-1],delta)   /  h / 4.

    return M


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

figure, (ax1, ax2,ax3) = plt.subplots(3, 1, sharex=True)
markers = ['|','.','*','+']
print("n,h,LLEM,EDM,VHM")


for i in range(2,6):
    n = np.power(2,i)
    h = 1./n
    nodes = n+1
    delta = 2*h
    
    x = np.linspace(0,1.,nodes)
    
    uLLEM = solve(LLEM(nodes,h),force(nodes,h))
    uEDM = solve(EDM(nodes,h),force(nodes,h))
    uVHM = solve(VHM(nodes,h),force(nodes,h))
    eEDM = errorplot(nodes,h,uEDM)
    uSCM = solve(SCM(nodes,h,delta),force(nodes,h))
    
    
    print(str(n)+","+str(h)+","+str(error(nodes,h,uLLEM))+","+str(error(nodes,h,uEDM))+","+str(error(nodes,h,uVHM))+","+str(error(nodes,h,uSCM)))

    
    # Plot the displacement
    if n == 8:
        ax1.plot(x,exactSolution(x),label="Exact",c="black")
        ax2.plot(x,exactSolution(x),label="Exact",c="black")
        ax3.plot(x,exactSolution(x),label="Exact",c="black")
    
    ax1.scatter(x, uLLEM,label=h)
    ax2.scatter(x, uEDM,label=h)
    ax3.scatter(x, uVHM,label=h)
        

lines_labels = [ax1.get_legend_handles_labels()]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

figure.legend(lines, labels,loc=8,ncol=5)


figure.suptitle(example+" Solution")
plt.xlabel("Node position")
ax1.set_title('LLEM')
ax1.set_ylabel('u')
ax1.grid()
ax2.set_title('EDM')
ax2.grid()
ax2.set_ylabel('u')
ax3.set_title('VHM')
ax3.grid()
ax3.set_ylabel('u')
plt.savefig(example+"-displacement.pdf",bbox_inches='tight')
    

#############################################################################
# Figure with errors in displacement
#############################################################################

plt.figure(2)
markers = ['s','o','x','.']

for i in range(2,6):
    n = np.power(2,i)
    h = 1./n
    nodes = n+1
    
    x = np.linspace(0,1.,nodes)
    
    uEDM = solve(EDM(nodes,h),force(nodes,h))
    eEDM = errorplot(nodes,h,uEDM)
    plt.scatter(x,eEDM,label="h="+str(h), marker=markers[i-2], c="black")

plt.legend(loc= "lower left")
plt.xlabel("x")
plt.ylabel('Error in displacement')
plt.xscale('linear')
plt.yscale('linear')
plt.title("Example with "+example+" Solution using EDM")
plt.grid(True)
plt.savefig("EDM-Error-"+example+"Solution.pdf",bbox_inches='tight')

#############################################################################
# Figure with errors in displacement
#############################################################################

plt.figure(3)
markers = ['s','o','x','.']

for i in range(2,6):
    n = np.power(2,i)
    h = 1./n
    nodes = n+1
    
    x = np.linspace(0,1.,nodes)
    
    uVHM = solve(VHM(nodes,h),force(nodes,h))
    eVHM = errorplot(nodes,h,uVHM)
    plt.scatter(x,eVHM,label="h="+str(h), marker=markers[i-2], c="black")

plt.legend(loc= "upper left")
plt.xlabel("x")
plt.ylabel('Error in displacement')
plt.xscale('linear')
plt.yscale('linear')
plt.title("Example with "+example+" Solution using VHM")
plt.grid(True)
plt.savefig("VHM-Error-"+example+"Solution.pdf",bbox_inches='tight')
