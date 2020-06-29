#!/usr/bin/env python3
# @author patrickdiehl@lsu.edu
# @author serge.prudhomme@polymtl.ca
# @date 07/02/2020
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
# Variable horizon function
#############################################################################
def delta_v(i,delta,n):
    if i == 1 or i == n-2:
        return delta / 2
    else: 
        return delta
    
#############################################################################
# Loading
#############################################################################

def f(x,delta,n,i):
    
    if example == "Cubic":
        return x
    elif example == "Quartic":
        return x*x + delta_v(i,delta,n) * delta_v(i,delta,n)  / 12.
    elif example == "Quadratic":
        return 1
    elif example == "Linear":
        return 0
    else:
        print("Error: Either provide linear or quadratic")
        sys.exit()


#############################################################################
# Loading 
#############################################################################

def force(n,h,x,delta):
    
    force = np.zeros(n)
   
    for i in range(1,n-1):
        force[i] = f(x[i],delta,n,i)
    
    force[n-1] = 1 
    
    return force

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
# Compute the maximum relative error for the LLEM, the EDM, and the VHM
#############################################################################

def error(x,u):
    e = []
    
    for i in range(1,len(x)):
        e.append(abs((exactSolution(x[i])-u[i])/exactSolution(x[i])))

    return max(e)
        

#############################################################################
# Assemble the stiffness matrix for delta=2h
#############################################################################

def VHM2(n,h):
        
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
# Assemble the stiffness matrix for delta=4h
#############################################################################

def VHM4(n,h):
    
    MVHM = np.zeros([n,n])
    
    #Condition
    MVHM[0][0] = 1. / 2. / h / h
    
    # Node with one neighbor
    MVHM[1][0] = -1.  / 2. / h / h *2 
    MVHM[1][1] = 2.   / 2. / h / h *2 
    MVHM[1][2] = -1./ 2. / h / h *2
    
    # Node with two neighbors
    MVHM[2][0] = -(1./8.) / 2. / h / h  *2
    MVHM[2][1] = -(1./2.)/ 2. / h / h  *2
    MVHM[2][2] = (5./4.)/ 2. / h / h  *2
    MVHM[2][3] = -(1./2.)/ 2. / h / h  *2
    MVHM[2][4] = -(1./8.)/ 2. / h / h  *2
    
    # Node with three neighbors
    MVHM[3][0] = -(1./27.)/ 2. / h / h  *2
    MVHM[3][1] = -(1./9.)/ 2. / h / h *2
    MVHM[3][2] = -(2./9.)/ 2. / h / h *2
    MVHM[3][3] = (20./27.)/ 2. / h / h *2
    MVHM[3][4] = -(2./9.)/ 2. / h / h *2
    MVHM[3][5] = -(1./9.)/ 2. / h / h *2
    MVHM[3][6] = -(1./27.)/ 2. / h / h *2
    
    
    # Nodes with the full neigborhood
    for i in range(4,n-4):
        MVHM[i][i-4] = -(1./64.)/ 2. / h / h*2
        MVHM[i][i-3] = -(1./24.)/ 2. / h / h *2
        MVHM[i][i-2] = -(1./16.)/ 2. / h / h  *2
        MVHM[i][i-1] = -(1./8.)/ 2. / h / h *2
        MVHM[i][i] = (47./96.)/ 2. / h / h *2
        MVHM[i][i+1] = -(1./8.)/ 2. / h / h  *2
        MVHM[i][i+2] = -(1./16.)/ 2. / h / h *2
        MVHM[i][i+3] = -(1./24.)/ 2. / h / h *2
        MVHM[i][i+4] = -(1./64.)/ 2. / h / h*2
    
    # Node with three neighbors
    MVHM[n-4][n-7] = -(1./27.)/ 2. / h / h *2
    MVHM[n-4][n-6] = -(1./9.)/ 2. / h / h *2
    MVHM[n-4][n-5] = -(2./9.)/ 2. / h / h *2
    MVHM[n-4][n-4] = (20./27.)/ 2. / h / h *2
    MVHM[n-4][n-3] = -(2./9.)/ 2. / h / h *2
    MVHM[n-4][n-2] = -(1./9.)/ 2. / h / h *2
    MVHM[n-4][n-1] = -(1./27.)/ 2. / h / h *2
    
    
    # Node with two neighbors
    MVHM[n-3][n-5] = -(1./8.)/ 2. / h / h *2
    MVHM[n-3][n-4] = -(1./2.)/ 2. / h / h *2
    MVHM[n-3][n-3] = (5./4.)/ 2. / h / h *2
    MVHM[n-3][n-2] = -(1./2.)/ 2. / h / h *2
    MVHM[n-3][n-1] = -(1./8.)/ 2. / h / h *2
    
    # Node with one neighbor
    MVHM[n-2][n-3] = -1.   / 2. / h / h *2
    MVHM[n-2][n-2] = 2.   / 2. / h / h *2 
    MVHM[n-2][n-1] = -1/ 2. / h / h *2
    
    #Conditon
    MVHM[n-1][n-1] = 3.   / h  / 2. 
    MVHM[n-1][n-2] = -4.  / h  / 2. 
    MVHM[n-1][n-3] = 1.  / h  / 2. 
     
    return MVHM

#############################################################################
# Assemble the stiffness matrix for delta=4h
#############################################################################

def VHM8(n,h):
    
    MVHM = np.zeros([n,n])
    
    #Condition
    MVHM[0][0] = 1. / 2. / h / h 
    
    # Node with one neighbor
    MVHM[1][0] = -1.  /  2. / h  / h * 2
    MVHM[1][1] = 2   /  2. /  h / h   * 2
    MVHM[1][2] = -1./ 2.  / h  / h  *2
    
    # Node with two neighbors
    MVHM[2][0] =  -(1./8.)/ 2. / h / h *2
    MVHM[2][1] = -(1./2.)/ 2. / h / h *2
    MVHM[2][2] =  (5./4.)/ 2. / h / h *2 
    MVHM[2][3] = -(1./2.)/ 2. / h / h *2 
    MVHM[2][4] =  -(1./8.)/ 2. / h / h *2
    
    # Node with three neighbors
    MVHM[3][0] = -(1./27.)/ 2. / h / h *2
    MVHM[3][1] = -(1./9.)/ 2. / h / h *2
    MVHM[3][2] = -(2./9.)/ 2. / h / h *2
    MVHM[3][3] = (20./27.)/ 2. / h / h *2
    MVHM[3][4] = -(2./9.)/ 2. / h / h *2
    MVHM[3][5] = -(1./9.)/ 2. / h / h *2
    MVHM[3][6] = -(1./27.)/ 2. / h / h *2
    
    # Node with four neighbors
    MVHM[4][4-4] = -(1./64.)/ 2. / h / h *2
    MVHM[4][4-3] = -(1./24.)/ 2. / h / h *2
    MVHM[4][4-2] = -(1./16.)/ 2. / h / h *2
    MVHM[4][4-1] = -(1./8.)/ 2. / h / h *2
    MVHM[4][4] = (47./96.)/ 2. / h / h *2
    MVHM[4][4+1] = -(1./8.)/ 2. / h / h *2
    MVHM[4][4+2] = -(1./16.)/ 2. / h / h *2
    MVHM[4][4+3] = -(1./24.)/ 2. / h / h*2
    MVHM[4][4+4] = -(1./64.)/ 2. / h / h*2
    
    # Node with 5 neighbors
    MVHM[5][5-5] = -(1/125) / 2. / h / h*2
    MVHM[5][5-4] = -(1/50) / 2. / h / h*2
    MVHM[5][5-3] = -(2/75) / 2. / h / h*2
    MVHM[5][5-2] = -(1/25) / 2. / h / h*2
    MVHM[5][5-1] = -(2/25) / 2. / h / h*2
    MVHM[5][5] =  (131/375) / 2. / h / h*2
    MVHM[5][5+1] =  -(2/25) / 2. / h / h*2
    MVHM[5][5+2] =  -(1/25) / 2. / h / h*2
    MVHM[5][5+3] = -(2/75) / 2. / h / h *2
    MVHM[5][5+4] = -(1/50) / 2. / h / h *2
    MVHM[5][5+5] = -(1/125) / 2. / h / h *2
    
    # Node with 6 neighbors
    MVHM[6][6-6] =  -(1/216) / 2. / h / h *2
    MVHM[6][6-5] =  -(1/90) / 2. / h / h *2
    MVHM[6][6-4] =  -(1/72) / 2. / h / h *2
    MVHM[6][6-3] =  -(1/54) / 2. / h / h *2
    MVHM[6][6-2] =  -(1/36) / 2. / h / h *2
    MVHM[6][6-1] =  -(1/18) / 2. / h / h *2
    MVHM[6][6] =  (71/270) / 2. / h / h *2
    MVHM[6][6+1] =  -(1/18) / 2. / h / h *2
    MVHM[6][6+2] =  -(1/36) / 2. / h / h *2
    MVHM[6][6+3] =  -(1/54) / 2. / h / h *2
    MVHM[6][6+4] =  -(1/72) / 2. / h / h *2
    MVHM[6][6+5] =  -(1/90) / 2. / h / h *2
    MVHM[6][6+6] =  -(1/216) / 2. / h / h *2
    
    # Node with 7 neighbors
    MVHM[7][7-7] = -(1/343)  / 2. / h / h*2
    MVHM[7][7-6] = -(1/147)  / 2. / h / h*2
    MVHM[7][7-5] = -(2/245)  / 2. / h / h*2
    MVHM[7][7-4] = -(1/98)  / 2. / h / h*2
    MVHM[7][7-3] =  -(2/147)  / 2. / h / h*2
    MVHM[7][7-2] =  -(1/49)  / 2. / h / h*2
    MVHM[7][7-1] = -(2/49)  / 2. / h / h*2
    MVHM[7][7] =  (353/1715)  / 2. / h / h*2
    MVHM[7][7+1] = -(2/49)  / 2. / h / h*2
    MVHM[7][7+2] =  -(1/49)  / 2. / h / h*2
    MVHM[7][7+3] =  -(2/147)  / 2. / h / h*2
    MVHM[7][7+4] = -(1/98)  / 2. / h / h*2
    MVHM[7][7+5] = -(2/245)  / 2. / h / h*2
    MVHM[7][7+6] = -(1/147)  / 2. / h / h*2
    MVHM[7][7+7] = -(1/343)  / 2. / h / h*2
    
    # Nodes with the full neigborhood
    for i in range(8,n-8):
        MVHM[i][i-8] = -(1/512) / 2. / h / h*2
        MVHM[i][i-7] = -(1/224) / 2. / h / h*2
        MVHM[i][i-6] = -(1/192) / 2. / h / h*2
        MVHM[i][i-5] = -(1/160) / 2. / h / h*2
        MVHM[i][i-4] = -(1/128) / 2. / h / h*2
        MVHM[i][i-3] = -(1/96) / 2. / h / h*2
        MVHM[i][i-2] = -(1/64) / 2. / h / h*2
        MVHM[i][i-1] = -(1/32) / 2. / h / h*2
        MVHM[i][i] =  (1487/8960) / 2. / h / h*2
        MVHM[i][i+1] = -(1/32) / 2. / h / h*2
        MVHM[i][i+2] = -(1/64) / 2. / h / h*2
        MVHM[i][i+3] = -(1/96) / 2. / h / h*2
        MVHM[i][i+4] = -(1/128) / 2. / h / h*2
        MVHM[i][i+5] = -(1/160) / 2. / h / h*2
        MVHM[i][i+6] = -(1/192) / 2. / h / h*2
        MVHM[i][i+7] = -(1/224) / 2. / h / h*2
        MVHM[i][i+8] = -(1/512) / 2. / h / h*2
     
    # Node with 7 neighbors
    MVHM[n-8][n-15] = -(1/343)  / 2. / h / h *2
    MVHM[n-8][n-14] = -(1/147)  / 2. / h / h*2
    MVHM[n-8][n-13] = -(2/245)  / 2. / h / h*2
    MVHM[n-8][n-12] = -(1/98)  / 2. / h / h*2
    MVHM[n-8][n-11] =  -(2/147)  / 2. / h / h*2
    MVHM[n-8][n-10] =  -(1/49)  / 2. / h / h*2
    MVHM[n-8][n-9] =  -(2/49)  / 2. / h / h*2
    MVHM[n-8][n-8] = (353/1715)  / 2. / h / h*2
    MVHM[n-8][n-7] = -(2/49)  / 2. / h / h*2
    MVHM[n-8][n-6] =  -(1/49)  / 2. / h / h*2
    MVHM[n-8][n-5] = -(2/147)  / 2. / h / h*2
    MVHM[n-8][n-4] = -(1/98)  / 2. / h / h*2
    MVHM[n-8][n-3] = -(2/245)  / 2. / h / h*2
    MVHM[n-8][n-2] = -(1/147)  / 2. / h / h*2
    MVHM[n-8][n-1] = -(1/343)  / 2. / h / h*2
    
    # Node with 6 neighbors
    MVHM[n-7][n-13] =  -(1/216) / 2. / h / h *2
    MVHM[n-7][n-12] =  -(1/90) / 2. / h / h *2
    MVHM[n-7][n-11] =  -(1/72) / 2. / h / h *2
    MVHM[n-7][n-10] =  -(1/54) / 2. / h / h *2
    MVHM[n-7][n-9] =  -(1/36) / 2. / h / h *2
    MVHM[n-7][n-8] =  -(1/18) / 2. / h / h *2
    MVHM[n-7][n-7] = (71/270) / 2. / h / h *2
    MVHM[n-7][n-6] =   -(1/18) / 2. / h / h *2
    MVHM[n-7][n-5] =  -(1/36) / 2. / h / h *2
    MVHM[n-7][n-4] =  -(1/54) / 2. / h / h *2
    MVHM[n-7][n-3] =   -(1/72) / 2. / h / h *2
    MVHM[n-7][n-2] =  -(1/90) / 2. / h / h *2
    MVHM[n-7][n-1] =  -(1/216) / 2. / h / h *2
    
    # Node with 5 neighbors   
    MVHM[n-6][n-11] = -(1/125) / 2. / h / h *2
    MVHM[n-6][n-10] = -(1/50) / 2. / h / h*2
    MVHM[n-6][n-9] = -(2/75) / 2. / h / h*2
    MVHM[n-6][n-8] = -(1/25) / 2. / h / h*2
    MVHM[n-6][n-7] = -(2/25) / 2. / h / h*2
    MVHM[n-6][n-6] = (131/375) / 2. / h / h*2
    MVHM[n-6][n-5] = -(2/25) / 2. / h / h*2
    MVHM[n-6][n-4] = -(1/25) / 2. / h / h*2
    MVHM[n-6][n-3] = -(2/75) / 2. / h / h*2
    MVHM[n-6][n-2] =-(1/50) / 2. / h / h*2
    MVHM[n-6][n-1] = -(1/125) / 2. / h / h*2
    
    # Node with four neighbors
    MVHM[n-5][n-9] = -(1./64.)/ 2. / h / h *2
    MVHM[n-5][n-8] = -(1./24.)/ 2. / h / h*2
    MVHM[n-5][n-7] = -(1./16.)/ 2. / h / h*2
    MVHM[n-5][n-6] = -(1./8.)/ 2. / h / h*2
    MVHM[n-5][n-5] = (47./96.)/ 2. / h / h*2
    MVHM[n-5][n-4] = -(1./8.)/ 2. / h / h*2
    MVHM[n-5][n-3] = -(1./16.)/ 2. / h / h*2
    MVHM[n-5][n-2] = -(1./24.)/ 2. / h / h*2
    MVHM[n-5][n-1] = -(1./64.)/ 2. / h / h*2
    
    # Node with three neighbors 
    MVHM[n-4][n-7] = -(1./27.)/ 2. / h / h *2
    MVHM[n-4][n-6] = -(1./9.)/ 2. / h / h *2
    MVHM[n-4][n-5] = -(2./9.)/ 2. / h / h *2
    MVHM[n-4][n-4] = (20./27.)/ 2. / h / h *2
    MVHM[n-4][n-3] = -(2./9.)/ 2. / h / h *2
    MVHM[n-4][n-2] = -(1./9.)/ 2. / h / h *2
    MVHM[n-4][n-1] = -(1./27.)/ 2. / h / h *2
    
    # Node with two neighbors
    MVHM[n-3][n-5] = -(1./8.)/ 2. / h / h *2
    MVHM[n-3][n-4] = -(1./2.)/ 2. / h / h *2 
    MVHM[n-3][n-3] =  (5./4.)/ 2. / h / h *2 
    MVHM[n-3][n-2] = -(1./2.)/ 2. / h / h *2
    MVHM[n-3][n-1] = -(1./8.)/ 2. / h / h *2
    
    # Node with one neighbor
    MVHM[n-2][n-3] = -1.  /  2. / h  / h *2
    MVHM[n-2][n-2] = 2   /  2. /  h / h  *2
    MVHM[n-2][n-1] = -1./ 2.  / h  / h  *2
    
    #Conditon
    MVHM[n-1][n-1] = 3.   / h   / 2. 
    MVHM[n-1][n-2] = -4. /  h  / 2    
    MVHM[n-1][n-3] = 1. /   h / 2 
        
    return MVHM

#############################################################################
#Computation
#############################################################################
print("h,nodes,error")
delta = 0.25

# Case 1  
h = delta / 2
nodes = int(1 / h) + 1
x2 = np.linspace(0,1.,nodes)
f2=force(nodes,h,x2,delta)
u2 = solve(VHM2(nodes,h),f2)
plt.plot(x2,exactSolution(x2)-u2,label=r"$m=2$",color="black",linestyle="-")
print(h,len(x2),error(x2,u2))

# Case 2
h = delta / 4
nodes = int(1 / h) + 1
x4 = np.linspace(0,1.,nodes)
f4=force(nodes,h,x4,delta)
u4 = solve(VHM4(nodes,h),f4)
plt.plot(x4,abs(exactSolution(x4)-u4),label=r"$m=4$",color="black",linestyle=":")
print(h,len(x4),error(x4,u4))

# Case 3
h = delta / 8
nodes = int(1 / h) + 1
x8 = np.linspace(0,1.,nodes)
f8=force(nodes,h,x8,delta)
u8 = solve(VHM8(nodes,h),f8)
plt.plot(x8,exactSolution(x8)-u8,label=r"$m=8$",color="black",linestyle="-.")
print(h,len(x8),error(x8,u8))

plt.title("Convergence Study with Quartic Solution using VHM")
plt.xlabel("x")
plt.ylabel("Error in displacement")
plt.grid()
plt.legend()
plt.savefig("VHM-Quartic-convergence.pdf",bbox_inches='tight')

plt.close()
#plt.plot(x2,u2,label=r"$m=2$",color="black",linestyle="-")
#plt.plot(x4,u4,label=r"$m=4$",color="black",linestyle=":")
plt.plot(x8,u8,label=r"$m=8$",color="black",linestyle="-.")
plt.plot(x8,exactSolution(x8))
plt.grid()
plt.legend()
plt.savefig("VHM-Quartic-convergence-displacement.pdf",bbox_inches='tight')

    



