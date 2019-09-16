import numpy as np
import sys 
import matplotlib.pyplot as plt

if not len(sys.argv) == 2:
    print("Error: Either provide linear or quadratic as the first argument")
    sys.exit()
    
example = sys.argv[1] 

#############################################################################
# Solve the system
#############################################################################

def solve(M,f):
    return np.linalg.solve(M,f)

#############################################################################
# Exact solution 
#############################################################################

def exactSolution(x):
    
    if example == "linear":
        return x*(3.-x)*(3.+x)/6.
    elif example == "quadratic":
        return x*(4.+x*x*x)/12.
    else:
        print("Error: Either provide linear or quadratic")
        sys.exit()
        
    
#############################################################################
# Loading
#############################################################################

def f(x):
    
    if example == "linear":
        return x
    elif example == "quadratic":
        return x*x
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
# Assemble the stiffness matrix for the extended domain model (EMD)
#############################################################################

def EDM(n,h):
    
    #h=1
    length = n+4
    
    MEDM = np.zeros([length,length])

    MEDM[0][0] = 1.
    MEDM[0][2] = -2.
    MEDM[0][4] = 1.
    
    MEDM[1][1] = 1
    MEDM[1][2] = -2.
    MEDM[1][3] = 1.
    
    MEDM[2][2] = 1

    for i in range(3,length-3):
        MEDM[i][i-2] = -1.
        MEDM[i][i-1] = -4.
        MEDM[i][i] = 10.
        MEDM[i][i+1] = -4.
        MEDM[i][i+2] = -1.


    MEDM[length-3][length-1] = h
    MEDM[length-3][length-2] = 2.*h
    MEDM[length-3][length-4] = -2.*h
    MEDM[length-3][length-5] = -1.*h
    
    MEDM[length-2][length-2] = 1.
    MEDM[length-2][length-3] = -2.
    MEDM[length-2][length-4] = 1.

    MEDM[length-1][length-1] = 1.
    MEDM[length-1][length-3] = -2.
    MEDM[length-1][length-5] = 1.
    
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
# Loading for the EMD
#############################################################################

def forceEDM(n,h):
    
    force = np.zeros(n+4)
    
    
    for i in range(3,n+4-3):
        force[i] = f((i-2) * h)
    
    force[n+4-3] = 1
    
    #print force
    
    return force

#############################################################################
# Compute the maximum relative error for the LLEM and the VHM
#############################################################################

def error(n,h,u):
    e = []
    
    for i in range(1,n):
        e.append(abs((exactSolution(i*h)-u[i])/exactSolution(i*h)))
        #print(exactSolution((i)*h),u[i],(i)*h)

    return max(e)


#############################################################################
# Compute the maximum relative error for the EDM
#############################################################################

def errorEDM(n,h,u):
    e = []
    
    for i in range(3,n+2):
        e.append(abs((exactSolution((i-2)*h)-u[i])/exactSolution((i-2)*h)))
        #print(exactSolution((i-2)*h),u[i],(i-2)*h)

    return max(e)


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
    
    x = np.linspace(0,1.,nodes)
    
    uLLEM = solve(LLEM(nodes,h),force(nodes,h))
    uVHM = solve(VHM(nodes,h),force(nodes,h))
    uEDM = solve(EDM(nodes,h),forceEDM(nodes,h))
    
    print(str(n)+","+str(h)+","+str(error(nodes,h,uLLEM))+","+str(errorEDM(nodes,h,uEDM))+","+str(error(nodes,h,uVHM)))
    
    
    # Plot the displacement
    if n == 4:
        ax1.plot(x,exactSolution(x),label="Exact",c="black")
        ax2.plot(x,exactSolution(x),label="Exact",c="black")
        ax3.plot(x,exactSolution(x),label="Exact",c="black")
    
    ax1.scatter(x, uLLEM,label=h)
    ax2.scatter(x, uVHM,label=h)
    ax3.scatter(x, uEDM[2:len(uEDM)-2])


    

lines_labels = [ax1.get_legend_handles_labels()]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

figure.legend(lines, labels,loc=8,ncol=5)


figure.suptitle("Loading: "+example)
plt.xlabel("Node position")
ax1.set_title('LLEM')
ax1.set_ylabel('u')
ax1.grid()
ax2.set_title('VHM')
ax2.grid()
ax2.set_ylabel('u')
ax3.set_title('EDM')
ax3.grid()
ax3.set_ylabel('u')
plt.savefig(example+"-displacement.pdf",bbox_inches='tight')
    
    
