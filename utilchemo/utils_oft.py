#!/usr/bin/python3.8
import numpy as np
import math

def find_interface(profile):
    ## Returns the interface position for a monotonous density profile
    x_interface = np.mean(np.argpartition(np.gradient(profile), -2)[-2:]) 
    return(x_interface)

def find_interface_tanh_interpol(profile):

    return(-1)

def latticeWrapIdx(index, lattice_shape):
    """returns periodic lattice index
    for a given iterable index

    Required Inputs:
        index :: iterable :: one integer for each axis
        lattice_shape :: the shape of the lattice to index to
    """
    if not hasattr(index, '__iter__'): return index         # handle integer slices
    if len(index) != len(lattice_shape): return index  # must reference a scalar
    if any(type(i) == slice for i in index): return index   # slices not supported
    if len(index) == len(lattice_shape):               # periodic indexing of scalars
        mod_index = tuple(( (i%s + s)%s for i,s in zip(index, lattice_shape)))
        return mod_index
    raise ValueError('Unexpected index: {}'.format(index))

def phi_to_free_energy(phi1, phi_total, phi2, A, A2, B2, K2, E):
    
    #In the context of Marangoni surfers, phi1 describes the diffusing component and phi2
    #describes the phase-separating component
    
    phi1 = np.array(phi1)
    phi2 = np.array(phi2)

    NX, NY, NZ = np.shape(phi1)    

    #Setting the value of phi to np.nan for the solid nodes before calculating free energy 
    phi1[phi1 < -1000] = np.nan 
    phi2[phi2 < -1000] = np.nan 
    
    # Periodic boundary conditions

    extended_phi = np.zeros((NX+2, NY+2, NZ+2))
    x_indices, y_indices, z_indices = np.indices((NX+2, NY+2, NZ+2))
    x_indices -= 1
    y_indices -= 1
    z_indices -= 1

    for i in range(NX+2):
        for j in range(NY+2):
            for k in range(NZ+2):
                extended_phi[i,j,k] = phi2[latticeWrapIdx((x_indices[i][j][k],y_indices[i][j][k],z_indices[i][j][k]), (NX,NY,NZ))]           
    
    grad = [np.gradient(extended_phi)[l][1:-1, 1:-1, 1:-1] for l in range(0,3)]
    
    #grad = np.array(np.gradient(phi2)) 
    
    #Calculating the free energies  
    FE = np.zeros((NX,NY,NZ)) 
    FE_2 = np.zeros((NX,NY,NZ)) 
    FE_1 = np.zeros((NX,NY,NZ)) 

    for i in range(NX): 
        for j in range(NY): 
            for k in range(NZ): 
                FE_2[i,j,k] = 0.5*A2*phi2[i,j,k]**2 + 0.25*B2*phi2[i,j,k]**4 + 0.5*K2*(grad[0][i,j,k]**2+ grad[1][i,j,k]**2+ grad[2][i,j,k]**2)
                FE_1[i,j,k] = 0.5*A*(phi1[i,j,k] - E*np.tanh(phi2[i,j,k]) - phi_total)**2  
    return FE_1, FE_2

def divide_around_colloid(NX,NY,ColX,ColY,N_cones): 
    theta = np.zeros((NX, NY)) 
    thetas = np.linspace(0, 2*math.pi, N_cones+1) 
    cones = np.zeros((NX, NY)) 
    for i in range(NX): 
        for j in range(NY): 
            I = i - ColX 
            J = j - ColY 
            if I > 0 and J >= 0: 
                theta[i,j] = math.atan(J / I) 
            elif I > 0 and J < 0: 
                theta[i,j] = math.atan(J / I) + 2*math.pi 
            elif I < 0: 
                theta[i,j] = math.atan(J / I) + math.pi 
            elif I == 0 and J > 0: 
                theta[i,j]  = math.pi / 2 
            elif I == 0 and J < 0: 
                theta[i,j]  = 3*math.pi / 2 

            for k in range(N_cones): 
                cones[i,j] = int(np.digitize(theta[i,j], bins = thetas)) 
    return np.flip(cones.T,0)

def edit_file(key, value, filename = 'input', info = True):

    with open(filename,'r') as f:
        lines = f.readlines()
        linenumber = 0
        key_found = False
        for line in lines:
            if line.startswith(key):
                key_found = True
                old_line = line.rstrip('\n')
                break
            linenumber +=1

        if type(value) == int:
            print('value type is int')
            lines[linenumber] = key+' '+"{:d}".format(value)+'\n'
        elif type(value) == float or type(value) == np.float64:
            print('value type is float')
            lines[linenumber] = key+' '+"{:.6f}".format(value)+'\n'

    with open(filename,'w') as f:
        f.writelines(lines)

    if info and key_found:
        if type(value) == int:
            print(old_line+' has been changed to '+ key+' '+"{:d}".format(value))
        elif type(value) == float or type(value) == np.float64:
            print(old_line+' has been changed to '+ key+' '+"{:.6f}".format(value))
    if not key_found:
        return(-1)

def FE_to_excess_FE(FE, x_interface, FE_bulk1, FE_bulk2):
    # Create matrix
    NX, NY, NZ = np.shape(FE)
    FE_step = np.zeros((NX, NY, NZ))
    for i in range(NX):
        for j in range(NY):
            for k in range(NZ):
                if i > x_interface:
                    FE_step[i,j,k] = FE_bulk1
                else:
                    FE_step[i,j,k] = FE_bulk2

    excess_FE = FE-FE_step
    return excess_FE
