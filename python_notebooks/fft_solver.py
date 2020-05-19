import numpy as np

import math
import cmath


# CIC algorithm like in L-PICOLA

def CIC(Nmesh, Box, pos, scalars):
    """
    Assigns the value of a scalar field in 3D at particle position
    to a mesh with 'Nmesh' nodes in each dimension.
    
    :param Nmesh: int; number of nodes in one dimension, preferably even
                  for faster fft
    :param Box: int; physical edge size of the simuation box
    :param pos: 3darray; array of particle positions in cartesian coordintes;
                po[0] refers to the x-component, etc.
    :param scalars: 1darray; array with size of number of particles in the simulation,
                    scalars[i] refers to the value of the scalar field at pos[i]
    :return field: 3darray; stores the field values at gridpoints in an array of shape
                   (Nmesh, Nmesh, Nmesh)
    """
    
    field = np.zeros((Nmesh, Nmesh, Nmesh))
    scaleBox = Nmesh/Box
    
    for i in range(pos.shape[0]):
        x, y, z = pos[i] * scaleBox
        Ix, Iy, Iz = int(x), int(y), int(z)
        Dx = x - Ix; Dy = y - Iy; Dz = z - Iz
        Tx, Ty, Tz = np.ones(3) - [Dx, Dy, Dz]
        Dy *= scalars[i]; Ty *= scalars[i]
        
        if(Ix >= Nmesh): Ix = 0
        if(Iy >= Nmesh): Iy = 0
        if(Iz >= Nmesh): Iz = 0
        
        Ixneigh = Ix + 1
        Iyneigh = Iy + 1
        Izneigh = Iz + 1
        if(Ixneigh >= Nmesh): Ixneigh = 0
        if(Iyneigh >= Nmesh): Iyneigh = 0
        if(Izneigh >= Nmesh): Izneigh = 0
        
        field[Ix, Iy, Iz]           += Tx*Ty*Tz;
        field[Ix, Iy, Izneigh]      += Tx*Ty*Dz;
        field[Ix, Iyneigh, Iz]      += Tx*Dy*Tz;
        field[Ix, Iyneigh, Izneigh] += Tx*Dy*Dz;

        field[Ixneigh, Iy, Iz]           += Dx*Ty*Tz;
        field[Ixneigh, Iy, Izneigh]      += Dx*Ty*Dz;
        field[Ixneigh, Iyneigh, Iz]      += Dx*Dy*Tz;
        field[Ixneigh, Iyneigh, Izneigh] += Dx*Dy*Dz;
    
    return field


# inverse CIC algorithm like in L-PICOLA

def ICIC(Nmesh, Box, pos, field):
    """
    Transforms the field values back to particle positions as oposed to CIC.
    
    :param Nmesh: int; number of nodes in one dimension, preferably even
                  for faster fft
    :param Box: int; physical edge size of the simuation box
    :param pos: 3darray; array of particle positions in cartesian coordintes;
                po[0] refers to the x-component, etc.
    :param field: 3darray; stores the field values at gridpoints in an array of shape
                   (Nmesh, Nmesh, Nmesh)
    :return field_part: 1darray; array with size of number of particles in the simulation,
                        field_part[i] refers to the value of the scalar field at pos[i]
    """
    
    Npart = pos.shape[0]
    field_part = np.zeros(Npart)
    scaleBox = Nmesh/Box
    
    for i in range(Npart):
        x, y, z = pos[i] * scaleBox
        Ix, Iy, Iz = int(x), int(y), int(z)
        Dx = x - Ix; Dy = y - Iy; Dz = z - Iz
        Tx, Ty, Tz = np.ones(3) - [Dx, Dy, Dz]
        
        if(Ix >= Nmesh): Ix = 0
        if(Iy >= Nmesh): Iy = 0
        if(Iz >= Nmesh): Iz = 0
        
        Ixneigh = Ix + 1
        Iyneigh = Iy + 1
        Izneigh = Iz + 1
        if(Ixneigh >= Nmesh): Ixneigh = 0
        if(Iyneigh >= Nmesh): Iyneigh = 0
        if(Izneigh >= Nmesh): Izneigh = 0
        
        field_part[i] = (field[Ix, Iy, Iz]           * Tx*Ty*Tz +
                         field[Ix, Iy, Izneigh]      * Tx*Ty*Dz +
                         field[Ix, Iyneigh, Iz]      * Tx*Dy*Tz +
                         field[Ix, Iyneigh, Izneigh] * Tx*Dy*Dz +

                         field[Ixneigh, Iy, Iz]           * Dx*Ty*Tz +
                         field[Ixneigh, Iy, Izneigh]      * Dx*Ty*Dz +
                         field[Ixneigh, Iyneigh, Iz]      * Dx*Dy*Tz +
                         field[Ixneigh, Iyneigh, Izneigh] * Dx*Dy*Dz)
    
    return field_part


# FFT routines for the poisson solver

def dft(data, inverse=False):
    """Return Discrete Fourier Transform (DFT) of a complex data vector"""
    N = len(data)
    transform = [ 0 ] * N
    for k in range(N):
        for j in range(N):
            angle = 2 * math.pi * k * j / float(N)
            if inverse:
                angle = -angle
            transform[k] += data[j] * cmath.exp(1j * angle)
    if inverse:
        for k in range(N):
            transform[k] /= float(N)
    return transform

def fft(data, inverse=False):
    """Return Fast Fourier Transform (FFT) using Danielson-Lanczos Lemma"""
    N = len(data)
    if N == 1:               # transform is trivial
        return [data[0]]
    elif N % 2 == 1:         # N is odd, lemma does not apply
        return dft(data, inverse)
    # perform even-odd decomposition and transform recursively
    even = fft([data[2*j] for j in range(N//2)], inverse)
    odd  = fft([data[2*j+1] for j in range(N//2)], inverse)
    W = cmath.exp(1j * 2 * math.pi / N)
    if inverse:
        W = 1.0 / W
    Wk = 1.0
    transform = [ 0 ] * N
    for k in range(N):
        transform[k] = even[k % (N//2)] + Wk * odd[k % (N//2)]
        Wk *= W
    if inverse:
        for k in range(N):
            transform[k] /= 2.0
    return transform

def sine_fft(data, inverse=False):
    """Return Fast Sine Transform of N data values starting with zero."""
    N = len(data)
    if data[0] != 0.0:
        raise Exception("data[0] != 0.0")
    extend_data = [ 0.0 ] * (2*N)
    for j in range(1, N):
        extend_data[j] = data[j]
        extend_data[2*N-j] = -data[j]
    transform = fft(extend_data)
    sineft = [ 0.0 ] * N
    for k in range(N):
        sineft[k] = transform[k].imag / 2.0
        if inverse:
            sineft[k] *= 2.0 / N
    return sineft

def cosine_fft(data, inverse=False):
    """Return Fast Cosine Transform of (N+1) data values
       including two boundary points with index 0, N.
    """
    N = len(data)-1
    extend_data = [ 0.0 ] * (2*N)
    extend_data[0] = data[0]
    for j in range(1, N):
        extend_data[j] = data[j]
        extend_data[2*N-j] = data[j]
    extend_data[N] = data[N]
    transform = fft(extend_data)
    cosineft = [ 0.0 ] * (N+1)
    for k in range(N+1):
        cosineft[k] = transform[k].real / 2.0
        if inverse:
            cosineft[k] *= 2.0 / N
    return cosineft

def poisson_solver(Nmesh, Box, Nsample, pos, omega0=0.276):
    """
    :param Nmesh: int; number of nodes in one dimension, preferably even
                  for faster fft
    :param Box: int; physical edge size of the simuation box
    :param Nsample: int; Nsample**3 is the total number of particles in the simulation
    :param pos: 3darray; array of particle positions in cartesian coordintes;
                po[0] refers to the x-component, etc.
    :param omega0: float; cosmological parameter of matter density
    :return density: 3darray; the gravitational potential field at gridpoints
                     stored in an array of shape (Nmesh, Nmesh, Nmesh)
    """
    
    # density of a particle represented as a 1x1x1 mesh-cube in code units
    dens_unit = np.ones(Nsample**3) * (Nmesh/Nsample)**3
    print("Starting CIC of matter density...")
    dens_field = CIC(Nmesh, Box, pos, dens_unit)
    print("... done.")
    dens0 = 3*0.5 * omega0
    h = 1.                                                   # cell size of the mesh in code units
    density = dens0 * (dens_field - 1) + 0j                  # rhs of the poisson eq.
    
    # forward FFT of the rhs
    print("Starting forward FFT of the density field ...")
    print("... FFT of the 3rd axis ...")
    for i in range(Nmesh):
        for j in range(Nmesh):
            density[i,j,:] = fft(density[i,j,:])

    print("... FFT of the 2nd axis ...")
    for i in range(Nmesh):
        for j in range(Nmesh):
            density[i,:,j] = fft(density[i,:,j])
        
    print("... FFT of the 1st axis ...")
    for i in range(Nmesh):
        for j in range(Nmesh):
            density[:,i,j] = fft(density[:,i,j])
    print("... forward FFT finished.")
    
    # solving the potential in fourier space
    print("Solving the potential in fourier space ...")
    W = np.exp(2.0 * np.pi * 1j / Nmesh)
    Wm = 1.0; Wn = 1.0; Wl = 1.0;
    for m in range(Nmesh):
        for n in range(Nmesh):
            for l in range(Nmesh):
                denom = -6.
                denom += Wm + 1.0 / Wm + Wn + 1.0 / Wn + Wl + 1.0 / Wl
                if (denom != 0.0): density[m][n][l] *= h * h * h / denom
                Wl *= W
            Wn *= W
        Wm *= W
    print("... done.")
    
    # inverse FFT of the potential
    print("Starting inverse FFT of the potential ...")
    print("... FFT of the 3rd axis ...")
    for i in range(Nmesh):
        for j in range(Nmesh):
            density[i,j,:] = fft(density[i,j,:], inverse=True)

    print("... FFT of the 2nd axis ...")
    for i in range(Nmesh):
        for j in range(Nmesh):
            density[i,:,j] = fft(density[i,:,j], inverse=True)
        
    print("... FFT of the 1st axis ...")
    for i in range(Nmesh):
        for j in range(Nmesh):
            density[:,i,j] = fft(density[:,i,j], inverse=True)
    print("... inverse FFT finished.")
    
    return density