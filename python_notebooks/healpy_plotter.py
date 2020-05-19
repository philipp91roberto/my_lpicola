import healpy as hp, pickle
import pynbody as pb

import numpy as np


# we need this for healpix maps

def cart2spher(xyz):
    """
    Transformation from cartesian to spherical coordinates.
    
    :param xyz: 2darray; xyz[i,0] holds the x-coordinate of the i-th particle, 
                xyz[i,1] holds the y-coordinate, etc.
    :return spher: 2darray; spher[i,0] holds the distance from origin of the i-th particle,
                  spher[i,1] holds the polar angle defined from the z-axis down,
                  spher[i,2] holds the azimuthal angle
    """
    spher = np.zeros_like(xyz)
    xy = xyz[:,0]**2 + xyz[:,1]**2
    spher[:,0] = np.sqrt(xy + xyz[:,2]**2)
    #ind = (spher[:,0] < 1e-6)
    #for i in range(len(ind)):
    #    if (ind[i] == True): 
    #        spher[i,1] = 0.
    #        spher[i,2] = 0.
    #    else: 
    spher[:,1] = np.arccos(xyz[:,2] / spher[:,0]) # for elevation angle defined from Z-axis down
    #spher[:,1] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    spher[:,2] = np.arctan2(xyz[:,1], xyz[:,0])
    return spher


def healpix_field(snside, fnside, field, minv, maxv, spherical, dgrade=True, monopol=0, dipol=0):
    """
    This function creates a HEALPix map N_side = 'snside' from the data in the list field
    and an downgrade of this map to N_side = 'fnside' with the function 'healpy.ud_grade'.
    We do so to ensure that all information from the scalar field is used to create a map.
    Pixels with no value from 'field' are set with the constant 'healpy.UNSEEN'.
    
    :param snside: int; (n-th power of 2) healpix nside parameter for the initial tessellation of the map
    :param fnside: int; (n-th power of 2) healpix nside parameter for the final tessellation of the map,
                   such that all UNSEEN pixels vanish on the map
    :param field: list of numpy.array; field[0] is an (n,3)-array of the particle position;
                  if 'spherical' = True the array must contain the particle positions in spherical coordinates;
                  otherwise they are cartesian
    :param spherical: bool; if True this function uses a routine with particle positions in spherical, 
                      otherwise in cartesian coordinates
    :return hpxmap: 1darray; a HEALPix map with a pixelisation given by 'snside'
    :return dhpxmap: 1darray; a HEALPix map with a pixelistion given by 'fnside' downgraded from image_masked
    """
    
    # Set the number of sources and the coordinates for the input
    npix = hp.nside2npix(snside)
    indices  = 0.
    hpxmap   = np.zeros(npix, dtype=np.float) + hp.UNSEEN
    
    if (spherical):
        # angular coordinates and the field f
        thetas     = field[0][:,1]
        phis       = field[0][:,2]
        indices    = hp.ang2pix(snside, thetas, phis)
    else:
        # cartesian coordinates and the field fs
        x          = field[0][:,0]
        y          = field[0][:,1]
        z          = field[0][:,2]
        indices    = hp.vec2pix(snside, x, y, z)
    
    hpxmap[indices] = field[1]
    
    if dgrade:
        dhpxmap = hp.ud_grade(hpxmap, fnside)
        image   = hp.mollview(dhpxmap, 1, xsize=2000, min=minv, max=maxv,
                              norm=None, remove_mono=monopol, remove_dip=dipol)
        
        return dhpxmap, image
    
    image = hp.mollview(hpxmap, 1, norm=None, min=minv, max=maxv,
                        remove_mono=monopol, remove_dip=dipol, size=2000)
    return hpxmap, image


def healpix_meshfield(snside, fnside, field, minv, maxv, dgrade=True, monopol=0, dipol=0):
    """
    This function creates a HEALPix map N_side = 'snside' from the data in the list field
    and an downgrade of this map to N_side = 'fnside' with the function 'healpy.ud_grade'.
    We do so to ensure that all information from the scalar field is used to create a map.
    Pixels with no value from 'field' are set with the constant 'healpy.UNSEEN'.
    
    :param snside: int; (n-th power of 2) healpix nside parameter for the initial tessellation of the map
    :param fnside: int; (n-th power of 2) healpix nside parameter for the final tessellation of the map,
                   such that all UNSEEN pixels vanish on the map
    :param field: list of numpy.array; field[0] is an (n,3)-array of the particle position;
                  if 'spherical' = True the array must contain the particle positions in spherical coordinates;
                  otherwise they are cartesian
    :return hpxmap: 1darray; a HEALPix map with a pixelisation given by 'snside'
    :return dhpxmap: 1darray; a HEALPix map with a pixelistion given by 'fnside' downgraded from image_masked
    """
    
    # Set the number of sources and the coordinates for the input
    npix = hp.nside2npix(snside)
    indices  = 0.
    hpxmap   = np.zeros(npix, dtype=np.float) + hp.UNSEEN
    
    # cartesian coordinates and the field fs
    n = field.shape[0]
    X = np.arange(n)
    Y = np.arange(n)
    Z = np.arange(n)
    x, y, z = np.meshgrid(X, Y, Z, indexing='ij')
    x = x.reshape(-1) - n*0.5
    y = y.reshape(-1) - n*0.5
    z = z.reshape(-1) - n*0.5
    indices    = hp.vec2pix(snside, x, y, z)
    
    hpxmap[indices] = field.reshape(-1)
    
    if dgrade:
        dhpxmap = hp.ud_grade(hpxmap, fnside)
        image   = hp.mollview(dhpxmap, 1, xsize=2000, min=minv, max=maxv, unit=r'$\tilde{\phi}\;[code\ units]$',
                              norm=None, remove_mono=monopol, remove_dip=dipol)
        
        return dhpxmap, image
    
    image = hp.mollview(hpxmap, 1, norm=None, min=minv, max=maxv,
                        remove_mono=monopol, remove_dip=dipol, size=2000)
    return hpxmap, image


def healpix_density(nside, pos, minv, maxv, normfac=1., monopol=0):
    """
    This function creates a HEALPix map with N_side = 'nside'
    to visualize the number density field of the simulation box.
    
    :param nside: int; (n-th power of 2) healpix nside parameter for the tessellation of the map
    :param pos: (n,3)-array; particle positions in cartesian coordinates.
    :param normfac: int; normalisation factor for the number density
    
    :return hpxmap: 1darray; a HEALPix map with a pixelisation given by 'nside'
    """
    
    npix = hp.nside2npix(nside)
    indices  = 0.
    hpxmap   = np.zeros(npix, dtype=np.float)
    
    # read the coordinates from partpos
    x          = pos[:,0]
    y          = pos[:,1]
    z          = pos[:,2]
    
    indices         = hp.vec2pix(nside, x, y, z)
    bincount        = np.bincount(indices, minlength=npix) / normfac
    hpxmap[indices] = bincount[indices]
    
    magma_cmap      = cm.magma
    magma_cmap.set_under("w")  # sets background to white
    image           = hp.mollview(hpxmap, 1, norm=None, min=minv, max=maxv,
                                  cmap=magma_cmap, remove_mono=monopol, xsize=2000)
    
    return hpxmap, image


def average_shells(r_coord, field_values, nshells, normfac=1.):
    """
    A spherical ball of radius 'r_coord.max()' is divided into 'nshells' concetric shells of equal width.
    This function computes the mean value over a scalar field, specified by each pair in (r_coord, field_values), 
    of each shell.
    
    :param r_coord: 1darray; distance from the origin of each particle in the box
    :param field_values: 1dlike; scalar values of each particle corresponding to r_coord
    :param nshells: int; number of concentric shells of equal witdth making up the box
    :param normfac: int, optional; factor for an arbitrary normalization of the mean value
    :return shellpos: 1darray; each entry specifies the radius of the mid-layer of the 'nshells' shells
    :return mean: 1darray; hold the mean values of the scalar field over the 'nshells' shells
    """
    
    r        = np.ceil(r_coord.max())                          # radius of the layer of the outmost shell
    radii    = np.linspace(0, r, num=nshells+1, endpoint=1)    # radii of the shell layers
    shellpos = 0.5*(radii[:-1] + radii[1:])                    # radius of the mid-layer of each shell
    mean     = np.zeros_like(shellpos)
    
    # this loop takes the mean of entries in field_values laying within a given shell
    for i in range(nshells):
        ind = np.logical_and(radii[i] < r_coord, radii[i+1] > r_coord)
        with np.errstate(invalid='raise'):
            try:
                mean[i] = np.mean(field_values[ind])
            except FloatingPointError:
                print("Taking the mean value of an empty shell! Reduce number of shells.")
                break
    
    return shellpos, mean 