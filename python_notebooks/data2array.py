from __future__ import print_function, division, absolute_import
import array

import pynbody as pb
import numpy as np
import sys

IS_PY3 = sys.version_info.major == 3


def _read_size_in_bytes(fh):
    dummy = array.array("I")
    if IS_PY3:
        dummy.fromfile(fh, 1)
    else:
        dummy.read(fh, 1)
    return dummy[0]


def read_float_block(fh):

    # read "dummy": size of data blok in bytes
    size_in_bytes = _read_size_in_bytes(fh)
    size_in_floats = size_in_bytes // 4

    float_block = array.array("f")
    if IS_PY3:
        float_block.fromfile(fh, size_in_floats)
    else:
        float_block.read(fh, size_in_floats)
    result = float_block.tolist()

    # read "dummy" again (given from file format), must be the same as before:
    assert _read_size_in_bytes(fh) == size_in_bytes

    return np.array(result)

def get_lpicola_snap(filenamebase, path2data_lpicola, nslice=64):
    """
    Read separately each slice of the outputbox and creates arrays of position, velocity and potential
    for the whole box.
    """
    coord_list = []
    vel_list = []
    pot_list = []

    for i in range(nslice):
        filename = path2data_lpicola+(filenamebase+'.{}'.format(i))
        with open(str(filename), "rb") as fh:

            # skip header (256 bytes as in var.h + 2 x 4 for "dummy"):
            header = fh.read(256 + 8)

            coord_list.append(read_float_block(fh).reshape(-1, 3))
            vel_list.append(read_float_block(fh).reshape(-1, 3))
            pot_list.append(read_float_block(fh))
            
    pos_wholebox = np.vstack(coord_list)
    vel_wholebox = np.vstack(vel_list)
    pot_wholebox = np.hstack(pot_list)
    
    return pos_wholebox, vel_wholebox, pot_wholebox

def get_gadget_snaps(whichredshifts, path2data_gadget):
    """
    Read GADGET outputs with file extensions given in the list-parameter 'whichfiles'
    and returns a list with all snapshots of lenght n=len(whichfiles).
    """
    snapshots_list = []
    for i in whichredshifts:
        if (i < 10): snapshots_list.append(pb.load(path2data_gadget+'snapshot_00{}'.format(str(i))))
        else: snapshots_list.append(pb.load(path2data_gadget+'snapshot_0{}'.format(str(i))))
    return snapshots_list


def get_min_max(llist):
    """
    Return minimum and maximum value of the given parameterlist 
    """
    
    list_min = np.zeros(len(llist))
    list_max = np.zeros(len(llist))
    for i in llist:
        list_min[i] = llist[i].min()
        list_max[i] = llist[i].max()
    return list_min.min(), list_max.max()