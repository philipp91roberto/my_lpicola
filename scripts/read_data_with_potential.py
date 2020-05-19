# encoding: utf-8

# ============================================================================================= #
# This is a simple Python code to read the particle's position, velocity and the gravitational  #
# potential at its position from a L-PICOLA snapshot slice output in GADGET_STYLE.              #
# The file header is omitted and the values are written in a python (Npart,3)-array, where      #
# the 3 elements of the last axis are the cartesian coordinates of either the particle's        #
# position, or the particle's velocity. The 1D-array holds the potential values of the          #
# particles ordered as they occur in the previous 3D-arrays. Npart is the number of particles   # 
# particular slice.                                                                             #
# ============================================================================================= #

from __future__ import print_function, division, absolute_import
import array
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

    
with open(filename, "rb") as fh:

    # skip header (256 bytes as in var.h + 2 x 4 for "dummy"):
    header = fh.read(256 + 8)

    pos = read_float_block(fh).reshape(-1, 3)
    vel = read_float_block(fh).reshape(-1, 3)
    pot = read_float_block(fh)

    print(pos.shape)
    print(vel.shape)
    print(pot.shape)