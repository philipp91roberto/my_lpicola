# encoding: utf-8

# ============================================================================================= #
# This is a simple Python code to read the gravitational potential on the mesh specified by     #
# 'Nmesh' from a L-PICOLA snapshot slice output in GADGET_STYLE.                                #
# The file header is omitted and the values are written in a python (N,4)-array, where the      #
# first 3 elements of the last axis are the cartesian coordinates of the mesh point and the     #
# last elemnet is the value of the potential at this point. 'N' is the number of mesh points.   #
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
    
    field = read_float_block(fh).reshape(-1, 4)

    print(field.shape)

