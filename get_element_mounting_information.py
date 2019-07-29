# Atom issues
#   Packages -> Script -> Configure
'''
=============================================================================
Auxiliary function to assist technicians in mounting strain gauges (SG's)
Finds: mounting point (i)
       orientation    (ii)

Required:
---------
  bdf_filename: bdf/dat filename; Nastran input file
  element_info: [(a, b)]; list of tuples
                a -- str(); Strain gauge (SG) orientation (in elemental
                              coordinate system with respect to x-axis),
                b -- int(); Element IDs (only CQUAD4) of SG.
   OPTIONAL:
   ==========
      gcid:     ID of Global CSYS (Coordinate System)
                    int; default = 0
                         gcid = 0 (basic CSYS) or >= 1 (local CSYS)
                    --> Local CSYS defined in COORDINATE SYSTEM CARD
                    --> Only type rectangular (CORD2R)
      mode:     type of Nastran solver; valid_modes = {'msc', 'nx'}
                      str; default = 'nx'
      method:   method for defining coordinates
                    int; default = 0
                         method  = 0 (results in global CSYS)
                         method  = 1 (results in global CSYS,
                                      relative to reference node)
      ref_node: reference node
                     int; default = None
                          only assign value if method == 1

NOTES:
------
   element:  each CQUAD4 element represents a virtual STRAIN GAUGE
   gcid:     global coordinate system (CSYS) identification number (ID)
                  BASIC and LOCAL CSYS are grouped as GLOBAL CSYS
                          --> Basic: also known as absolute CSYS (gcid = 0)
                          --> Local: defined in COORDINATE SYSTEM CARD;
                                      any unique integer value (gcid >= 1)
   ecsys:   elemental coordinate system (CSYS);
                  unique for each element (see Nastran User Manual)

=============================================================================
'''

from pyNastran.bdf.bdf import read_bdf, BDF
from pyNastran.bdf.bdf_interface.get_card import GetCard
from pyNastran.bdf.cards.coordinate_systems import Coord
from tabulate import tabulate

import numpy as np
import pandas as pd
import math as m
import os

import get_strain_gauge_orientation as gsgo

# ====== Inputs ===============================================================
bdf_filename = 'z_rotation_test.dat'
element_info =  [('45', 1), ('0', 10)] # C_design.index.values

# OPTIONAL
gcid         = 0
mode         = 'nx'
method       = 0
ref_node     = None

# Create BDF object
bdf = read_bdf(bdf_filename=bdf_filename, mode=mode)

# ====== Error Testing ========================================================
try:
    # Test if bdf_filename & mode is of type string
    if not all(isinstance(i, str) for i in [bdf_filename, mode]):
        raise TypeError

    # Test if bdf_filename file exists in directory
    if os.path.isfile('./' + bdf_filename) == False:
        raise FileNotFoundError

    # Test if GRID & CQUAD4 data cards are in the BDF/DAT file
    if not all(item in bdf.card_count.keys() for item in ('GRID', 'CQUAD4')):
        raise CardNotFoundError

    # Test if CORD1R or CORD2R data cards are in the BDF/DAT file
    if not any(item in bdf.card_count.keys() for item in ('CORD1R', 'CORD2R')):
        raise CardNotFoundError

    # Test if mode has a value of either 'nx' or 'msc'
    if mode not in {'nx', 'msc'}:
        raise ValueError

    # Test if CSYS ID are in the DAT/BDF file
    if gcid not in list(bdf.coords.keys()):
        raise ValueError

    # Test if user chose an appropriate method
    if method not in [0, 1]:
        raise ValueError

    # Test if elements are a subset of all the elements
    total_element_ids = [eids[0] for eids in sorted(bdf.elements.items())]
    eidSet = set([elem[1] for elem in element_info])
    if not eidSet.issubset(total_element_ids):
        raise ValueError

    # Test reference_node and method combinations
    if ref_node != None and method == 0:  # reset to default settings
        ref_node = None
        print('Incorrect value combination assigned to variable; reset to default')
    elif ref_node != None and method == 1: # check if ref_node is in list of nodes
        total_node_ids = [nids[0] for nids in sorted(bdf.nodes.items())]
        if ref_node not in total_node_ids:
            raise ValueError
    elif ref_node == None and method == 1: # reset to default settings
        method = 0
        print('Incorrect value combination assigned to variable; reset to default')

except TypeError:
    print('Incorrect variable type')
except ValueError:
    print('Incorrect variable range')
except FileNotFoundError:
    print('BDF/DAT file not in current working directory')
except CardNotFoundError:
    print('Data Cards not found in BDF')


# ====== Create Output File ===================================================
f  = open('Strain Gauge Placement Information.txt', 'w')
f.write('Find below the information on where and how to mount strain gauges.\n')
f.write('Coordinate System (CSYS): \t\t\t\t' + str(gcid) + '\n')
if method == 1:
    f.write('With respect to Reference Node: \t' + str(ref_node)+ '\n')
f.write('\n')


# ====== Extract Information from input file ==================================
# Get index and transformation matrices for nodes
out = bdf.get_displacement_index_xyz_cp_cd()
icd_transform, icp_transform, xyz_cp, nid_cp_cd = out

nids = nid_cp_cd[:, 0]  # Get ids of nodes

# Get coordinates in ** gcid ** coordinate system
xyz_cid = bdf.transform_xyzcp_to_xyz_cid(xyz_cp, nids, icp_transform, cid=gcid)

# ====== Get Coordinates of Mounting Point ====================================
# Iterate over elements in element_info
for i, j in element_info:
    # Get nodes of element
    G1, G2, G3, G4 = bdf.elements[j].nodes[:]

    # Get coordinates of element
    coords_G1 = np.reshape(xyz_cid[nids == G1], 3)
    coords_G2 = np.reshape(xyz_cid[nids == G2], 3)
    coords_G3 = np.reshape(xyz_cid[nids == G3], 3)
    coords_G4 = np.reshape(xyz_cid[nids == G4], 3)

    if method == 1:

        # Get coordinates of reference node
        coords_reference_node = bdf.nodes[ref_node].xyz

        # Get coordinates of nodes of element relative to reference node
        coords_G1 = coords_G1 - coords_reference_node
        coords_G2 = coords_G2 - coords_reference_node
        coords_G3 = coords_G3 - coords_reference_node
        coords_G4 = coords_G4 - coords_reference_node

    # Construct info on element mounting point
    element   = np.array([coords_G1, coords_G2, coords_G3, coords_G4])
    mount_loc = element.mean(axis=0)

# ====== Get Orientation of Strain Gauge ========================================

    # Define GLOBAL and ELEMENTAL CSYS
        # coords_Gi are with respect to gcid;
        # --> thus ecsys defined in gcid
        # --> and gcsys equal to standard identity matrix
    ecsys = gsgo.get_elemental_csys(coords_G1, coords_G2, coords_G3, coords_G4)
    gcsys = np.identity(3)

    # Get transformation matrix, Q
        # from ECSYS -> GCSYS
    Q = gsgo.get_transformation_matrix(ecsys, gcsys)

    # Get direction of strain gauge (SG) in ecsys
    dir_SG_ecsys = [m.cos(m.radians(int(i))), m.sin(m.radians(int(i))), 0.]

    # Transform direction of SG in ecsys to gcsys
    dir_SG_gcsys = Q @ dir_SG_ecsys

    # Get orientation angles with respect to gcid
    alpha_x = m.degrees(m.acos(round(dir_SG_gcsys[0], 2)))
    beta_y  = m.degrees(m.acos(round(dir_SG_gcsys[1], 2)))
    gamma_z = m.degrees(m.acos(round(dir_SG_gcsys[2], 2)))

# ====== Write & Close Output Text File ==========================================

    header = [' ', 'x', 'y', 'z']
    matrix  = [['Coordinates', round(mount_loc[0], 2), round(mount_loc[1], 2), round(mount_loc[2], 2)],
               ['Rotation',    round(alpha_x, 2), round(beta_y, 2), round(gamma_z, 2)]]

    f.write('Element: ' + str(j) + '\n')
    f.write(tabulate(matrix, headers=header, tablefmt='psql') + '\n\n')
f.close()
