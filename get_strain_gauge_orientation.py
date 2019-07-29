import numpy as np
import math as m

def get_normalized_vector(gA, gB):

    '''
    Utility function to calculate a normalized vector between grid points;
    order of arguments are important - vector will point from gA -> gB

    Inputs:
    -------
        gA, gB: numpy.arrays;
            grid points at gA and gB;

    ... to do:: gaan maak seker in watter CSYS gA en gB gedefinieer is


    Returns:
    --------
        norm_vector: numpy.arrays;
             normalized vector

    '''

    gAB = gB - gA
    gAB_Length = np.linalg.norm( gAB, 2 )

    norm_vector = gAB/gAB_Length

    return norm_vector


def get_elemental_csys(G1, G2, G3, G4):

    '''
    Calculates the elemental coordinate system (ECSYS) of a CQUD4 element

    Inputs:
    -------
        G1, G2, G3, G4: numpy.arrays;
                coordinate points of nodes G1, G2, G3, G4 that define
                CQUAD4 element.

    Returns:
    --------
        ECSYS: numpy.array;
            defines the elemental coordinate system in CSYS defined by cid

    '''
    # Find vector between nodes
    G12 = get_normalized_vector(G1, G2)     # From nodes G1 -> G2
    G13 = get_normalized_vector(G1, G3)     # From nodes G1 -> G3
    G21 = get_normalized_vector(G2, G1)     # From nodes G2 -> G1
    G24 = get_normalized_vector(G2, G4)     # From nodes G2 -> G4

    # Find angles
    beta = m.degrees(m.acos(np.dot(G12, G13)))
    gamma = m.degrees(m.acos(np.dot(G21, G24)))
    alpha = (beta + gamma)/2

    # Information from Nastran Element Library Reference (2014:4.14)

    # The orientation of the element coordinate system is determined by the order
    # of the connectivity for the grid points. The positive direction is from G1 to G4.
    # The element’s y-axis is perpendicular to the element x-axis

    # The === ELEMENTAL Z-AXIS === is normal to the x-y plane of the element. The positive
    # direction is defined by applying the right-hand rule to the ordering sequence of
    # G1 through G4 i.e. the  elememental z-axis is equal to the cross product of either vectors
    # (G13 x G24), (G24 x G31), (G31 x G42) or (G42 x G13).

    # ========= Z =====
    z_elemental = np.cross(G13, G24)
    z_elemental = z_elemental/np.linalg.norm(z_elemental,2)

    # The === ELEMENTAL X-AXIS & Y-AXIS === lies in the plane defined by G1, G2, G3 and G4.
    # The elemental y-axis is perpendicular to the elemental x-axis. The x-axis is the result of the in-plane
    # rotation of the vector G42 in an anti-clockwise direction about the z-axis.

    # ========= X =====
    G42 = get_normalized_vector(G4, G2)     # From nodes G4 -> G2

    c = m.cos(m.radians(alpha))
    s = m.sin(m.radians(alpha))

    # v' = cos(theta) * v + sin(theta) * (n X v)
    x_elemental = c*G42 + s*np.cross(z_elemental, G42)
    x_elemental = x_elemental/np.linalg.norm(x_elemental,2)

    # ========= Y =====
    y_elemental = np.cross(z_elemental, x_elemental)



    ECSYS = np.array([x_elemental, y_elemental, z_elemental])
    return ECSYS


def get_transformation_matrix(A, B):

    '''
    Utility function to calculate the rotation matrix from coordinate system A
    to coordinate system B. Both A and B are 3x3 matrices with
        Row 1: X-vector
        Row 2: Y-vector
        Row 3: Z-vector

    Definition of Rotation Matrix can be found at:
        Widnall, S. (2009) 'Vectors, Matrices and Coordinate Transformations', pp.10. Available at:
        https://ocw.mit.edu/courses/aeronautics-and-astronautics/16-07-dynamics-fall-2009/lecture-notes
        /MIT16_07F09_Lec03.pdf
        (Accessed: 10 April 2019).

    '''

    rot = np.array([  [np.dot(B[0],A[0]), np.dot(B[0],A[1]),   np.dot(B[0],A[2])],
                      [np.dot(B[1],A[0]), np.dot(B[1],A[1]),   np.dot(B[1],A[2])],
                      [np.dot(B[2],A[0]), np.dot(B[2],A[1]), 2*np.dot(B[2],A[2])]])

    return rot
