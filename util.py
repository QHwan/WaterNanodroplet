from __future__ import print_function, division, absolute_import

import numpy as np

def distance_matrix(pos_mat):
    """Calculate distance matrix of given position matrix

    Parameters
    ----------
    pos_mat : float[:,:], shape = (num_atoms, 3)

    Returns
    -------
    dist_mat : float[:,:], shape = (num_atoms, num_atoms)
    """
    num_atoms = len(pos_mat)
    dist_mat = np.array([np.linalg.norm(pos_mat-pos_mat[i], axis=1)
                         for i in range(num_atoms)])
    return(dist_mat)


def center_of_mass(pos_mat):
    """Calculate center of mass of given position matrix

    Parameters
    ----------
    pos_mat : float[:,:], shape = (num_atoms, 3)

    Returns
    -------
    com_vec : float[:], shape = 3
    """
    com_vec = np.sum(pos_mat, axis=0)
    com_vec /= len(pos_mat)
    return(com_vec)


def check_pbc(ref_pos_vec, pos_mat, box_vec):
    """Check periodicity of system and move atom positions

    Parameters
    ----------
    ref_pos_vec : float[:], shape = 3
    pos_mat : float[:,:], shape = (num_atoms, 3)
    box_vec : float[:], shape = 3

    Returns
    -------
    pbc_pos_mat : float[:,:], shape = (num_atoms, 3)
    """
    pbc_pos_mat = np.copy(pos_mat)
    for i in range(3):
        mask1 = pos_mat[:,i] - ref_pos_vec[i] > box_vec[i]/2 
        mask2 = ref_pos_vec[i] - pos_mat[:,i] > box_vec[i]/2
        pbc_pos_mat[mask1,i] -= box_vec[i]
        pbc_pos_mat[mask2,i] += box_vec[i]
    return(pbc_pos_mat)


def running_mean(x):
    """Calculate running mean of x

    Parameters
    ----------
    x : float[:]

    Returns
    -------
    run_x : float[:], shape_like x
    """
    run_x = np.zeros_like(x)
    for i in range(len(x)):
        if i == 0:
            avg = x[i]
        else:
            avg *= i
            avg += x[i]
            avg /= (i+1)
        run_x[i] = avg
    return(run_x)
