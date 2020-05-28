from __future__ import print_function, division, absolute_import

import numpy as np
from tqdm import tqdm
import MDAnalysis as md

from parameter import Parameter
from util import check_pbc, center_of_mass


class Shape(object):
    """Calculate shape of nanodroplet."""

    def __init__(self, universe):
        """

        Parameters
        ----------
        universe : :obj:'MDAnalysis.core.universe.Universe'

        """
        self._universe = universe
        self._ow_vec = self._universe.select_atoms('name OW')

        self._num_frame = len(self._universe.trajectory)
        self._num_ow = len(self._ow_vec)

        self._param = Parameter()
        self._mass_ow = self._param._mass_dict['OW']


    def _inertia_tensor_frame(self, ow_pos_mat, mass_ow):
        """Calculate inertia tensor of instanteneous nanodroplet snapshot.
        Parameters
        ----------
        ow_pos_mat : float[:,:], shape = (num_ow, 3)
            position matrix of ow atoms.
        mass_ow : float
            mass of ow atom.
        Returns
        -------
        inertia_mat : float[:,:], shape = (3, 3)
            moment of inertia tensor.
        """
        inertia_mat = np.zeros((3, 3))
        n_row, n_col = ow_pos_mat.shape

        _ow_pos_mat = ow_pos_mat - center_of_mass(ow_pos_mat)
        r2_vec = _ow_pos_mat[:,0]**2 + _ow_pos_mat[:,1]**2 + _ow_pos_mat[:,2]

        for i, _ow_pos_vec in enumerate(_ow_pos_mat):
            for j in range(3):
                inertia_mat[j, j] += r2_vec[i]
            for j in range(3):
                for k in range(3):
                    inertia_mat[j, k] -= _ow_pos_vec[j] * _ow_pos_vec[k]

        inertia_mat *= mass_ow

        return(inertia_mat)

    def _principal_axes(self, inertia_tensor, mass_ow, num_ow):
        principal_axes = np.zeros(3)

        eigvals, _ = np.linalg.eig(inertia_tensor)
        eigval_sum = np.sum(eigvals)
        for i in range(3):
            principal_axes[i] = eigval_sum - 2*eigvals[i]
        principal_axes *= 5/2/mass_ow/num_ow
        principal_axes = np.sqrt(principal_axes)
        return(principal_axes)


        


## Test Suite ##
if __name__ == "__main__":
    u = md.Universe('trj/md128_280k.tpr', 'trj/md128_280k_1000frame.xtc')
    shape = Shape(u)
    ts = shape._universe.trajectory[0]
    pos_ow = shape._ow_vec.positions
    inertia_tensor = shape._inertia_tensor_frame(pos_ow, shape._mass_ow)
    principal_axes = shape._principal_axes(inertia_tensor, shape._mass_ow, shape._num_ow)

    print("Inertia tensor: {}".format(inertia_tensor))
    print("Principal axes: {}".format(principal_axes))
           