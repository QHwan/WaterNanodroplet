from __future__ import print_function, division, absolute_import

import numpy as np
from tqdm import tqdm
import MDAnalysis as md
import MDAnalysis.analysis.distances as mdanadist

from .parameter import Parameter
from .util import check_pbc, center_of_mass, distance_vector


class Potential(object):
    """Calculate potential of system.
    It can calculate potential of TIP4P/2005 system.
    Further modification is required.
    """

    def __init__(self, universe):
        """

        Parameters
        ----------
        universe : :obj:'MDAnalysis.core.universe.Universe'

        """
        self._universe = universe
        self._atom_vec = self._universe.select_atoms('all')

        self._num_frame = len(self._universe.trajectory)
        self._num_atom = len(self._atom_vec)

        self._param = Parameter()
        self._charge_vec, self._sigma_vec, self._epsilon_vec = self._initialize_parameters(self._param)


    def _initialize_parameters(self, param):
        """Initialize relevant parameters of atoms
        which cannot obtained from MDAnalysis module.
        Return flattend matrix: 2D -> 1D.

        Parameters
        ----------
        param : :obj:'parameter.Parameter'

        Returns
        -------
        charge_vec : float[:], shape = (self._num_atom * self._num_atom)
            q_i * q_j
        sigma_vec : float[:], shape = (self._num_atom * self._num_atom)
            (sigma_i+sigma_j)/2
        epsilon_vec : float[:], shape = (self._num_atom * self._num_atom)
            sqrt(eps_i*eps_j)
        """
        atom_name_vec = self._atom_vec.names
        charge_mat = np.array([[self._param.charge_dict[i]*self._param.charge_dict[j]
                                for j in atom_name_vec]
                                for i in atom_name_vec])
        sigma_mat = np.array([[(self._param.sigma_dict[i]+self._param.sigma_dict[j])*0.5
                                for j in atom_name_vec]
                                for i in atom_name_vec])
        sigma_mat *= 10
        epsilon_mat = np.array([[np.sqrt(self._param.epsilon_dict[i]*self._param.epsilon_dict[j])
                                for j in atom_name_vec]
                                for i in atom_name_vec])

        for i in range(int(self._num_atom/4)):
            for j in range(4):
                for k in range(4):
                    charge_mat[4*i+j, 4*i+k] = 0
        #np.fill_diagonal(charge_mat, 0)
        np.fill_diagonal(sigma_mat, 0)
        np.fill_diagonal(epsilon_mat, 0)
        return(charge_mat.ravel(), sigma_mat.ravel(), epsilon_mat.ravel())


    def potential_matrix(self):
        """Calculate molecular potential of nanodroplet.
        Returns
        -------
        lj_mat : float[:,:], shape = (self._num_frame, num_mol), unit = (kJ/mol)
        coul_mat : float[:,:], shape = (self._num_frame, num_mol), unit = (kJ/mol)
        """
        num_mol = int(self._num_atom/4)
        lj_mat = np.zeros((self._num_frame, num_mol))
        coul_mat = np.zeros_like(lj_mat)

        for i, ts in tqdm(enumerate(self._universe.trajectory), total=self._num_frame):
            dist_atom_vec = np.zeros((self._num_atom**2))
            dist_atom_vec = mdanadist.distance_array(self._atom_vec.positions, self._atom_vec.positions, box=ts.dimensions).ravel()

            lj_mat[i] += np.sum(np.sum(self._lennard_jones(dist_atom_vec), axis=1).reshape((-1,4)), axis=1)
            coul_mat[i] += np.sum(np.sum(self._coulomb(dist_atom_vec), axis=1).reshape((-1,4)), axis=1)

        return(lj_mat, coul_mat)



    def perturbed_potential_matrix(self, zeta=1e-5):
        """Calculate perturbed potential of nanodroplet.
        Returns
        -------
        dlj_mat : float[:,:], shape = (self._num_frame, num_mol), unit = (kJ/mol)
        dcoul_mat : float[:,:], shape = (self._num_frame, num_mol), unit = (kJ/mol)
        """
        num_mol = int(self._num_atom/4)
        extn_lj_mat = np.zeros((self._num_frame, num_mol))
        comp_lj_mat = np.zeros_like(extn_lj_mat)
        extn_coul_mat = np.zeros_like(extn_lj_mat)
        comp_coul_mat = np.zeros_like(extn_lj_mat)

        for i, ts in tqdm(enumerate(self._universe.trajectory), total=self._num_frame):
            box_vec = ts.dimensions[:3] 
            pos_atom_mat = self._atom_vec.positions
            pbc_pos_atom_mat = check_pbc(pos_atom_mat[0], pos_atom_mat, box_vec)

            com_vec = center_of_mass(pbc_pos_atom_mat[0::4])
            pbc_pos_atom_mat -= com_vec
            
            extn_pos_atom_mat = np.zeros_like(pos_atom_mat)
            comp_pos_atom_mat = np.zeros_like(pos_atom_mat)
            
            extn_pos_atom_mat[0::4] = pbc_pos_atom_mat[0::4]*(1+zeta)
            comp_pos_atom_mat[0::4] = pbc_pos_atom_mat[0::4]*(1)

            for j in range(1,4):
                extn_pos_atom_mat[j::4] = extn_pos_atom_mat[0::4] + (pbc_pos_atom_mat[j::4] - pbc_pos_atom_mat[0::4])
                comp_pos_atom_mat[j::4] = comp_pos_atom_mat[0::4] + (pbc_pos_atom_mat[j::4] - pbc_pos_atom_mat[0::4])


            extn_dist_atom_vec = distance_vector(extn_pos_atom_mat)
            comp_dist_atom_vec = distance_vector(comp_pos_atom_mat)

            extn_lj_mat[i] += np.sum(np.sum(self._lennard_jones(extn_dist_atom_vec), axis=1).reshape((-1,4)), axis=1)
            extn_coul_mat[i] += np.sum(np.sum(self._coulomb(extn_dist_atom_vec), axis=1).reshape((-1,4)), axis=1)
            comp_lj_mat[i] += np.sum(np.sum(self._lennard_jones(comp_dist_atom_vec), axis=1).reshape((-1,4)), axis=1)
            comp_coul_mat[i] += np.sum(np.sum(self._coulomb(comp_dist_atom_vec), axis=1).reshape((-1,4)), axis=1)

        return(extn_lj_mat - comp_lj_mat, extn_coul_mat - comp_coul_mat)


    def _lennard_jones(self, dist_vec):
        """Calculate lennard jones potential matrix of given distance matrix
        Parameters
        ----------
        dist_mat : float[:], shape = (self._num_atom * self._num_atom)
        
        Returns
        -------
        lj_mat : float[:,:], shape = (self._num_atom, self._num_atom)
        """
        lj_vec = np.zeros(self._num_atom**2)
        r6_vec = np.zeros_like(lj_vec)
        mask = np.where(self._epsilon_vec != 0)
        r6_vec[mask] = (self._sigma_vec[mask]/dist_vec[mask])**6
        lj_vec[mask] = 4*self._epsilon_vec[mask]*r6_vec[mask]*(r6_vec[mask]-1)
        return(lj_vec.reshape((self._num_atom, self._num_atom)))


    def _coulomb(self, dist_vec):
        """Calculate coulomb potential matrix of given distance matrix
        Parameters
        ----------
        dist_vec : float[:], shape = (self._num_atom * self._num_atom)
        
        Returns
        -------
        coul_mat : float[:,:], shape = (self._num_atom, self._num_atom)
        """
        coul_vec = np.zeros(self._num_atom**2)
        mask = np.where(self._charge_vec != 0)
        coul_vec[mask] = self._charge_vec[mask]/dist_vec[mask]
        coul_vec *= 138.935458 * 10 # 10 is angstrom -> nm conversion
        return(coul_vec.reshape((self._num_atom, self._num_atom)))

           


## Test Suite ##
if __name__ == "__main__":
    u = md.Universe('trj/md3.tpr', 'trj/md3.gro')
    pot = Potential(u)
    box_vec = u.trajectory[0].dimensions[:3] 
    pos_atom_mat = pot._atom_vec.positions
    pbc_pos_atom_mat = check_pbc(pos_atom_mat[0], pos_atom_mat, box_vec)
    dist_atom_vec = distance_vector(pbc_pos_atom_mat)

    lj_mat = pot._lennard_jones(dist_atom_vec)
    coul_mat = pot._coulomb(dist_atom_vec)

    print('#############################################')
    print('######## Unit Test: Potential Module ########')
    print('#############################################')
    print('\n')
    print('Potential calculated from WaterNanodroplet.potential module')
    print('-----------------------------------------------------------')
    print('LJ: {lj}, Coul: {coul}'.format(lj=np.sum(lj_mat)/2,
                                          coul=np.sum(coul_mat)/2))
    print('\n')
    print('Potential calculated from gromacs gmx energy')
    print('--------------------------------------------')
    print('LJ: {lj}, Coul: {coul}'.format(lj=3.9041, coul=-25.7737))
