from __future__ import print_function, division, absolute_import

import numpy as np
from tqdm import tqdm
import MDAnalysis as md

from parameter import Parameter

class RadialDensity(object):
    """Calculate radial density of system."""

    def __init__(self, universe):
        """

        Parameters
        ----------
        universe : :obj:'MDAnalysis.core.universe.Universe'

        """
        self._universe = universe
        self._box_mat = np.array([ts.dimensions for ts in self._universe.trajectory])
        self._atom_vec = self._universe.select_atoms('all')
        self._mass_vec = self._initialize_parameters(Parameter())

        self._num_frame = len(self._universe.trajectory)
        self._num_atom = len(self._atom_vec)


    def _initialize_parameters(self, param):
        """Initialize relevant parameters of atoms
        which cannot obtained from MDAnalysis module.

        Parameters
        ----------
        param : :obj:'parameter.Parameter'

        Returns
        -------
        mass_vec : float[:], shape = (num_atom)

        """
        mass_dict = param.mass_dict
        atom_name_vec = self._atom_vec.names
        mass_vec = np.array([mass_dict[i] for i in atom_name_vec])
        return(mass_vec)


    def radial_density(self, r_vec, atom_name):
        """Calculate radial density of system.
        Parameters
        ----------
        r_vec : float[:]
            radial position vector. Distance between two points should be equal.
        atom_name_list : str
            atomname for calcuating radial density.
            Ex) atom_name_list = 'OW' -> density of water oxygen atoms.
        Returns
        -------
        rad_den_mat : float[:,:], shape = (len(r_vec), 2), unit = (#/A3)
            2 columns contain r and rho(r).
        """
        if isinstance(atom_name, list):
            print("Calculate density of multiple atom species is not yet supported.")
            exit(1)

        r_min = r_vec[0]
        r_max = r_vec[-1]
        dr = r_vec[1] - r_vec[0]

        vol_vec = np.zeros_like(r_vec)
        for i, r in enumerate(r_vec):
            vol_vec[i] += 4*np.pi/3*((r+dr)**3 - (r-dr)**3)

        rad_den_mat = np.zeros((len(r_vec), 2))
        rad_den_mat[:,0] += r_vec
        for ts in tqdm(self._universe.trajectory, total=self._num_frame):
            box_vec = ts.dimensions[:3]
            pos_atom_mat = self._atom_vec.positions
            atom_name_vec = self._atom_vec.names
            atom_sel_mask = (atom_name_vec == atom_name)
            pos_sel_mat = pos_atom_mat[atom_sel_mask]
            pos_sel_mat -= self._center_of_mass(pos_sel_mat, box_vec)
            rad_dist_vec = np.linalg.norm(pos_sel_mat, axis=1)
            idx_rad_vec = np.floor(((rad_dist_vec - r_min - 0.5*dr)/dr)).astype(int)

            for j, idx_rad in enumerate(idx_rad_vec):
                rad_den_mat[idx_rad,1] += 1

        rad_den_mat[:,1] /= vol_vec * self._num_frame

        return(rad_den_mat)


    def _center_of_mass(self, pos_mat, box_vec):
        """Calculate center of mass of given position matrix

        Parameters
        ----------
        pos_mat : float[:,:], shape = (num_atoms, 3)
        box_vec : float[:], shape = 3

        Returns
        -------
        com_vec : float[:], shape = 3
        """
        pbc_pos_mat = self._check_pbc(pos_mat[0], pos_mat, box_vec)
        com_vec = np.sum(pbc_pos_mat, axis=0)
        com_vec /= len(pbc_pos_mat)
        return(com_vec)

    
    def _check_pbc(self, ref_pos_vec, pos_mat, box_vec):
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

            
    def _running_mean(self, x):
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
        return run_x
