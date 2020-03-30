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


    def radial_density(self, r_vec, atom_name_list):
        """Calculate radial density of system.
        Parameters
        ----------
        r_vec : float[:]
            radial position vector. Distance between two points should be equal.
        atom_name_list : str or [str]
            atomnames for calcuating radial density.
            Ex) atom_name_list = 'OW' -> density of water oxygen atoms.
                atom_name_list = ['OW', 'HW1', 'HW2'] -> density of water molecules.
        Returns
        -------
        rad_den_mat : float[:,:], shape = (len(r_vec), 2), unit = (#/A3)
            2 columns contain r and rho(r).
        """
        rad_den_mat = np.zeros((len(r_vec), 2))
        return(rad_den_mat)

            
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
