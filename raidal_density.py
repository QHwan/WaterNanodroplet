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
        charge_vec : float[:], shape = (num_atom)

        """
        charge_dict = param.charge_dict
        atom_name_vec = self._atom_vec.names
        charge_vec = np.array([charge_dict[i] for i in atom_name_vec])
        return(charge_vec)


    def total_dipole(self):
        """Calculate total amount of dipole of system.
        
        Returns
        -------
        tot_dip_mat : float[:,:], shape = (num_frame, 4), unit = (eA)
            4 columns contain x-, y-, z- direction and total.
        """
        tot_dip_mat = np.zeros((self._num_frame, 4))
        for i, _ in tqdm(enumerate(self._universe.trajectory), total=self._num_frame):
            pos_atom_mat = self._atom_vec.positions
            tot_dip_mat[i,:3] = np.sum(pos_atom_mat * self._charge_vec.reshape(-1,1), axis=0) # broadcasting along axis = 1
        tot_dip_mat[:,3] = np.linalg.norm(tot_dip_mat[:,:3], axis=1)
        return(tot_dip_mat)


    def static_dielectric_constant(self):
        """Calculate relative static dielectric constant (w=0)
        Ref: Mol. Phys. 50, 841-858 (1983)

        Returns
        -------
        dielec_const_vec : float[:], shape = (num_frame)
            relative dielectric constant.
        """
        const = Constant()
        tot_dip_mat = self.total_dipole()
        run_avg_dip_vec = self._running_mean(np.sum(tot_dip_mat[:,:3], axis=1))
        run_avg_sqr_dip_vec = self._running_mean(np.sum(tot_dip_mat[:,:3]**2, axis=1))

        vol_vec = self._box_mat[:,0]*self._box_mat[:,1]*self._box_mat[:,2]

        dielec_const_vec = np.zeros(self._num_frame)
        dielec_const_vec.fill(4*np.pi/3)
        dielec_const_vec *= run_avg_sqr_dip_vec - run_avg_dip_vec**2
        dielec_const_vec /= vol_vec*const.kB*300
        dielec_const_vec /= const.eps0*1e-10/(1.602*1.602*1e-38)
        dielec_const_vec += 1
        return(dielec_const_vec)

            
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
