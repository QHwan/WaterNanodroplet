from __future__ import print_function, division, absolute_import

import math
import numpy as np
from numpy.core.umath_tests import inner1d
import scipy.optimize
from tqdm import tqdm
import MDAnalysis as md

from .parameter import Parameter
from .util import unit_vector

import matplotlib.pyplot as plt


class TwoPhaseThermodynamics(object):
    """Calculate entropy of system using 2PT method.
    Note: It can use only water model.
    Further modification is required.
    """

    def __init__(self, universe):
        """

        Parameters
        ----------
        universe : :obj:'MDAnalysis.core.universe.Universe'

        """
        self._universe = universe
        self._atom_vec = self._universe.select_atoms('name OW or name HW1 or name HW2')

        self._num_frame = len(self._universe.trajectory)
        self._num_atom = len(self._atom_vec)

        self._param = Parameter()
        self._mass_vec = self._initialize_parameters(self._param)
        self._mass_h2o = self._param.mass_dict['OW'] + self._param.mass_dict['HW1']*2


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


    def entropy(self, freq_vec, dos_trn_vec, dos_rot_vec,
                      temperature, volume):
        kB = 1.380649*1e-23 # (J/K)
        N_avo = 6.02214*1e23
        m = self._mass_h2o * 1e-3 / N_avo # (kg)
        h = 6.62607*1e-34 # (Js)

        N = 1
        T = temperature
        V = volume * 1e-30   # (A3 -> m3)

        I_px, I_py, I_pz = 1.344, 0.5968, 1.9408

        # Decompose dos -> dos_s and dos_g
        dos0_trn = dos_trn_vec[0]
        dos0_rot = dos_rot_vec[0]

        delta_trn = (dos0_trn*3.33565*1e-11)*(2/9./N) * \
                    (math.pi*kB*T/m)**0.5 * \
                    (N/V)**(1./3.) * \
                    (6/math.pi)**(2./3.)
        delta_rot = (dos0_rot*3.35565*1e-11)*(2/9./N) * \
                    (math.pi*kB*T/m)**0.5 * \
                    (N/V)**(1./3.) * \
                    (6/math.pi)**(2./3.)

        def func_trn(f):
            return(2*(delta_trn**-4.5)*(f**7.5) - 6*(delta_trn**-3)*(f**5) - (delta_trn**-1.5)*(f**3.5) + 6*(delta_trn**-1.5)*(f**2.5) + 2*f - 2)
        def func_rot(f):
            return(2*(delta_rot**-4.5)*(f**7.5) - 6*(delta_rot**-3)*(f**5) - (delta_rot**-1.5)*(f**3.5) + 6*(delta_rot**-1.5)*(f**2.5) + 2*f - 2)

        f_trn = scipy.optimize.brentq(func_trn, 0, 1)
        f_rot = scipy.optimize.brentq(func_rot, 0, 1)

        y_trn = (f_trn**2.5)/(delta_trn**1.5)
        z_trn = (1 + y_trn + y_trn**2 - y_trn**3)/((1 - y_trn)**3)

        y_rot = (f_rot**2.5)/(delta_rot**1.5)
        z_rot = (1 + y_rot + y_rot**2 - y_rot**3)/((1 - y_rot)**3)

        # Hard Sphere entropy S_HS/k
        S_HS_trn = 2.5 + math.log(((2*math.pi*m*kB*T/h/h)**1.5)*V/f_trn/N*z_trn) + y_trn*(3*y_trn-4)/((1-y_trn)**2)

        TA = (h**2)/(8*(math.pi**2)*kB*I_px*1e-18*1e-3/N_avo)
        TB = (h**2)/(8*(math.pi**2)*kB*I_py*1e-18*1e-3/N_avo)
        TC = (h**2)/(8*(math.pi**2)*kB*I_pz*1e-18*1e-3/N_avo)
        S_HS_rot = math.log((((math.pi**0.5)*(math.exp(1)**1.5))/(3))*(((T**3)/(TA*TB*TC))**0.5))

        dos_trn_gas_vec = dos0_trn / (1 + (math.pi*dos0_trn*freq_vec/6/f_trn/N)**2)
        dos_trn_sol_vec = dos_trn_vec - dos_trn_gas_vec
        dos_rot_gas_vec = dos0_rot / (1 + (math.pi*dos0_rot*freq_vec/6/f_rot/N)**2)
        dos_rot_sol_vec = dos_rot_vec - dos_rot_gas_vec


        # Setting bhv
        #bhv_vec = 2.9979*1e10*freq_vec*h/kB/T
        bhv_vec = freq_vec*h/kB/T/3.33565/1e-11

        # Calculate Entropy
        W_trn_sol_vec = np.zeros(len(freq_vec))
        W_trn_gas_vec = np.zeros(len(freq_vec))
        W_rot_sol_vec = np.zeros(len(freq_vec))
        W_rot_gas_vec = np.zeros(len(freq_vec))

        for i, bhv in enumerate(bhv_vec):
            if i != 0:
                W_trn_sol_vec[i] = bhv/(math.exp(bhv)-1) - math.log(1-math.exp(-bhv))
                W_rot_sol_vec[i] = bhv/(math.exp(bhv)-1) - math.log(1-math.exp(-bhv))
            W_trn_gas_vec[i] = 1./3.*S_HS_trn
            W_rot_gas_vec[i] = 1./3.*S_HS_rot

        S_trn_sol = np.trapz(dos_trn_sol_vec * W_trn_sol_vec, freq_vec) * kB * N_avo/N
        S_trn_gas = np.trapz(dos_trn_gas_vec * W_trn_gas_vec, freq_vec) * kB * N_avo/N
        S_trn = S_trn_sol + S_trn_gas

        S_rot_sol = np.trapz(dos_rot_sol_vec * W_rot_sol_vec, freq_vec) * kB * N_avo/N
        S_rot_gas = np.trapz(dos_rot_gas_vec * W_rot_gas_vec, freq_vec) * kB * N_avo/N
        S_rot = S_rot_sol + S_rot_gas


        return(S_trn, S_rot)

        '''
        ref = np.loadtxt('w.2pt.pwr')
        plt.plot(ref[:,0], ref[:,3]/512, 'o', markersize=2)
        plt.plot(ref[:,0], ref[:,4]/512, 'o', markersize=2)
        plt.plot(ref[:,0], ref[:,6]/512, 'o', markersize=2)

        plt.plot(freq_vec, dos_rot_gas_vec, linewidth=2)
        plt.plot(freq_vec, dos_rot_sol_vec, linewidth=2)
        plt.plot(freq_vec, dos_rot_vec, linewidth=2)
        plt.xlim((0, 1200))
        plt.show()

        exit(1)
        '''



        




    def density_of_state(self, t_vec, vel_corr_vec, temperature):
        """Calculate density of states(DOS) from velocity correlation.

        Parameters
        ----------
        t_vec : float[:]
        vel_corr_vec : float[:]
        temperature : float

        Returns
        -------
        dos_mat : float[:,:]
            columns -> freq, dos
        """
        dt = t_vec[1] - t_vec[0]
        n = len(t_vec)

        t_mirror_vec = np.zeros(2*n-1)
        vel_corr_mirror_vec = np.zeros_like(t_mirror_vec)
        for i in range(n):
            t_mirror_vec[i] += -1*t_vec[n-1-i]
            t_mirror_vec[n+i-1] += t_vec[i]
            vel_corr_mirror_vec[i] += vel_corr_vec[n-1-i]
            if i != 0:
                vel_corr_mirror_vec[n+i-1] += vel_corr_vec[i]

        # ps to cm-1
        # f = 1/(N*t), 1 Hz = 3.33565*1e-11 cm-1
        freq_vec = 0.5 * (np.arange(n)*3.33565*1e-11)/(n*dt*1e-12)

        # mass and velocity conversion
        vel_corr_mirror_vec = np.fft.fft(vel_corr_mirror_vec)
        # cm unit conversion
        vel_corr_mirror_vec *= 2/temperature/(1.38*1e-23)*(1/6.02*1e-26)*1e-8
        #vel_corr_mirror_vec /= (3.33565*1e-11)
        vel_corr_mirror_vec *= 3.003*1e10
        # Caution! check it
        vel_corr_mirror_vec /= 250

        freq_vec = freq_vec[range(int(n/2))]
        vel_corr_mirror_vec = vel_corr_mirror_vec[range(int(n/2))]
        vel_corr_mirror_vec = np.abs(np.abs(vel_corr_mirror_vec))

        dos_mat = np.array([freq_vec,
                            vel_corr_mirror_vec]).T
        
        return(dos_mat)


    def velocity_correlation(self, t_i, t_f, t_c=0.5):
        """Calculate translational and rotation velocity correlaiton.
        
        Parameters
        ----------
        t_i : float
        t_f : float
            t_i, t_f -> Start, end time of trajectory
        t_c : float
            Correlation time.

        Returns
        -------
        t_vec : float[:], shape = len(corr)
        trn_corr_mat : float[:,:], shape = (len(corr), num_mol)
        rot_corr_mat : float[:,:], shape = (len(corr), num_mol)
        """
        if t_c*2 > t_f - t_i:
            print("Correlation time is too long. Maximum: half of trajectory.")
            exit(1)

        dt = self._universe.trajectory[1].time - self._universe.trajectory[0].time
        t_vec = np.arange(0, t_c*1.0001, dt)
        frame_i = int(t_i/dt)
        frame_f = int(t_f/dt)
        num_frame = frame_f - frame_i + 1

        num_mol = int(self._num_atom/3)
        vel_trn_mat3 = np.zeros((num_frame, num_mol, 3))
        vel_rot_mat3 = np.zeros_like(vel_trn_mat3)

        trn_corr_mat = np.zeros((len(t_vec), num_mol))
        rot_corr_mat = np.zeros_like(trn_corr_mat)

        print(frame_i*dt, frame_f*dt)
        for i in tqdm(range(num_frame)):
        #for i, ts in tqdm(enumerate(self._universe.trajectory), total=self._num_frame):
            ts = self._universe.trajectory[frame_i + i]
            box_vec = ts.dimensions
            pos_atom_mat = self._atom_vec.positions
            vel_atom_mat = self._atom_vec.velocities
            vel_trn_mat3[i], vel_rot_mat3[i], I_mat = self._decompose_velocity(pos_atom_mat, vel_atom_mat, box_vec)

        for i in tqdm(range(len(t_vec))):
            vel_trn_0 = vel_trn_mat3[0:-1-i].reshape((-1,3))
            vel_trn_t = vel_trn_mat3[i:-1].reshape((-1,3))
            trn_corr_mat[i] = self._mass_h2o*np.mean(inner1d(vel_trn_0, vel_trn_t).reshape((-1, num_mol)), axis=0)

        I_vec = np.array([I_mat[0,0], I_mat[1,1], I_mat[2,2]])
        for i in tqdm(range(len(t_vec))):
            vel_rot_0 = I_vec*vel_rot_mat3[0:-1-i].reshape((-1,3))
            vel_rot_t = vel_rot_mat3[i:-1].reshape((-1,3))
            rot_corr_mat[i] = np.mean(inner1d(vel_rot_0, vel_rot_t).reshape((-1,num_mol)), axis=0)

        return(t_vec, trn_corr_mat, rot_corr_mat)


    def _decompose_velocity(self, pos_atom_mat,
                                  vel_atom_mat,
                                  box_vec,
                                  translation=True,
                                  rotation=True):
        """Decompose velocity of molecule
        into translational, rotational, and vibrational velocity.
        Note: vibration part is not yet built. Future goal.

        Parameters
        ----------
        pos_atom_mat : float[:,:], shape = (num_atom, 3)
        vel_atom_mat : float[:,:], shape = (num_atom, 3)
        box_vec : float[:], shape = 3
        translation : bool
        rotation : bool

        Returns
        -------
        vel_trn_mat (optional) : float[:,:], shape = (num_sol, 3)
        vel_rot_mat (optional) : float[:,:], shape = (num_sol, 3)
        """
        if translation:
            vel_trn_mat = self._translation_velocity(vel_atom_mat)
        if rotation:
            vel_rot_mat, I_mat = self._rotation_velocity(pos_atom_mat, 
                                                  vel_atom_mat,
                                                  box_vec)

        return(vel_trn_mat, vel_rot_mat, I_mat)


    def _translation_velocity(self, vel_atom_mat):
        """Calculate translational velocity of molecule.

        Parameters
        ----------
        vel_atom_mat : float[:,:], shape = (num_atom, 3)

        Returns
        -------
        vel_trn_mat : float[:,:], shape = (num_mol, 3)
        """
        num_atom = len(vel_atom_mat)
        num_mol = int(num_atom/3)
        vel_trn_mat = np.sum((vel_atom_mat*self._mass_vec.reshape((-1,1))).reshape((num_mol, 3, 3)), axis=1)
        vel_trn_mat /= self._mass_h2o
        return(vel_trn_mat)


    def _rotation_velocity(self, pos_atom_mat, vel_atom_mat, box_vec):
        """Calculate rotational velocity of molecule.
        Note: calculation of translational velocity should be preceded.

        Parameters
        ----------
        pos_atom_mat : float[:,:], shape = (num_atom, 3)
        vel_atom_mat : float[:,:], shape = (num_atom, 3)
        box_vec : float[:], shape = 3

        Returns
        -------
        vel_rot_mat : float[:,:], shape = (num_mol, 3)
            It containes angular velocity, w along three principal axes.
        """
        num_atom = len(pos_atom_mat)
        num_mol = int(num_atom/3)

        #pbc_pos_atom_mat = util.check_pbc(pos_atom_mat[0], pos_atom_mat, box_vec)
        pbc_pos_atom_mat = pos_atom_mat
        pos_com_mat = np.sum((pbc_pos_atom_mat*self._mass_vec.reshape((-1,1))).reshape((num_mol,3,3)), axis=1)
        pos_com_mat /= self._mass_h2o

        # retract pos_com
        rel_pos_atom_mat = pbc_pos_atom_mat - np.repeat(pos_com_mat, repeats=3, axis=0)

        # construct frame_rot_mat
        frame_rot_mat3 = np.zeros((num_mol, 3, 3))
        # x->dipolar axis, y->h2-h1, z->x x y
        frame_rot_mat3[:,0] = unit_vector(-2*rel_pos_atom_mat[::3] +
                                                  rel_pos_atom_mat[1::3] +
                                                  rel_pos_atom_mat[2::3])
        frame_rot_mat3[:,1] = unit_vector(rel_pos_atom_mat[2::3] - rel_pos_atom_mat[1::3])
        frame_rot_mat3[:,2] = np.cross(frame_rot_mat3[:,0], frame_rot_mat3[:,1])

        
        # rotate frame: R -> r (lab -> molecular)
        new_pos_atom_mat = np.zeros((num_atom, 3))
        new_vel_atom_mat = np.zeros((num_atom, 3))
        for j in range(3):
            new_pos_atom_mat[:,j] = inner1d(np.repeat(frame_rot_mat3,3,axis=0)[:,j], rel_pos_atom_mat)
            new_vel_atom_mat[:,j] = inner1d(np.repeat(frame_rot_mat3,3,axis=0)[:,j], vel_atom_mat)
        
        # inertia moment vector
        I_mat = np.zeros((3,3))
        I_mat[0,0] = np.sum((new_pos_atom_mat[:3,1]**2 + new_pos_atom_mat[:3,2]**2)*self._mass_vec[:3])
        I_mat[1,1] = np.sum((new_pos_atom_mat[:3,0]**2 + new_pos_atom_mat[:3,2]**2)*self._mass_vec[:3])
        I_mat[2,2] = np.sum((new_pos_atom_mat[:3,0]**2 + new_pos_atom_mat[:3,1]**2)*self._mass_vec[:3])

        I_inv_mat = np.linalg.inv(I_mat)
        
        # ww : L = m(r x v) = Iw
        L_mat = np.zeros((num_mol, 3))
        L_mat += self._param.mass_dict['OW'] * np.cross(new_pos_atom_mat[::3],
                                                        new_vel_atom_mat[::3])
        L_mat += self._param.mass_dict['HW1'] * np.cross(new_pos_atom_mat[1::3],
                                                         new_vel_atom_mat[1::3])
        L_mat += self._param.mass_dict['HW1'] * np.cross(new_pos_atom_mat[2::3],
                                                         new_vel_atom_mat[2::3])

        # finally vel_rot_mat
        vel_rot_mat = np.matmul(I_inv_mat, L_mat.T).T

        return(vel_rot_mat, I_mat)



#I_mat = np.sum((new_pos_atom_mat**2*self._mass_vec.reshape(-1,1)).reshape((num_mol,3,3)), axis=1)
