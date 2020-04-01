import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as md

from density import Density
from two_phase import TwoPhaseThermodynamics


u = md.Universe('trj/2pt0.tpr',
                'trj/2pt0.trr')

h2o = u.select_atoms('name OW or name HW1 or name HW2')
ts = u.trajectory[0]
vel = h2o.velocities
v = TwoPhaseThermodynamics(u)
vel_corr_mat = v.velocity_correlation()
dos = v.density_of_state(vel_corr_mat[:,0],
                         vel_corr_mat[:,1]/32,
                         temperature=280)

plt.plot(dos[:,0], dos[:,1], '-o')
#plt.plot(vel_corr_mat[:,0], vel_corr_mat[:,1]/32, '-o')
#plt.plot(vel_corr_mat[:,0], vel_corr_mat[:,2], '-o')
plt.show()
