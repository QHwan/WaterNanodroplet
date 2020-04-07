import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as md

from density import Density
from two_phase import TwoPhaseThermodynamics


u = md.Universe('2pt.tpr',
                '2pt_pbc.trr')

h2o = u.select_atoms('name OW or name HW1 or name HW2')
ts = u.trajectory[0]
vel = h2o.velocities
v = TwoPhaseThermodynamics(u)
t, trn, rot = v.velocity_correlation(t_i=1, t_f=2, t_c=0.5)

#print(trn[0])

ref = np.loadtxt('w.2pt.vac')

#plt.errorbar(t, np.mean(rot, axis=1), yerr=np.std(rot, axis=1))
plt.plot(t, np.mean(rot, axis=1), 'o')
plt.plot(ref[:,0], ref[:,2]/512, '-')
plt.xlim((0, 0.2))
plt.show()
