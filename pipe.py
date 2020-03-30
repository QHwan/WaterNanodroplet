import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as md


u = md.Universe('trj/md_300k.tpr',
                'trj/md_300k_100frame_pbc.xtc')
d = Dipole(u)

di_vec = d.static_dielectric_constant()
Mtot = np.loadtxt('trj/Mtot.xvg')
eps = np.loadtxt('trj/eps.xvg')

print(di_vec)

plt.plot(di_vec)
plt.plot(eps[:,1], 'o')
plt.show()
