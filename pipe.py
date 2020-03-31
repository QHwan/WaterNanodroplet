import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as md

from density import Density


u = md.Universe('trj/md512_280k.tpr',
                'trj/md512_280k_100frame.xtc')
r = Density(u)

r_vec = np.linspace(0.5, 100.5, 501)
rad_den_mat = r.radial_density(r_vec, 'OW')

plt.plot(rad_den_mat[:,0], rad_den_mat[:,1], '-o')
plt.show()
