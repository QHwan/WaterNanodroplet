import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as md

from raidal_density import RadialDensity


u = md.Universe('trj/md512_280k.tpr',
                'trj/md512_280k_100frame.xtc')
r = RadialDensity(u)

r_vec = np.linspace(0, 100, 1)
rad_den_mat = r.radial_density(r_vec, 'OW')

plt.plot(rad_den_mat[:,0])
plt.show()
