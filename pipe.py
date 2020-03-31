import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as md

from density import Density
from potential import Potential


u = md.Universe('trj/md512_280k.tpr',
                'trj/md512_280k_100frame.xtc')
r = Potential(u)

r_vec = np.linspace(0.5, 100.5, 501)
rad_den_mat = r._potential_matrix()

plt.plot(rad_den_mat[:,0], rad_den_mat[:,1], '-o')
plt.show()
