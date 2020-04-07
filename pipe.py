import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import MDAnalysis as md

from density import Density
from potential import Potential


u = md.Universe('trj/md128_280k.tpr',
                'trj/md128_280k.xtc')

h2o = u.select_atoms('name OW or name HW1 or name HW2')
pot = Potential(u)

for _ in tqdm(u.trajectory, total=len(u_trajectory)):
    pot.potential_matrix()


'''
plt.plot(t, np.mean(rot, axis=1), 'o')
plt.plot(ref[:,0], ref[:,2]/512, '-')
plt.xlim((0, 0.2))
plt.show()
'''
