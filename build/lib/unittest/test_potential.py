import numpy as np
import MDAnalysis as md

from WaterNanodroplet import potential


u = md.Universe('trj/md3.tpr',
                'trj/md3.gro')
p = potential.Potential(u)
pos_atom_mat = p._atom_vec.positions
print(pos_atom_mat)