import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as md

from two_phase import TwoPhaseThermodynamics
from util import check_pbc, center_of_mass


u = md.Universe('2pt128_280k.tpr',
                '2pt128_280k.trr')
ow = u.select_atoms('name OW')

r_vec = np.arange(-0.25, 30.25+0.01, 0.5)
r_min = r_vec[0]
dr = r_vec[1]-r_vec[0]
dt = u.trajectory[1].time - u.trajectory[0].time
t_c = 1
t_vec = np.arange(0, t_c+0.001*dt, dt)

count_vec = np.zeros_like(r_vec)
rad_trn_mat = np.zeros((len(r_vec), len(t_vec)))
rad_rot_mat = np.zeros_like(rad_trn_mat)

num_frame = len(u.trajectory)
dframe = int(t_c*2/dt)+1
total_step = int(num_frame/dframe)

v = TwoPhaseThermodynamics(u)
for i in range(total_step):
    frame_i = i*dframe
    if frame_i + dframe >= num_frame:
        continue

    _, trn, rot = v.velocity_correlation(t_i=t_c*2*i, t_f=t_c*2*(i+1), t_c=1)

    ts = u.trajectory[frame_i]
    box_vec = ts.dimensions[:3]
    pos_ow_mat = ow.positions
    pbc_pos_ow_mat = check_pbc(pos_ow_mat[0], pos_ow_mat, box_vec)
    pos_ow_mat -= center_of_mass(pbc_pos_ow_mat)
    rad_pos_vec = np.linalg.norm(pos_ow_mat, axis=1)
    idx_rad_vec = np.floor(((rad_pos_vec - r_min)/dr)).astype(int)

    for j, idx_rad in enumerate(idx_rad_vec):
        if idx_rad >= len(r_vec):
            continue
        count_vec[idx_rad] += 1
        rad_trn_mat[idx_rad] += trn[:,j]
        rad_rot_mat[idx_rad] += rot[:,j]

for i, count in enumerate(count_vec):
    if count == 0:
        continue

    rad_trn_mat[i] /= count
    rad_rot_mat[i] /= count

np.savez('rad_2pt.npz', r=r_vec, t=t_vec, trn=rad_trn_mat, rot=rad_rot_mat)



'''
plt.plot(t, np.mean(rot, axis=1), 'o')
plt.plot(ref[:,0], ref[:,2]/512, '-')
plt.xlim((0, 0.2))
plt.show()
'''
