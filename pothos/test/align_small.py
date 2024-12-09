from pothos.align import align
import matplotlib.pyplot as plt
import time
import numpy as np

filename = 'small.dump'

for i in [1]:
    pa = align(filename)

    leafs = 13

    pa.ncpus=i
    pa.series=True
    # pa.atom_types=[] # use for atomistic or selecting backbone atoms

    pa.align_type = 'P2'
    pa.length_coeff = 2

    pa.legendre_cutoff = 0.95

    pa.method = 'BallTree'
    pa.eps = 1.2
    pa.lam = 0.985
    pa.leaf_size = leafs
    pa.min_pts = 2

    pa.min_length = 2
    pa.second_filt = True
    pa.window = 'parzen'

    pa.crystalmin = 25
    pa.coloring = 'herman'

    pa.find()

    out = pa.series_stats

import numpy as np
import matplotlib.pyplot as plt

f = []
cost = []

for _out in out:
    _cost =  (_out[1]*0. +  _out[2]*0. +  _out[3]*1.)/(np.sqrt(_out[1]**2 + _out[2]**2 + _out[3]**2) * 1)
    _f = 3/2*_cost**2 -1/2

    f.append(_f)
    cost.append(_cost)

file = open(f'small_vectors.txt', 'w+')
file.write(f'i vx vy vz f \n')
for i, _out in enumerate(out):
    file.write(f'{i} {_out[0]} {_out[1]} {_out[2]} {_out[3]} {f[i]} \n')
file.close()
