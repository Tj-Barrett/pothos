from pothos.legendre import legendre
import time

for i in [1]:
    start = time.time()
    vv = legendre('small.dump')
    vv.compute(align_type='P2',length_coeff=4, ncpus=i, series=True)#, atom_types=[1])
    vv.stats()
    end = time.time()

    total = end-start
    print(f"Legendre with {i} cpus : {total} s")
