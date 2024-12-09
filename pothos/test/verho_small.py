from pothos.verho import verho
import time

for i in [1]:
    start = time.time()
    vv = verho('small.dump')
    vv.compute(length_coeff=2, ncpus=i)
    end = time.time()

    total = end-start
    print(f"Verho with {i} cpus : {total} s")
