from numba import jit, cuda
import numpy as np
# to measure exec time
from timeit import default_timer as timer   

# normal function to run on cpu
def func(a):                                
    for i in range(10000000):
        a[i]+= 1      

# GPU kernel version
@cuda.jit
def func2(a):
    for i in range(10000000):
        a[i]+= 1
if __name__=="__main__":
    n = 10000000                            
    a = np.ones(n, dtype = np.float64)
    
    start = timer()
    func(a)
    print("without GPU:", timer()-start)

    # Allocate on GPU
    d_a = cuda.to_device(a)

    # Specify thread count and block count for GPU
    threadsperblock = 32
    blockspergrid = (a.size + (threadsperblock - 1)) # threadsperblock
    
    start = timer()
    func2[blockspergrid, threadsperblock](d_a)
    print("with GPU:", timer()-start)