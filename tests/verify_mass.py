from mass_fft.mass3_fft_pyfftw import *
from mass_naive.mass_naive import mass_naive,mass_naive_eu
import numpy as np
import copy


# first verify n >= m
n = 3
m = 4
try:
    mass = Mass3(n, m)
except NlessMException as e:
    print(e)
    print("** pass test for n >= m")

# n > 2 and m > 2
try:
    n = 100
    m = 2
    mass = Mass3(n, m)
except NM2Exception as e:
    print("** pass test for m > 2")

# verify if all pre-defined batch sizes are larger than or equals to 3
try:
    n = 100
    m = 40
    mass = Mass3(n, m, 2, 40, 40)
except BatchLess3Exception:
    print("** pass test for pre-defined batch size 1")
try:
    n = 100
    m = 40
    mass = Mass3(n, m, 40, 2, 40)
except BatchLess3Exception:
    print("** pass test for pre-defined batch size 2")
try:
    n = 100
    m = 40
    mass = Mass3(n, m, 40, 27, 2)
except BatchLess3Exception:
    print("** pass test for pre-defined batch size 3")



# verify execution when n < m
mass = Mass3(2 ** 15, 400)
n = 100
m = 101
data = np.random.rand(n)
query = np.random.rand(m)
r = mass.execute(data, query)
if r is None:
    print("** pass test for execution when data length smaller than query length")

# verify execution when m > batch size
mass = Mass3(2 ** 15, 400)
min_batch_size = np.array([mass.mean_batch_size, mass.fft_batch_size, mass.corr_batch_size]).min()
data = np.random.rand(min_batch_size + 500)
query = np.random.rand(min_batch_size + 1)
r = mass.execute(data, query)
r2 = mass.execute(np.random.rand(min_batch_size), np.random.rand(min_batch_size))

mass = Mass3(-1, -1, 400, 400, 400)
data = np.random.rand(400 + 500)
query = np.random.rand(400 + 1)
r3 = mass.execute(data, query)
if r is None and r2 is not None and r3 is None:
    print("** pass test for execution when query length > batch size")


#verify different batch size:
n = 100
m = 10
data = np.random.rand(n)
query = np.random.rand(m)
r_naive = mass_naive(data, query)
for b1 in range(m, 3 * n):
    print(b1)
    for b2 in range(m, 3 * n):
        for b3 in range(m, 3 * n):
            mass = Mass3(0, 0, b1, b2, b3)
            r = mass.execute(data, query)
            np.testing.assert_allclose(r, r_naive, err_msg="Error")
print("** pass test for execution with different batch sizes")



# Testing if the calculation will change the input data
n = 3000
m = 200
query = np.random.rand(m)
query1 = copy.deepcopy(query)
data = np.random.rand(n)
data1 = copy.deepcopy(data)
mass = Mass3(n, m)
r = mass.execute(data, query)
r_compare = mass_naive(data, query)
#np.testing.assert_allclose(r, r_compare, err_msg="Error", atol=1e-07)
np.testing.assert_allclose(r, r_compare, err_msg="Error")
np.testing.assert_allclose(data, data1, err_msg="Error")
np.testing.assert_allclose(query, query1, err_msg="Error")
print("** pass test for maintaining input data integrity")

# Testing euclidean distance
n = 3000
m = 200
query = np.random.rand(m)
query1 = copy.deepcopy(query)
data = np.random.rand(n)
data1 = copy.deepcopy(data)
mass = Mass3(n, m)
r = mass.execute_eu(data, query)
r_compare = mass_naive_eu(data, query)
#np.testing.assert_allclose(r, r_compare, err_msg="Error", atol=1e-07)
np.testing.assert_allclose(r, r_compare, err_msg="Error")
np.testing.assert_allclose(data, data1, err_msg="Error")
np.testing.assert_allclose(query, query1, err_msg="Error")
print("** pass test for computing the euclidean distance")

# Testing on very long time series
n = 987654321
m = 3210
query = np.random.rand(m)
data = np.random.rand(n)
mass = Mass3(n, m)
r = mass.execute(data, query)
r_compare = mass_naive(data, query)
#np.testing.assert_allclose(r, r_compare, err_msg="Error", atol=1e-07)
np.testing.assert_allclose(r, r_compare, err_msg="Error")
print("** pass test for large time series")




