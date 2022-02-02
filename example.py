from mass_fft.mass3_fft_pyfftw import Mass3
import numpy as np

# basic
n = 5000
m = 200
mass = Mass3(n, m)
print(mass)
search_data = np.random.rand(n)
query = np.random.rand(m)
result_basic = mass.execute(search_data, query)

# Initiate Mass3 based on the printed results
mass = Mass3(-1, -1, 4096, 4096, 8192)
result_advance = mass.execute(search_data, query)