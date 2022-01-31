# MASS3_FFTW
This is a python3 implementation of MASS algorithm(3rd version in particular). For more details about MASS algorithm 
please refer to page:

[MASS: Mueen's Algorithm for Similarity Search](https://www.cs.unm.edu/~mueen/FastestSimilaritySearch.html)

This version of implementation will utilize one process. The multi-process version for multi-cores system is under
development. The Java version will also be released soon. 
---
### What's new:
- This implementation not only performs batch processing for FFT but also for calculating mean & standard deviation 
and final correlation computation. 
- This code can automatically find the optimal batch size for each environment based on various 
input sizes.
---
### Prerequisites
- [pyFFTW](https://pypi.org/project/pyFFTW/)
- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/)

Please find the installation guides from the links.

---
### How to run - Basic
There are two steps to use MASS for similarity search task. 
<ol>
<li>Initiate a Mass3 object: </li>
<p style="color:red">n should be greater than or equals to m</p>
<p style="color:red">m should be no less than 3</p>

```
from mass_fft.mass3_fft_pyfftw import Mass3
n = 5000
m = 200
mass = Mass3(n, m)
```
This process may take some time depending on the n, m. The program will automatically find the best 
batch size for the machine based on given n and m. 

<p style="color:blue">Note that if the actual data length is not available beforehand. Then it is 
ok to use a larger value of n and m than the actual data length.</p>


The value for three batch sizes can be viewed by the following code. This information is import 
for advanced execution.

```
print(mass)
```

<li>Execute the searching</li>
The random data used here for demonstration purpose. For real-world application, the input data should be a
1d numpy array. The complete searching results are saved in result array, each element in the results 
is Pearson Correlation value. 

```
search_data = np.random.rand(n)
query = np.random.rand(m)
result = mass.execute(search_data, query)
```
To compute the Euclidean distance of normalized time-series:

```
result_eu = mass.execute_eu(search_data, query)
```

</ol>

Note: Once Mass3 object initiated, it can be repeatedly used for searching and can be applied on other search_data 
and query as long as new query length is shorter or equals to m.


---
### How to run - Advanced
The second way to run MASS is initiating object by giving the specific batch sizes. This knowledge can 
either be achieved from previous basic run or prior knowledge from other sources. 
<ol>
<li>Initiate a Mass3 object: </li>

```
fft_batch_size = 128
mean_batch_size = 256
corr_batch_size = 128

mass = Mass3(-1, -1, fft_batch_size, mean_batch_size, corr_batch_size)
```
----
<p style="color:red">Requirements:</p>
<ul>
<li>All three batch sizes should be given </li>
<li>The smallest batch size among the three should be larger than or equals to the len(query)</li>
<li>The smallest batch size among the three should be larger than 2.</li>

</ul>

Runtime exception will be raised if any of the condition does not hold, or the None value will be returned 
after execution.

<li>Execute the searching</li>
Same as basic. 
<p style="color:red">len(search_data) should be no less than len(query):</p>
</ol>


