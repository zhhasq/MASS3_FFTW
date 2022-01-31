import numpy as np
import pyfftw
from timeit import Timer
import mass_fft.mass_utils as utils
import scipy.ndimage.filters as ndif



class NM2Exception(Exception):
    def __init__(self):
        self.msg = "Exception: n and m should be no less than 3"
        super().__init__(self.msg)

    def __str__(self):
        return self.msg


class NlessMException(Exception):
    def __init__(self):
        self.msg = "Exception: n should not be less than m"
        super().__init__(self.msg)

    def __str__(self):
        return self.msg


class MissBatchSizeException(Exception):
    def __init__(self):
        self.msg = "Exception: All three batch size should be given"
        super().__init__(self.msg)

    def __str__(self):
        return self.msg


class BatchLess3Exception(Exception):
    def __init__(self):
        self.msg = "Exception: All three batch size should be larger than 2"
        super().__init__(self.msg)

    def __str__(self):
        return self.msg


class Mass3Func:
    @staticmethod
    def _pyfftw_dot_batch(m, ts, fftw_forw_ts, fftw_back_obj, fftw_in_ts, fftw_out_ts, fftw_out_query, fftw_in_results):
        n = len(ts)
        fftw_in_ts[:n] = ts
        fftw_in_ts[n:] = 0
        fftw_forw_ts.execute()
        fftw_in_results[:] = fftw_out_ts * fftw_out_query
        fftw_back_obj(normalise_idft=True)
        return fftw_back_obj.output_array[m - 1: n]

    @staticmethod
    def _mean_std2_batch(input_data, left, right, w):
        a = np.array(input_data[left: right], dtype=np.double)
        input2 = a ** 2

        n = len(a)
        num_r = n - w + 1

        start = int(np.floor(w / 2))
        end = start + num_r - 1
        mean = ndif.uniform_filter1d(a, w)[start: end + 1]
        std2 = ndif.uniform_filter1d(input2, w)[start: end + 1] - mean ** 2
        return mean, std2

    @staticmethod
    def find_optimal_batch_size(n, m):
        if n <= 2 ** 15:
            fft_sizes = []
            mean_sizes = []
            corr_sizes = []
            next_power = utils.next_power_of_2(n)
            for i in range(next_power, 16):
                n = 2 ** i
                a, b, c = Mass3Func._find_optimal_batch_size_helper(n, m)
                fft_sizes.append(a)
                mean_sizes.append(b)
                corr_sizes.append(c)
            fft_sizes = sorted(fft_sizes, key=lambda x: x[1])
            mean_sizes = sorted(mean_sizes, key=lambda x: x[1])
            corr_sizes = sorted(corr_sizes, key=lambda x: x[1])
            return fft_sizes[0][0], mean_sizes[0][0], corr_sizes[0][0]
        elif n <= 2 ** 22:
            a, b, c = Mass3Func._find_optimal_batch_size_helper(n, m)
            return a[0], b[0], c[0]
        else:
            a, b, c = Mass3Func._find_optimal_batch_size_helper(2 ** 22, m)
            return a[0], b[0], c[0]

    @staticmethod
    def _find_optimal_batch_size_helper(n=2 ** 20, m=2 ** 9):
        # 2 ^ 20 = 1048576
        # 2 ^ 9 = 512

        next_power = utils.next_power_of_2(n)
        low = utils.next_power_of_2(m)
        n = 2 ** next_power

        best_fft_sizes = []
        best_mean_sizes = []
        best_corr_sizes = []

        for k in range(3):
            batch_sizes = []
            dot_product_time = []
            mean_std2_time = []
            compute_corr_time = []

            for i in range(low, next_power + 1):
                batch_size = 2 ** i
                num_round = n / batch_size
                num_round = 5 * num_round
                batch_sizes.append(batch_size)
                # print(f"{i}, batch size {batch_size}, round:{num_round}")

                x = np.random.rand(batch_size)
                y = np.random.rand(m)

                fft_input_len = batch_size
                fft_output_len = int(fft_input_len / 2) + 1
                fftw_in_ts = pyfftw.zeros_aligned(fft_input_len, dtype='float64')
                fftw_out_ts = pyfftw.zeros_aligned(fft_output_len, dtype='complex128')
                fftw_in_query = pyfftw.zeros_aligned(fft_input_len, dtype='float64')
                fftw_out_query = pyfftw.zeros_aligned(fft_output_len, dtype='complex128')

                fftw_forward_ts = pyfftw.FFTW(fftw_in_ts, fftw_out_ts, flags=["FFTW_ESTIMATE"],
                                              direction='FFTW_FORWARD')
                fftw_forward_query = pyfftw.FFTW(fftw_in_query, fftw_out_query, flags=["FFTW_ESTIMATE"],
                                                 direction='FFTW_FORWARD')

                fftw_out_results = pyfftw.zeros_aligned(fft_input_len, dtype='float64')
                fftw_in_results = pyfftw.zeros_aligned(fft_output_len, dtype='complex128')
                fftw_backward_results = pyfftw.FFTW(fftw_in_results, fftw_out_results, flags=["FFTW_ESTIMATE"],
                                                    direction='FFTW_BACKWARD')
                fftw_in_query[:m] = np.flip(y)
                fftw_forward_query.execute()

                # Testing the batch size for performing inner product based on pyfftw
                t = Timer(
                    lambda: Mass3Func._pyfftw_dot_batch(batch_size, x, fftw_forward_ts, fftw_backward_results,
                                                        fftw_in_ts,
                                                        fftw_out_ts,
                                                        fftw_out_query, fftw_in_results))
                time_dot = t.timeit(number=int(num_round))
                # print('inner product time: %1.8f seconds' % time_dot)
                dot_product_time.append(time_dot)

                # Testing the batch size for performing the mean and std2
                # t_mean_std = Timer(lambda: utils.mov_mean_std(x, m))
                t_mean_std = Timer(lambda: Mass3Func._mean_std2_batch(x, 0, len(x), m))
                time_mean_std = t_mean_std.timeit(number=int(num_round))
                # print('mean,std2 compute time: %1.8f seconds' % time_mean_std)
                mean_std2_time.append(time_mean_std)

                # Testing the batch size for computing the correlation
                dot = Mass3Func._pyfftw_dot_batch(batch_size, x, fftw_forward_ts, fftw_backward_results, fftw_in_ts,
                                                  fftw_out_ts,
                                                  fftw_out_query, fftw_in_results)
                mean_ts, std2_ts = utils.mov_mean_std(x, m)
                mean_query = y.mean()
                std2_query = y.std()
                timer_corr = Timer(lambda: utils.p_corr(dot, m, mean_ts, mean_query, std2_ts, std2_query))
                time_corr = timer_corr.timeit(number=int(num_round))
                # print('corr compute time: %1.8f seconds' % time_corr)
                compute_corr_time.append(time_corr)

            dot_product_time = np.array(dot_product_time)
            mean_std2_time = np.array(mean_std2_time)
            compute_corr_time = np.array(compute_corr_time)

            best_fft_sizes.append((batch_sizes[np.argmin(dot_product_time)], dot_product_time.min()))
            best_mean_sizes.append((batch_sizes[np.argmin(mean_std2_time)], mean_std2_time.min()))
            best_corr_sizes.append((batch_sizes[np.argmin(compute_corr_time)], compute_corr_time.min()))

        best_fft_sizes = sorted(best_fft_sizes, key=lambda x: x[0])
        best_mean_sizes = sorted(best_mean_sizes, key=lambda x: x[0])
        best_corr_sizes = sorted(best_corr_sizes, key=lambda x: x[0])

        return best_fft_sizes[1], best_mean_sizes[1], best_corr_sizes[1]

    @staticmethod
    def mass_batch_eu(data, query, best_batch_size_fft, best_compute_size, best_corr_size, fftw_forward_ts,
                      fftw_backward_results, fftw_in_ts, fftw_out_ts, fftw_out_query, fftw_in_results):
        n = len(data)
        m = len(query)

        dot_results = pyfftw.zeros_aligned(n - m + 1, dtype='float64')
        mean_ts = pyfftw.zeros_aligned(n - m + 1, dtype='float64')
        std2_ts = pyfftw.zeros_aligned(n - m + 1, dtype='float64')
        eu_result = pyfftw.zeros_aligned(n - m + 1, dtype='float64')

        # compute dot product
        left = 0
        i = 0
        right = left + best_batch_size_fft
        while i <= n - m:
            z = Mass3Func._pyfftw_dot_batch(m, data[left: right], fftw_forward_ts, fftw_backward_results,
                                            fftw_in_ts, fftw_out_ts,
                                            fftw_out_query, fftw_in_results)
            dot_results[i: i + len(z)] = z
            i = i + len(z)
            left = right - m + 1
            right = left + best_batch_size_fft

        # compute mean, std2
        left = 0
        i = 0
        while i <= n - m:
            right = left + best_compute_size
            if right > len(data):
                right = len(data)
            cur_mean, cur_std = Mass3Func._mean_std2_batch(data, left, right, m)
            size = len(cur_mean)
            mean_ts[i: i + size] = cur_mean
            std2_ts[i: i + size] = cur_std
            i = i + size
            left = right - m + 1

        # compute corr
        left = 0
        right = left + best_corr_size
        mean_query = query.mean()
        std2_query = query.std() ** 2
        size = len(dot_results)
        stop = False
        while not stop:
            if right > size:
                right = size
                stop = True
            cur_corr = utils.p_corr(dot_results[left: right], m, mean_ts[left:right], mean_query,
                                    std2_ts[left:right], std2_query)
            corr2 = 2 * m * (1 - cur_corr)
            corr2[corr2 < 0] = 0
            eu_dist = np.sqrt(corr2)
            eu_result[left: right] = eu_dist
            left = right
            right = left + best_corr_size
        return eu_result

    @staticmethod
    def mass_batch(data, query, best_batch_size_fft, best_compute_size, best_corr_size, fftw_forward_ts,
                   fftw_backward_results, fftw_in_ts, fftw_out_ts, fftw_out_query, fftw_in_results):
        n = len(data)
        m = len(query)

        dot_results = pyfftw.zeros_aligned(n - m + 1, dtype='float64')
        mean_ts = pyfftw.zeros_aligned(n - m + 1, dtype='float64')
        std2_ts = pyfftw.zeros_aligned(n - m + 1, dtype='float64')
        corr_result = pyfftw.zeros_aligned(n - m + 1, dtype='float64')

        # compute dot product
        left = 0
        i = 0
        right = left + best_batch_size_fft
        while i <= n - m:
            z = Mass3Func._pyfftw_dot_batch(m, data[left: right], fftw_forward_ts, fftw_backward_results, fftw_in_ts,
                                            fftw_out_ts,
                                            fftw_out_query, fftw_in_results)
            dot_results[i: i + len(z)] = z
            i = i + len(z)
            left = right - m + 1
            right = left + best_batch_size_fft

        # compute mean, std2
        left = 0
        i = 0
        while i <= n - m:
            right = left + best_compute_size
            if right > len(data):
                right = len(data)
            cur_mean, cur_std = Mass3Func._mean_std2_batch(data, left, right, m)
            size = len(cur_mean)
            mean_ts[i: i + size] = cur_mean
            std2_ts[i: i + size] = cur_std
            i = i + size
            left = right - m + 1

        # compute corr
        left = 0
        right = left + best_corr_size
        mean_query = query.mean()
        std2_query = query.std() ** 2
        size = len(dot_results)
        stop = False
        while not stop:
            if right > size:
                right = size
                stop = True
            cur_corr = utils.p_corr(dot_results[left: right], m, mean_ts[left:right], mean_query, std2_ts[left:right],
                                    std2_query)
            corr_result[left: right] = cur_corr
            left = right
            right = left + best_corr_size
        return corr_result


class Mass3:
    def __init__(self, n, m, fft_batch_size=None, mean_batch_size=None, corr_batch_size=None):
        if fft_batch_size is not None and mean_batch_size is not None and corr_batch_size is not None:
            if fft_batch_size < 3 or mean_batch_size < 3 or corr_batch_size < 3:
                raise BatchLess3Exception()
            self.fft_batch_size = fft_batch_size
            self.mean_batch_size = mean_batch_size
            self.corr_batch_size = corr_batch_size
        else:
            if n < m:
                raise NlessMException()
            if n <= 2 or m <= 2:
                raise NM2Exception()
            # Initiate Mass from input batch sizes
            self.fft_batch_size, self.mean_batch_size, self.corr_batch_size = Mass3Func.find_optimal_batch_size(n, m)

        fft_input_len = self.fft_batch_size
        fft_output_len = int(fft_input_len / 2) + 1

        self.fftw_in_ts = pyfftw.zeros_aligned(fft_input_len, dtype='float64')
        self.fftw_out_ts = pyfftw.zeros_aligned(fft_output_len, dtype='complex128')

        self.fftw_in_query = pyfftw.zeros_aligned(fft_input_len, dtype='float64')
        self.fftw_out_query = pyfftw.zeros_aligned(fft_output_len, dtype='complex128')

        self.fftw_forward_ts = pyfftw.FFTW(self.fftw_in_ts, self.fftw_out_ts, flags=["FFTW_ESTIMATE"],
                                           direction='FFTW_FORWARD')
        self.fftw_forward_query = pyfftw.FFTW(self.fftw_in_query, self.fftw_out_query, flags=["FFTW_ESTIMATE"],
                                              direction='FFTW_FORWARD')

        self.fftw_out_results = pyfftw.zeros_aligned(fft_input_len, dtype='float64')
        self.fftw_in_results = pyfftw.zeros_aligned(fft_output_len, dtype='complex128')
        self.fftw_backward_results = pyfftw.FFTW(self.fftw_in_results, self.fftw_out_results, flags=["FFTW_ESTIMATE"],
                                                 direction='FFTW_BACKWARD')
        self.info = f"FFT_batch_size:{self.fft_batch_size}, mean_batch_size:{self.mean_batch_size}, " \
                    f"corr_batch_size:{self.corr_batch_size}"

    def __str__(self):
        return self.info

    def execute(self, data, query):
        if len(query) > self.fft_batch_size or len(query) > self.mean_batch_size or len(query) > self.corr_batch_size:
            print("Error, query length is larger than the batch size due to the pre-defined m size is "
                  "smaller than input query length")
            return None
        if len(data) < len(query):
            print("Error, input data length is smaller than query")
            return None
        else:
            self.fftw_in_query[:] = 0
            self.fftw_in_query[:len(query)] = np.flip(query)
            self.fftw_forward_query.execute()
            r = Mass3Func.mass_batch(data, query, self.fft_batch_size, self.mean_batch_size, self.corr_batch_size,
                                     self.fftw_forward_ts,
                                     self.fftw_backward_results, self.fftw_in_ts, self.fftw_out_ts, self.fftw_out_query,
                                     self.fftw_in_results)
            return r

    def execute_eu(self, data, query):
        if len(query) > self.fft_batch_size or len(query) > self.mean_batch_size or len(query) > self.corr_batch_size:
            print("Error, query length is larger than the batch size due to the pre-defined m size is "
                  "smaller than input query length")
            return None
        if len(data) < len(query):
            print("Error, input data length is smaller than query")
            return None
        else:
            self.fftw_in_query[:] = 0
            self.fftw_in_query[:len(query)] = np.flip(query)
            self.fftw_forward_query.execute()
            r = Mass3Func.mass_batch_eu(data, query, self.fft_batch_size, self.mean_batch_size, self.corr_batch_size,
                                        self.fftw_forward_ts,
                                        self.fftw_backward_results, self.fftw_in_ts, self.fftw_out_ts,
                                        self.fftw_out_query, self.fftw_in_results)
            return r
