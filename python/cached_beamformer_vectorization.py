import time

import config

if config.backend == "gpu":
    import cupy as xp
    xp.cuda.runtime.setDevice(0)
elif config.backend == "cpu":
    import numpy as xp

import numpy
from scipy import signal

from scipy.io import wavfile
import signal as interupt


propagation_speed = 340  # Speed of sound in air


def calculate_filter_coefficients(f_sampling, frequency_bands, scale_factor, n_bands, filter_order):
    """Calculate filter coefficients beforehand. Scipy firwin must have a Numpy array so force allocate
    array and convert later to CuPy if backend is GPU."""
    f_coefficients = numpy.zeros((n_bands, filter_order))
    for freq_ind in range(n_bands):
        nu_0 = 2*frequency_bands[freq_ind]/f_sampling
        cut_off = [nu_0 - nu_0/scale_factor, nu_0 + nu_0/scale_factor]
        b = signal.firwin(filter_order, cut_off, window="hamming",
                          pass_zero=False)  # filter coefficients
        f_coefficients[freq_ind, :] = b
    return xp.array(f_coefficients)


def adaptive_array_config_matrix(matrix_array, modes):
    # Creates the weight matrix
    row_elements = matrix_array.row_elements
    column_elements = matrix_array.row_elements

    # weight_matrix = xp.zeros((7, row_elements*column_elements))
    weight_matrix = xp.zeros((7, row_elements*column_elements))

    for mode in range(1, modes+1):
        # weight = xp.zeros((1, row_elements*column_elements))
        weight = xp.zeros((row_elements*column_elements))
        row_lim = xp.ceil(row_elements/mode)
        column_lim = xp.ceil(column_elements/mode)
        for i in range(int(row_lim)):
            for j in range(int(column_lim)):
                # this calculation could be wrong thanks to matlab and python index :))
                element_index = (mode*i*row_elements + mode*j)
                weight[element_index] = 1
        weight_matrix[mode-1, :] = weight
    return weight_matrix


def load_calibration_weights(array, elements, f_bands):
    # placeholder function, to be completed later
    # function should load calibration weights form file
    # returns matrix with calibration weightsfor all microphones, at all calibration frequencies
    weights = xp.ones((f_bands, elements))
    return weights


class FIFO:

    def __init__(self, shape):
        self.buffer = xp.empty(shape)

    def put(self, data):
        self.buffer = xp.concatenate((self.buffer, data))

    def get(self, size):
        data = self.buffer[:size]

        self.buffer = self.buffer[size:]
        return data

    def peek(self, size):
        return self.buffer[:size]

    def getvalue(self):
        """No copy peek"""
        return self.buffer

    def __len__(self):
        return len(self.buffer)


class RTCachedBeamformer(object):

    """Real-Time Cached Beamformer."""

    running = True

    def __init__(self, arrays, theta, phi, window: int, config, debug=True):
        self.array_matrices = arrays
        self.theta = theta
        self.phi = phi
        self.window = window
        self.debug = debug

        # calculates what mode to use, depending on the wavelength of the signal
        self.uni_distance = config.distance              # distance between elements
        self.modes = config.modes
        self.fs = config.f_sampling
        self.normal_coefficient = config.normal_coefficient
        # Setup listening direction
        self.set_listen_direction(self.theta, self.phi)
        self.elements = config.rows*config.columns
        

        # load adaptive weights, calibration weights and filter coefficients
        # self.frequency_bands = xp.linspace(
        #     config.bandwidth[0], config.bandwidth[1], config.f_bands_N)
        self.frequency_bands = xp.linspace(
            config.bandwidth[0], config.bandwidth[1], config.f_bands_N)

        # load adaptive weights, calibration weights and filter coefficients
        self.adaptive_weights_matrix = adaptive_array_config_matrix(
            self.array_matrices[0], self.modes)

        self.calibration_weights = load_calibration_weights(
            0, config.rows*config.columns, config.f_bands_N)

        if config.backend == "gpu":
            freq_bands = xp.asnumpy(self.frequency_bands)
        else:
            freq_bands = self.frequency_bands
        self.filter_coefficients = calculate_filter_coefficients(
            config.f_sampling, freq_bands, config.scale_factor, config.f_bands_N, config.filter_order)
        # print(self.filter_coefficients.shape[0])
        # self.freq_jobs = [freq_ind for freq_ind in range(
        #     len(self.filter_coefficients[:, 0]))]  # Jobs to do in parallel
        self.freq_jobs = [freq_ind for freq_ind in range(self.filter_coefficients.shape[0])]

        # Buffer is empty of data, but prepared to fill with each element
        self.receive_buffer = FIFO((0, self.elements))
        self.output_buffer = FIFO((0, 1))

        # Signal Catcher for Ctrl-C to stop the program gracefully
        interupt.signal(interupt.SIGINT, self.exit_gracefully)
        interupt.signal(interupt.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args, **kwargs):
        """Break loop on `SIGINT` or `SIGTERM`"""
        self.running = False

    def set_listen_direction(self, theta, phi):
        """Set the current listen direction."""
        self.x_factor = xp.sin(theta) * xp.cos(phi)
        self.y_factor = xp.sin(theta) * xp.sin(phi)

    def weight_index(self, frequency):
        # relative wavelength to distance between microphone elements
        wavelength_rel = frequency*self.uni_distance/propagation_speed

        if wavelength_rel > 0.1581:
            mode = 1
        elif (wavelength_rel <= 0.156) and (wavelength_rel > 0.0986):
            mode = 3
        elif (wavelength_rel <= 0.0986) and (wavelength_rel > 0.085):
            mode = 5
        elif (wavelength_rel <= 0.085) and (wavelength_rel > 0.07):
            mode = 6
        else:
            mode = 7
        return mode

    def get_phase_shift(self):
        self.phase_shift_values = xp.empty(
            (len(self.freq_jobs), self.elements))
        for freq_ind in self.freq_jobs:
            # On demand calculation
            # Calculate the phaseshift value according to listening direction (x_factor and y_factor)
            self.phase_shift_values[freq_ind] = -self.frequency_bands[freq_ind] * (2*xp.pi/propagation_speed) * (
                self.r_prime[0, ] * self.x_factor + self.r_prime[1, ]*self.y_factor)

        self.shifted_cos = xp.cos(self.phase_shift_values)
        self.shifted_sin = xp.sin(self.phase_shift_values)

        return self.phase_shift_values

    def load_weights(self):
        self.weights = xp.empty((len(self.freq_jobs), self.elements))
        for freq_ind in self.freq_jobs:
            self.weights[freq_ind] = self.adaptive_weights_matrix[self.weight_index(
                self.frequency_bands[freq_ind])-1, :]

        return self.weights


    def process(self, window):
        b = self.filter_coefficients

        # Load weights for microphones
        weights = self.load_weights()

        # Filter all 64 on all frequency_bands signals using vectorization
        filtered_signals = xp.empty(
            (len(self.freq_jobs), self.window, self.elements))

        # TODO Fix this function to vectorize, as this function takes around 8 seconds for 45 bands
        for freq_ind in self.freq_jobs:
            # Convolve over the coefficients and apply to the window using scipy linear filter
            filtered_signals[freq_ind] = xp.apply_along_axis(lambda y: xp.convolve(
            b[freq_ind, :], y), -1, window)[(slice(None, None, None), slice(None, self.elements, None))]
            # The slicing part is speed optimized for array broadcasting 
            # and not human readable but will result in ValueError when shape of window is other than
            # defined, eg. when a buffer is almost empty
        
        # Normalized frequency (Nyqvist)
        all_nyqvists = self.frequency_bands / self.fs
        # Calculate values beforehand (Reduces processing time)
        cossed = self.shifted_cos
        #sinned = self.shifted_sin / (2*xp.pi/propagation_speed * all_nyqvists.reshape(
        #    len(self.freq_jobs), 1))  # The narrowband frequency
        sinned = self.shifted_sin / (2*xp.pi * all_nyqvists.reshape(
            len(self.freq_jobs), 1))  # The narrowband frequency

        # Allocate memory
        x_length = filtered_signals.shape[1]
        y = xp.zeros_like(filtered_signals)

        # Get the derivative for all microphones using vectorization
        for i in range(1, x_length-1):
            y[:, i] = cossed * filtered_signals[:, i] + \
                sinned * \
                (filtered_signals[:, i+1]/2 - filtered_signals[:, i-1]/2)
        
        # Summarize all weighted and phaseshifted microphones to a single output stream
        beamformed_audio = xp.sum(
            y * weights[:, None], axis=(0, 2)).reshape((x_length, 1))

        return beamformed_audio / xp.sum(weights)

    def loop(self):
        """Main process loop."""
        self.r_prime = self.array_matrices[0].r_prime
        self.get_phase_shift()
        while self.running:
            start = time.time()
            # Get the data in realtime
            audio_data = self.receive_buffer.get(self.window)
            if len(audio_data) == 0:
                if self.debug:
                    print("Queue is empty...")
                time.sleep(0.1)
                continue
            
            # Process for each array
            # Will only work with one array at the moment
            for array in self.array_matrices:
                self.r_prime = array.r_prime
                try:
                    output = self.process(audio_data)
                except ValueError:  # could not broadcast input array from shape(674, 64) into shape(2034, 64)
                    continue
            # Append ouput to a buffer
            self.output_buffer.put(output)

            if self.debug:
                rate = (self.window / (time.time() - start) /
                        self.fs)**-1
                print(f"Real-Time Crunch Rate: {round(rate, 3)}s", end="\r")
                
        
        # Save the beamformed signal to a WAV file
        result = self.output_buffer.getvalue()
        result = result.reshape(len(result))
        result *= self.normal_coefficient
        xp.savetxt(config.filename+"spatial_filtered", result.astype(xp.float32), delimiter=',\t ', newline='\n', header='', footer='', comments='# ', encoding=None)
        wavfile.write(config.filename+"_2.wav", self.fs, result.astype(xp.float32)) # assumes that result contains values between -1.0 and 1.0 
