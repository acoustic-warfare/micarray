import time
import multiprocessing

import config

if config.backend == "gpu":
    import cusignal as signal
    import cupy as np
    np.cuda.runtime.setDevice(0)
elif config.backend == "cpu":
    from scipy import signal
    import numpy as np


from scipy.io import wavfile
import signal as interupt


propagation_speed = 340  # Speed of sound in air


def calculate_filter_coefficients(f_sampling, frequency_bands, scale_factor, n_bands, filter_order):
    f_coefficients = np.zeros((n_bands, filter_order))
    for freq_ind in range(n_bands):
        nu_0 = 2*frequency_bands[freq_ind]/f_sampling
        cut_off = [nu_0 - nu_0/scale_factor, nu_0 + nu_0/scale_factor]
        b = signal.firwin(filter_order, cut_off, window="hamming",
                          pass_zero=False)  # filter coefficients
        f_coefficients[freq_ind, :] = b
    return f_coefficients


def adaptive_array_config_matrix(matrix_array, modes):
    # Creates the weight matrix
    row_elements = matrix_array.row_elements
    column_elements = matrix_array.row_elements

    # weight_matrix = np.zeros((7, row_elements*column_elements))
    weight_matrix = np.zeros((7, row_elements*column_elements))

    for mode in range(1, modes+1):
        # weight = np.zeros((1, row_elements*column_elements))
        weight = np.zeros((1, row_elements*column_elements))
        row_lim = np.ceil(row_elements/mode)
        column_lim = np.ceil(column_elements/mode)
        for i in range(int(row_lim)):
            for j in range(int(column_lim)):
                # this calculation could be wrong thanks to matlab and python index :))
                element_index = (mode*i*row_elements + mode*j)
                weight[0, element_index] = 1
        # weight_matrix[mode-1, :] = weight
        print(weight_matrix[0], weight, weight.shape)
        weight_matrix[mode-1, :] = weight
    return weight_matrix


def load_calibration_weights(array, elements, f_bands):
    # placeholder function, to be completed later
    # function should load calibration weights form file
    # returns matrix with calibration weightsfor all microphones, at all calibration frequencies
    weights = np.ones((f_bands, elements))
    return weights


class FIFO:

    def __init__(self, shape):
        self.buffer = np.empty(shape)

    def put(self, data):
        self.buffer = np.concatenate((self.buffer, data))

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
        # self
        

        # load adaptive weights, calibration weights and filter coefficients
        self.frequency_bands = np.linspace(
            config.bandwidth[0], config.bandwidth[1], config.f_bands_N)

        # load adaptive weights, calibration weights and filter coefficients
        self.adaptive_weights_matrix = adaptive_array_config_matrix(
            self.array_matrices[0], self.modes)

        self.calibration_weights = load_calibration_weights(
            0, config.rows*config.columns, config.f_bands_N)

        self.filter_coefficients = calculate_filter_coefficients(
            config.f_sampling, self.frequency_bands, config.scale_factor, config.f_bands_N, config.filter_order)
        
        self.freq_jobs = [freq_ind for freq_ind in range(
            len(self.filter_coefficients[:, 0]))]  # Jobs to do in parallel

        self.manager = multiprocessing.Manager()

        # Gather results
        self.return_queue = self.manager.dict()

        elements = config.rows*config.columns

        # Buffer is empty of data, but prepared to fill with each element
        self.receive_buffer = FIFO((0, elements))
        self.output_buffer = FIFO((0, 1))

        # Signal Catcher for Ctrl-C to stop the program gracefully
        interupt.signal(interupt.SIGINT, self.exit_gracefully)
        interupt.signal(interupt.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args, **kwargs):
        """Break loop on `SIGINT` or `SIGTERM`"""
        self.running = False

    def set_listen_direction(self, theta, phi):
        """Set the current listen direction."""
        self.x_factor = np.sin(theta) * np.cos(phi)
        self.y_factor = np.sin(theta) * np.sin(phi)

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


    def _process(self, freq_ind, multi_window, r_prime, return_queue=None):
        """Filtering for each microphone with phaseshift.
        This method would run for each frequency span in the bandwidth."""

        # Filter coefficients for this frequency index
        b = self.filter_coefficients[freq_ind, :]

        # Center frequency
        frequency = self.frequency_bands[freq_ind]

        # The narrowband frequency
        k = 2*np.pi*frequency/propagation_speed

        # Normalized frequency (Nyqvist)
        ny = frequency/self.fs

        # Load weights for microphones
        weights = self.adaptive_weights_matrix[self.weight_index(
            frequency)-1, :]  

        # Filter all 64 signals using vectorization
        filtered_signals = self.calibration_weights[freq_ind,
                                                    ] * signal.lfilter(b, 1.0, multi_window[:, ])
        
        # Calculate the phaseshift value according to listening direction (x_factor and y_factor)
        phase_shift_value = -k * \
            (r_prime[0, ] * self.x_factor + r_prime[1, ]*self.y_factor)

        x_length = len(filtered_signals)
        y = np.zeros_like(filtered_signals)

        # Calculate values beforehand (Reduces processing time)
        cos_shift = np.cos(phase_shift_value)
        sin_shift_ny = np.sin(phase_shift_value) / 2*np.pi*ny

        # Get the derivative
        for i in range(1, x_length-1):
            y[i] = cos_shift * filtered_signals[i] + \
                sin_shift_ny * (filtered_signals[i+1]/2 - filtered_signals[i-1]/2)

        # Summarize all microphones and reshape it to fit the shape of (x, 1), see Numpy for reference
        mic_data = np.sum(weights[:, ] * y, axis=(1)).reshape((x_length, 1))

        norm_coeff = 1/sum(weights)

        # Assign the multiprocessing return queue with the value received at the beamforming process
        return_queue[freq_ind] = mic_data * norm_coeff

    def loop(self):
        """Main process loop."""
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
                r_prime = array.r_prime

                # All jobs that will be made
                jobs = []  

                # Create a separate process for each frequency band
                for freq_ind in self.freq_jobs:
                    p = multiprocessing.Process(
                        target=self._process, args=(freq_ind, audio_data, r_prime), kwargs=dict(return_queue=self.return_queue))

                    jobs.append(p)

                # Start the jobs
                for p in jobs:  
                    p.start()

                # Gather the jobs to the main loop of the program
                for proc in jobs: 
                    proc.join()

            # Summarize all data gathered from the separate jobs
            output = np.sum(self.return_queue.values(), axis=(0))

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
        wavfile.write("result.wav", self.fs, result.astype(np.float32))


