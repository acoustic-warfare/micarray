# --- EMULATING DATA variables ---
f_sampling = 16000          # sampling frequency in Hz
samples = 16000             # number of samples to generate
t_start = 0                 # start time of simulation 
t_end = samples/f_sampling  # end time of simulation

# Source variables
away_distance = 700         # distance between the array and sources
# source1
f_start1 = 300              # lowest frequency that the source emitts
f_end1 = 350                # highest frequency that the source emitts
f_res1 = 20                 # resolution of frequency
theta_deg1 = 40             # theta angel of source placement, relative to origin of array
phi_deg1 = 20               # phi angel of source placement, relative to origin of array
t_start1 = 0                # start time of emission
t_end1 = 0.5                # end time of emission

# source1
f_start2 = 500              # lowest frequency that the source emitts
f_end2 = 550                # highest frequency that the source emitts
f_res2 = 20                 # resolution of frequency
theta_deg2 = 40             # theta angel of source placement, relative to origin of array
phi_deg2 = -20              # phi angel of source placement, relative to origin of array
t_start2 = 0.5              # start time of emission
t_end2 = 0.5                # end time of emission


# --- ANTENNA ARRAY setup variables ---
r_a1 = [0, 0, 0]            # coordinate position of origin of array1
r_a2 = [0.08, 0, 0]         # coordinate position of origin of array2
r_a3 = [-0.24, 0, 0]        # coordinate position of origin of array3
r_a4 = [0.24, 0, 0]         # coordinate position of origin of array4
rows = 8                    # number of rows
columns = 8                 # number of columns
elements = rows*columns     # number of elements
distance = 20 * 10**(-3)    # distance between elements (m)


# --- FILTER variables ---
filter_order = 200          # filter order
scale_factor = 1000         # scale factor, adjusting filter width
f_bands_N = 45              # number of frequency bands


# --- OTHER variables ---
# Number of modes for adaptive weights
modes = 7

# Beamforming resolution
x_res = 10                          # resolution in x
y_res = 10                          # resolution in y

# Listening values
theta_listen = 0                    # direction to listen in, theta angle
phi_listen = 45                     # direction to listen in, theta angle


# --- RECORDED OR EMULATED DATA ---
#   choose if the program should work with recorded or emulated data
audio_signals = 'recorded'              # 'emulated' or 'recorded'
active_arrays = 1                       # number of active arrays (supporting up to 4 arrays)

#  Emulated data
sources = 1                             # number of sources to emulate data from (supporting up to 2 sources)

# Recorded data
filename = 'filename'                   # filename of recorded data
path = ''   # path to file
initial_values = 4000                  # number of initial values, if values in start up of recording are of low quality
