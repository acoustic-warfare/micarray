import numpy as np
import math
import matplotlib.pyplot as plt
from pathlib import Path
import config
import sounddevice
import soundfile as sf

def load_data_BB(filename):
    # OLD FUNCTION TO LOAD DATA FROM BEAGELBONE
    #filename = 'calibration_signals_array1'
    #path = Path('/home/batman/BatSignal/Batprogram/calibration/calibration_data/' + filename + '.bin.txt')
    #path = Path('/home/batman/BatSignal/data/studion1506/' + filename + 'txt')
    path = Path('/home/batman/BatSignal/' + filename + '.txt')

    # Load calibration data from file
    data = np.loadtxt(open(path,'rb'),delimiter=',')
    f_sampling = data[0,0]  # sampling frequency
    data = data[:,2:66]
    return data, f_sampling

def load_data_FPGA(filename):
    #   FUNCTION TO LOAD DATA FROM .TXT FILE INTO NUMPY ARRAY 
    #   (RECORDED BY FPGA)

    path = Path('/home/batman/github/micarray-gpu/FPGA_data/1108/' + filename + '.txt')

    # Load recorded data from file
    data = np.loadtxt(open(path,'rb').readlines()[:-1],delimiter=',')
    f_sampling = 48828.125  # get sampling frequency
    data = data[:,4:]       # take out data from microphones only
    print(len(data[:,4]))

    # Order of microphones, to make it compatible with beamforming algoritms
    order = np.array([57, 58, 59, 60, 61, 62, 63, 64, \
    56, 55, 54, 53, 52, 51, 50, 49, \
    41, 42, 43, 44, 45, 46, 47, 48, \
    40, 39, 38, 37, 36, 35, 34, 33, \
    25, 26, 27, 28, 29, 30, 31, 32, \
    24, 23, 22, 21, 20, 19, 18, 17, \
    9, 10, 11, 12, 13, 14, 15, 16, \
    8, 7, 6, 5, 4, 3, 2, 1], dtype=int)-1

    # Rearrange microphone signals after correct order
    #data = data[:,order] 

    return data, int(f_sampling)

def delete_mic_data(signal, mic_to_delete):
    #   FUNCTION THAT SETS SIGNALS FROM 'BAD' MICROPHONES TO 0
    for mic in range(len(mic_to_delete)):
        for samp in range(len(signal[:,0])):
            signal[samp,mic_to_delete[mic]] = 0
    return signal

def write_to_txt_file(filename, signals):
    #   FUNCTION THAT WRITES VALUES TO .TXT FILE
    np.savetxt(filename, signals, delimiter=',\t ', newline='\n', header='', footer='', comments='# ', encoding=None)

def write_to_npy_file(filename, signals):
    array_signals = np.zeros((1), dtype=object)
    array_signals[0] = signals
    np.save(filename+'.npy', array_signals)

def play_sound(sound_signal, f_sampling):
    scaled = sound_signal/np.max(np.abs(sound_signal))
    sounddevice.play(scaled, f_sampling, blocking=True)

    #sf.write("test.wav", sound_signal, int(f_sampling), 'PCM_24')


def main():
    recording_device = 'FPGA' # choose between 'FPGA' and 'BB' (BeagelBone) 
    #filename = '0908_440Hz_0deg'
    filename = config.filename
    print(filename)
    initial_samples = 30000                 # initial samples, at startup phase of Beaglebone recording

    # Choose the mic signals that should be set to zero
    mics_to_delete = [18, 64]
    arr_mics_to_delete = np.array(mics_to_delete, dtype = int)-1 # converts mic_to_delete to numpy array with correct index

    # Plot options
    show_plots = 1      # if show_plots = 1, then plots will be shown
    plot_period = 1     # periods to plot
    f0 = 800            # frequency of recorded sinus signal

    # Load data from .txt file
    if recording_device == 'FPGA':
        data, fs = load_data_FPGA(filename)
        #write_to_npy_file(filename,data)
    elif recording_device == 'BB':
        data, fs = load_data_BB(filename)
    total_samples = len(data[:,0])          # Total number of samples
    initial_data = data[0:initial_samples,] # takes out initial samples of signals 


    if recording_device == 'FPGA':
        ok_data = data # all data is ok
    elif recording_device == 'BB':
        ok_data = data[initial_samples:,] # initial startup values are ignored


    plot_samples = math.floor(plot_period*(fs/f0))                     # number of samples to plot, to use for axis scaling
    max_value_ok = np.max(np.max(ok_data[0:4000,],axis=0)) # maximum value of data, to use for axis scaling in plots

    #play_sound(data[:,21],fs)
    print('f_sampling: '+ str(int(fs)))

    # --- PLOT ---
    #   of bad microphones
    plt.figure()
    for mic in range(len(arr_mics_to_delete)):
        plt.plot(ok_data[:,arr_mics_to_delete[mic]])
    plt.xlim([0, plot_samples])
    plt.ylim([-max_value_ok*1.1, max_value_ok*1.1])
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.title('Bad microphones')


    # --- PLOT ---
    #   of all individual signals in subplots, two periods
    fig, axs = plt.subplots(4,16)
    fig.suptitle("Individual signals", fontsize=16)
    start_val = 4000
    for j in range(4):
        for i in range(16):
            axs[j,i].plot(ok_data[start_val:start_val+plot_samples,i+j*16])
            axs[j,i].set_title(str(i+j*16+1), fontsize=8)
            axs[j,i].set_ylim(-max_value_ok*1.1, max_value_ok*1.1)
            axs[j,i].axis('off')

    # Set microphone signals of bad mics to zero
    #clean_data = delete_mic_data(ok_data, arr_mics_to_delete)
    #clean_initial_data = delete_mic_data(initial_data, arr_mics_to_delete)

    # --- PLOT ---
    #   plot of all microphones, after bad signals have been set to 0
    plt.figure()
    plt.plot(ok_data)
    plt.xlim([0, plot_samples])
    plt.ylim([-max_value_ok*1.1, max_value_ok*1.1])
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.suptitle('All microphones')

    
    # --- PLOT ---
    #   of selected microphones
    plot_mics = [8, 18, 60, 64]                     # what microphones to plot
    arr_plot_mics = np.array(plot_mics)-1   # convert plot_mics to numpy array with correct index
    mic_legend = []                         # empty list that should hold legends for plot
    plt.figure()
    for i in range(len(arr_plot_mics)):
        plt.plot(ok_data[:,int(arr_plot_mics[i])], '-*')
        mic_legend = np.append(mic_legend,str(arr_plot_mics[i]+1))
    plt.xlim([0, plot_samples])
    plt.ylim([-max_value_ok*1.1, max_value_ok*1.1])
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.suptitle('Selected microphones microphones')
    plt.legend(mic_legend)

    # --- PLOT ---
    plt.figure()
    plt.plot(initial_data[:,3])
    plt.xlim([0, initial_samples])
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.suptitle('Initial values')

    # --- PLOT ---
    #   of FFT of one signal
    mic = 1         # mic signals of FFT
    samples = len(data[:,0])
    t_stop = samples/fs
    t = np.linspace(0,t_stop,samples)
    data_FFT = np.fft.fft(ok_data[:,mic-1])
    energy = abs(data_FFT)**2
    freq = np.fft.fftfreq(t.shape[-1])
    plt.figure()
    plt.plot(fs*freq,energy)
    plt.title('Energy of signal')
    plt.xlabel('Frequency [Hz]')
    plt.legend(str(mic))

    
    # --- PLOT ---
    #   of FFT of several signals
    mics_FFT = [1,18, 64]
    arr_mics_FFT = np.array(mics_FFT,dtype=int)-1
    FFT_mic_legend = []                         # empty list that should hold legends for plot
    plt.figure()
    for i in range(len(arr_mics_FFT)):
        data_FFT = np.fft.fft(ok_data[:,int(arr_mics_FFT[i])])
        energy = abs(data_FFT)**2
        freq = np.fft.fftfreq(t.shape[-1])
        plt.plot(fs*freq,energy)
        FFT_mic_legend = np.append(FFT_mic_legend,str(arr_mics_FFT[i]+1))
    plt.suptitle('Energy of selected microphones signals')
    plt.xlabel('Frequency [Hz]')
    plt.legend(FFT_mic_legend)

    # Show all plots
    if show_plots:
        plt.show()

main()