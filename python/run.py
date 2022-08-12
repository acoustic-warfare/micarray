from common import Array

import config

if config.backend_type == "vectorization":
    from cached_beamformer_vectorization import RTCachedBeamformer

elif config.backend_type == "multiprocessing":
    from cached_beamformer_multiprocessing import RTCachedBeamformer

else:
    raise RuntimeError("Incorrect process method")

if config.backend == "gpu":
    import cupy as xp
else:
    import numpy as xp

from numpy import load

def test():
    arrays = [Array(config.r_a1, config.distance, config.rows, config.columns)]

    beamformer = RTCachedBeamformer(
        arrays, config.theta_listen, config.phi_listen, config.window_size, config)

    filename = "/home/batman/Downloads/organization/micarray-gpu/demo/tmp/final/FPGA_data/1108_studion_music_45deg_2sources.npy"
    #filename = "/home/batman/Downloads/organization/micarray-gpu/demo/tmp/final/FPGA_data/two_sources.npy"
    #filename = "FPGA_data.npy"
    array_audio_signals = load(filename, allow_pickle=True)
    print("Loading from Memory: " + filename)
    print("theta_listen = " + str(config.theta_listen) + ", phi_listen = " + str(config.phi_listen))
    # Generate an arbitrary amount of cache and process it
    seconds = 1
    for _ in range(seconds):
        if config.backend == "gpu":
            beamformer.receive_buffer.put(xp.array(array_audio_signals[0]))
        else:
            beamformer.receive_buffer.put(array_audio_signals[0])

    # Process it until it is interupted
    beamformer.loop()

def test2():
    arrays = [Array(config.r_a1, config.distance, config.rows, config.columns)]

    beamformer = RTCachedBeamformer(
        arrays, config.theta_listen, config.phi_listen, config.window_size, config)

    filename = "/home/batman/Downloads/organization/micarray-gpu/demo/tmp/final/FPGA_data/"+config.filename+".txt"
    array_audio_signals = xp.empty(1, dtype = object)
    data = xp.loadtxt(open(filename).readlines()[:-1], delimiter=',')[:,4:]
    order = xp.empty((64),dtype=int)
    for i in range(64):
        order[i] = i
    for i in range(0,63,8):
        order[i:i+8] = xp.flip(order[i:i+8])
    data = data[:,order]
    array_audio_signals[0] = data
    print("Loading from Memory: " + filename)
    print("theta_listen = " + str(config.theta_listen) + ", phi_listen = " + str(config.phi_listen))
    # Generate an arbitrary amount of cache and process it
    seconds = 1
    for _ in range(seconds):
        if config.backend == "gpu":
            beamformer.receive_buffer.put(xp.array(array_audio_signals[0]))
        else:
            beamformer.receive_buffer.put(array_audio_signals[0])

    # Process it until it is interupted
    beamformer.loop()


if __name__ == "__main__":
    test2()
