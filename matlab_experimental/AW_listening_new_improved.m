function audio_out = AW_listening_new_improved(array_matrices,array_audio_signal,mega_filter_coefficients,center_frequencies)
%
%   Spatial filtering in r_hat(theta,phi)
%
%
%   This function takes in the audio_signals generated and the
%   corresponding array matrix-geometries. This function further needs some
%   information about the audio_signals such as the sampling frequency,
%   soundspeed in the medium (c).
%
%   A direction is needed (theta,phi) to output a spatially filtered
%   signal.
%
%   The function splits the input audio signals recorded by the
%   array_matrices and then perform narrowband beamforming on the signals.
%   Different audio signals corresponds to different sub_arrays.
%

% Initialize our audio out vector
samples = length(array_audio_signal(1).audio_signals(:,1));
audio_in = array_audio_signal(1).audio_signals;
audio_out = zeros(samples,1);

elements = array_matrices(1).elements;

for freq_ind = 1:length(center_frequencies)
    for mic_ind = 1:elements
        row_index = (freq_ind-1)*elements + mic_ind;
        audio_out = audio_out + filter(mega_filter_coefficients(row_index,:),1,audio_in(:,mic_ind));
    end
end

end
