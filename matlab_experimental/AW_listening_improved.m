function audio_out = AW_listening_improved(c,f_sampling,array_matrices,theta_listen,phi_listen,array_audio_signal,filter_coefficients,center_frequencies)
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
audio_out = zeros(samples,1);

% Array data
sub_arrays = length(array_matrices);

% Convert input theta and phi in degrees to radians
theta = pi/180*theta_listen;
phi = pi/180*phi_listen;

%theta = theta_listen;
%phi = phi_listen;

x_factor = sin(theta)*cos(phi);
y_factor = sin(theta)*sin(phi);

% Weight matrix
weight_m = weight_matrix(array_matrices(1),7);

for array = 1:sub_arrays
    % Filter the signals generated on the arrays
    r_prime = array_matrices(array).r_prime;
    x = r_prime(1,:);
    y = r_prime(2,:);
    audio_signal = array_audio_signal(array).audio_signals;
    elements = array_matrices(array).elements;

    for freq_ind = 1:length(filter_coefficients(:,1))
        % Filter coefficients for each band
        b = filter_coefficients(freq_ind,:);

        % Center frequency
        frequency = center_frequencies(freq_ind);

        % The narrowband frequency
        k = 2*pi*frequency/c;

        % The normalized frequency
        ny = frequency/f_sampling;

        w_index = weight_index(array_matrices(array),frequency,c);
        weight = weight_m(w_index,:);
        audio_temp = filter(b,1,audio_signal);
        mic_data = zeros(samples,1);

        for (mic_ind = 1:elements)
            if weight(mic_ind) == 1
                % Apply filter on every signal recorded from the elements
                %audio_temp(:,1) = filter(b,1,audio_signal(:,mic_ind));

                % Phase shift value that is dependent on the frequency and
                % the location of the element (x,y)
                phase_shift_value = -k*(x(mic_ind)*x_factor + ...
                    y(mic_ind)*y_factor);
                
                % Add the signals together
                %mic_data(:,1) = mic_data(:,1) + ...
                %    phase_shift(audio_temp,ny,phase_shift_value);
                b2 = [sin(phase_shift_value)/(4*pi*ny), cos(phase_shift_value), -sin(phase_shift_value)/(4*pi*ny)];

                mic_data(:,1) = mic_data(:,1) + ...
                    filter(b2,1,audio_temp(:,mic_ind));

                %mic_data(:,mic_ind) = phase_shift(filter(b,1,audio_signal(:,mic_ind)),ny,phase_shift_value);
            end
        end
       % audio_out = audio_out + sum(mic_data,2);
       audio_out = audio_out + mic_data;
    end
    
end

%audio_out = audio_out/max(audio_out);
end
