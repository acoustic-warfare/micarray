%% Initialization
clear all

% In this section, several settings should be defined by the user.
% This includes if the signal should be emulated by Matlab (and if so, the 
% properties of the sound source/sources) or if data should be loaded from 
% a .txt file. It also includes array setup and geometry, bandpass filter
% properties and listening direction (for spatial filtering).

c = 340;                    % speed of sound (m/s)

% Choose if the algorithm should be run on emulated or recorded data:
signal_type = "emulated";   % options: "emulated", "recorded"

% If the data is recorded, state filename:
filename = '0908_ute_2000Hz_45_-45deg_2sources_reduced.txt'; % name of .txt file that in which the recorded data are stored

% If the signals should be emulated, state simulation properties
f_sampling = 15625;         % sampling frequency in Hz
t_start = 0;                % time of simulation start
t_end = 0.1;                  % time of simulation end

t_total = t_end - t_start;  % total simulation time
samples = floor(f_sampling*t_total); % number of samples
t = linspace(t_start, t_end, samples); % time vector

% --- SOURCES ---
away_distance = 20;     % The distance between the array and the sources (m)
N_sources = 1;          % Chose between 1 or 2 sources to be active.
                        % If 1 is chosen, the properties of source1 below
                        % will be used, and source2 ignored
% Properties soruce 1
theta_deg1 = 30; phi_deg1 = 0;
f_start1 = 3000; f_end1 = f_start1; f_res1 = 20;
t_start1 = 0; t_end1 = t_end;
% Properties soruce 2
theta_deg2 = -50; phi_deg2 = 20;
f_start2 = 2000; f_end2 = f_start2; f_res2 = 20;
t_start2 = 0; t_end2 = 0.1;

% --- ARRAY GEOMETRY ---
N_arrays = 1;               % Number of active arrays. Chose up to 4 arrays.
row_elements = 8;           % number of row microphone elements 
column_elements = 8;        % number of row microphone elements
uni_distance = 20*10^-3;    % uniform distance between elements
N_modes = 7;                % number of modes for the adaptive array configuration
% Center locations of arrays, note that this should adjusted according to the number of active arrays.
r_a1 = 10^-3 * [0;0;0];     % center location of array 1 (in cm)
r_a2 = 10^-3 * [-110;110;0];   % center location of array 2 (in cm)
r_a3 = 10^-3 * [110;-110;0];  % center location of array 3 (in cm)
r_a4 = 10^-3 * [-110;-110;0]; % center location of array 4 (in cm)

% --- FILTER ---
f_bands_N = 45;          % number of linearly spaced bands
frequency_bands = linspace(100, f_sampling/2-f_sampling/100, f_bands_N); % Center frequencies of the frequency bands to check
filter_order = 200;     % filter order

% --- OUTPUT ---
% Listening direction (spatial_filtering)
theta_listen = 0;    % listening direction in theta (deg)
phi_listen = 0;      % listening direction in phi (deg)

listen_unfiltered = 0;  % if true, plays the sound of the unfiltered signal
listen_filtered = 0;    % if true, plays the sound of the spatially filtered signal

% --- PLOT OPTIONS ---
plot_array_setup = 1;       % if True, plots the array setup
plot_filters = 0;           % if True, plots all bandpass filters
plot_generated_signals = 1; % if True, plots generated signals. Note that this only works if signal_type = "emulated"
plot_audio_signals = 0;     % if True, plots audio signals before and after beam forming (both in time and frequency domain)

%% Point audio source
% Initialization of properties of two audio sources
% The number of sources can be added by creating more sources of type
% audio_source, and then added in source_list
source1 = audio_source(f_start1, f_end1, f_res1, theta_deg1, phi_deg1, away_distance, t_start1, t_end1);
soruce2 = audio_source(f_start2, f_end2, f_res2, theta_deg2, phi_deg2, away_distance, t_start2, t_end2);

% Save properties in list
source_list = [source1 soruce2];

% Take out the user defined number of sources (defined by the variable N_sources)
% The variable "sources" holds the properties of the active sources
sources = source_list(1:N_sources);

%% Audio Array Geometry
% Initialization of properties of four arrays
elements = row_elements * column_elements; % total number of elements in the array

% Create properties of all sub-arrays
array_matrix1 = matrix_array(r_a1,uni_distance,row_elements,column_elements);
array_matrix2 = matrix_array(r_a2,uni_distance,row_elements,column_elements);
array_matrix3 = matrix_array(r_a3,uni_distance,row_elements,column_elements);
array_matrix4 = matrix_array(r_a4,uni_distance,row_elements,column_elements);

% Save all arrays in list
array_matrices_list = [array_matrix1, array_matrix2, array_matrix3, array_matrix4];

% Take out the user defined number of active arrays (defined by the variable N_arrays)
% The variable "array_matrices" holds the properties of the active arrays
array_matrices = array_matrices_list(1:N_arrays);

sub_arrays = length(array_matrices); % number of acitve sub-arrays

% --- Plot array setup ---
if plot_array_setup
    figure(1);      %Plot the geometry in the xy-plane
    for array = 1:sub_arrays
        z_array_coord = zeros(length(array_matrices(array).r_prime(1,:)));
        plot3(array_matrices(array).r_prime(1,:),array_matrices(array).r_prime(2,:),...
            z_array_coord,'linestyle','none','marker','o');
        axis square
        xlim([-0.5 0.5])
        ylim([-0.5 0.5])
        zlim([-0.5 0.5])
        hold on
    end
    for source = 1:length(sources)
        rho_source = sources(source).rho;
        theta_source = sources(source).theta;
        phi_source = sources(source).phi;
        x_coord = rho_source*sin(theta_source)*cos(phi_source);
        y_coord = rho_source*sin(theta_source)*sin(phi_source);
        z_coord = rho_source*cos(theta_source);
        plot3(x_coord,y_coord,...
            z_coord,'linestyle','none','marker','o','MarkerFaceColor','#D9FFFF');
    end
end

%% Generate signals on each element of the audio array
% Signals are generated at each run

if signal_type == "emulated"
    % Create a vector containing the audio_data for each sub_array
    array_audio_signals(sub_arrays) = audio_data;
    
    for array = 1:sub_arrays
        % Generate the audio signals on each array-element for each sub-array
        temp_signal = generate_array_signals(array_matrices(array),sources,f_sampling,t,c);
        array_audio_signals(array).audio_signals = temp_signal;
        disp(strcat('Audio signal for array ', int2str(array), ' generated'));
    end
end

% --- Plot generated signals ---
if plot_generated_signals && signal_type == "emulated"
    figure(2)
    samples_max = 100; % samples to plot
    for i = 1:length(array_audio_signals(1).audio_signals(1,:))
        plot(1:samples_max,array_audio_signals(1).audio_signals(1:samples_max,i))
        hold on
    end
    set(gca,'TickLabelInterpreter','latex','FontSize',18)
    xlabel('n ','Interpreter','latex','FontSize',18);
    ylabel('x[n]','Interpreter','latex','FontSize',18)
    title('Generated signals', 'latex','FontSize',18)
    set(gca,'GridALpha',0.1,'LineWidth',.3);

    % export figure
    %exportgraphics(gcf,'generated_signals_close.png','Resolution',300)
end


%% Load real data
if signal_type == "recorded"
    array_audio_signals(1).audio_signals = audio_data;
    [array_audio_signals(1).audio_signals,f_sampling] = load_data(filename); 
    array_audio_signals(1).audio_signals = array_audio_signals(1).audio_signals/(max(max(array_audio_signals(1).audio_signals)));
    samples = length(array_audio_signals(1).audio_signals(:,1));
    t = linspace(0, samples/f_sampling, samples);
    disp(strcat("Loading data from file: ",filename))
end

%% Listen to sound
if listen_unfiltered
    mic_listen = 1; % the microphone to listen at
    soundsc(array_audio_signals(1).audio_signals(:,mic_listen), f_sampling)
end

%% 
% 
% 
%                   All signals generated or loaded from file after this point  
% 
% 
% 
%
%
%
%
%                   Begin signal processing on data
%
%


%% Generate filter coefficients
filter_coefficients = get_filter_coefficients(f_sampling,frequency_bands);

% --- Plot of all passband filters ---
if plot_filters
    % colors of frequency bands
    colors = [0 0.4470 0.7410; 0.8500 0.3250 0.0980;0.9290 0.6940 0.1250;0.4940 0.1840 0.5560;...
        0.4660 0.6740 0.1880;0.3010 0.7450 0.9330];
    
    figure(9)
    for freq_ind = 1:f_bands_N
        color_ind = mod(freq_ind-1,6) + 1;
        [h,f] = freqz(filter_coefficients(freq_ind,:),1,1000,'whole',f_sampling);
        plot(f,20*log10(abs(h)),'color',colors(color_ind,:),'linewidth',0.8);
        set(gca,'TickLabelInterpreter','latex','FontSize',11)
        xlim([0 f_sampling/2])
        ylim([-60 0])
        xlabel('Frequency (Hz)','Interpreter','latex','FontSize',20);
        ylabel('Magnitude (dB)','Interpreter','latex','FontSize',20)
        title('Bandpass filters','FontSize',title_size)
        grid on
        hold on
    end
end


%% Load adaptive weight matrix
weight_m = weight_matrix(array_matrices(1),N_modes);

%% Spatial filtering of audio signal
% start spatial filtering, and store the spatial filtered signal in audio_out
audio_out = spatial_filtering(c,f_sampling,array_matrices,49,180,array_audio_signals,f_coefficients,frequency_bands);

%%
if listen_filtered
    soundsc(audio_out, f_sampling)
end

%% Plot results
if plot_audio_signals
    figure(10)
    plot(t,Audio_signal(:,2)/max(Audio_signal(:,2)))
    set(gca,'TickLabelInterpreter','latex','FontSize',18)
    ylim([-1 1])
    xlabel('t (s) ','Interpreter','latex','FontSize',18);
    ylabel('$y[f_s t]$','Interpreter','latex','FontSize',18)
    % export figure
    %exportgraphics(gcf,'time_domain_signal.png','Resolution',300)
    
    figure(11)
    plot(t,audio_out)
    set(gca,'TickLabelInterpreter','latex','FontSize',18)
    ylim([-1 1])
    xlabel('t (s) ','Interpreter','latex','FontSize',18);
    ylabel('$y[f_s t]$','Interpreter','latex','FontSize',18)
    % export figure
    %exportgraphics(gcf,'time_domain_signal_AW.png','Resolution',300)
    
    [spectrum_original,nu] = tdftfast(Audio_signal(:,2));
    [spectrum_bm,nu2] = tdftfast(audio_out2);
    
    figure(12)
    plot(nu,abs(spectrum_bm));
    title('Spectrum of spatially filtered signal')
    
    figure(13)
    plot(nu2,abs(spectrum_original));
    title('Spectrum of original signal')
end



