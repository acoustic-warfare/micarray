%% Initialization
clear all

% In this section, several settings should be defined by the user.
% This includes if the signal should be emulated by Matlab (and if so, the 
% properties of the sound source/sources) or if data should be loaded from 
% a .txt file. It also includes array setup and geometry, bandpass filter
% properties and beam forming resolution (for heatmap).
% It is also possible to chose what plots to display, and frame settings
% for a video.

c = 340;                    % speed of sound (m/s)

% Choose if the algorithm should be run on emulated or recorded data:
signal_type = "emulated";   % options: "emulated", "recorded"

% If the data is recorded, state filename:
filename = '0908_ute_2000Hz_45_-45deg_2sources_reduced.txt'; % name of .txt file that in which the recorded data are stored

% If the signals should be emulated, state simulation properties
f_sampling = 15625;         % sampling frequency in Hz
t_start = 0;                % time of simulation start
t_end = 0.1;                % time of simulation end

t_total = t_end - t_start;              % total simulation time
samples = floor(f_sampling*t_total);    % number of samples
t = linspace(t_start, t_end, samples);  % time vector

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
% Beamforming resolution (for heatmap)
x_res = 40;                         % Resolution in x
y_res = 40;                         % Resolution in y

% Video settings (for heatmap)
fps_rate = 24;  % chose the fps rate. If you do not want to generate a movie, set fps_rate = 0.

% Listening direction (spatial_filtering)
theta_listen = 0;    % listening direction in theta (deg)
phi_listen = 0;      % listening direction in phi (deg)

% --- PLOT OPTIONS ---
plot_array_setup = 1;       % if True, plots the array setup
plot_filters = 0;           % if True, plots all bandpass filters
plot_generated_signals = 1; % if True, plots generated signals. Note that this only works if signal_type = "emulated"
plot_audio_signals = 0;     % if True, plots audio signals before and after beam forming (both in time and frequency domain)
plot_intensity_grad = 0;    % if True, plots grad1 and grad2 of intensity maps


%% Movie initialization
frames_N = ceil(t_total*fps_rate);
frames_index = zeros(2,frames_N+1);

time_frames_N = floor(length(t)/frames_N);
last_frame_ind = 1;

for frame_ind = 1:frames_N
    if frame_ind == frames_N
       frames_index(1,frame_ind+1) = last_frame_ind ;
       frames_index(2,frame_ind+1) = length(t);
    else 
       frames_index(1,frame_ind+1) = last_frame_ind ;
       frames_index(2,frame_ind+1) = frame_ind*time_frames_N;
       last_frame_ind = frame_ind*time_frames_N;
    end
end

frames_index(1,1) = 1;
frames_index(2,1) = length(t);

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
%soundsc(array_audio_signals(1).audio_signals(:,1), f_sampling)

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

%% Beamforming
%   Set up scanning window
x_listen = linspace(-1,1,x_res);    %Our scanning window, x coordinates
y_listen = linspace(-1,1,y_res);    %Our scanning window, y coordinates
r_scan = sqrt(2);                   %radius of scanning window. r_scan^2 = x^2 + y^2 + z^2

% Create N-audio complete audio signals for every band
audio_filtered_complete(sub_arrays,f_bands_N) = audio_data;
%filter_coefficients = zeros(f_bands_N,filter_order+1);

% Create colormaps for each frequency-band
color_maps_complete = color_map.empty(0,f_bands_N);
for freq_ind = 1:f_bands_N
    color_map_new = zeros(length(y_listen),length(x_listen));  
    color_maps_complete(freq_ind) = color_map(color_map_new);
end

% Create colormaps for the movie
if fps_rate > 0
    color_maps_movie(length(frames_index(1,:))-1,f_bands_N) = color_map;
    for frame_ind = 1:length(frames_index(1,:))-1
        for freq_ind = 1:f_bands_N
            color_map_new = zeros(length(y_listen),length(x_listen));  
            color_maps_movie(frame_ind,freq_ind) = color_map(color_map_new);
        end
    end
end

%% Generate filter coefficients
filter_coefficients = get_filter_coefficients(f_sampling,frequency_bands);

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

%% Filter all signals
for array = 1:sub_arrays
    % Filter the signals generated on the arrays
    Audio_signal = array_audio_signals(array).audio_signals;
    elements = array_matrices(array).elements;

    for freq_ind = 1:length(frequency_bands)
        % load filter coefficients
        b = filter_coefficients(freq_ind,:);
    
        audio_temp = zeros(samples,elements); 

        for mic_ind = 1:elements
            % Apply filter on every signal recorded from the elements
            audio_temp(:,mic_ind) = filter(b,1,Audio_signal(:,mic_ind));
        end
    
        % Add the complete set of element-signals
        audio_filtered_complete(array,freq_ind) = audio_data(audio_temp);
    end
end

%% Load adaptive weight matrix
weight_m = weight_matrix(array_matrices(1),N_modes);

%% Scanning
for x_ind = 1:length(x_listen)
    x = x_listen(x_ind);
    x_ind
    for y_ind = 1:length(y_listen)
         y = y_listen(y_ind);
         z_0 = sqrt(r_scan^2-x^2-y^2);
         theta = abs(acos(z_0/(sqrt(x^2 + y^2 + z_0^2))));   %Get theta from our x,y coordinates
         phi = atan2(y,x);                              %Get phi from our x,y coordinates
          for freq_ind = 1:length(frequency_bands)      %Apply beamforming algo. in every freq. band
              frequency = frequency_bands(freq_ind);

              % Create empty mic data for every frequency band
              mic_data = 0;

              for array = 1:sub_arrays
                  % Use the filtered audio signals 
                  audio_temp_signals = audio_filtered_complete(array,freq_ind).audio_signals;
    
                  % Adaptive configuration of the antenna array
                  % Select only necessary antenna-elements to maintain small
                  % beamwidth.
                  w_index = weight_index(array_matrices(array),frequency,c);
                  weight = weight_m(w_index,:);

                  % Perform the beamforming algorithm (phase-shift input signal according to listening direction)
                  mic_data = mic_data + beam_forming_alogrithm(array_matrices(array),[theta,phi],...
                      weight,audio_temp_signals,frequency,f_sampling,c);

              end
             
              % Obtain relative power in the listening direction
              color = sum(abs(mic_data(:,1)).^2)/samples;

              % Relative power in the direction [theta,phi] saved in matrix
              color_maps_complete(freq_ind).color_data_matrix(y_ind,x_ind)  = color;

              % Movie-Time!
              if fps_rate > 0
                  for frame_ind = 2:length(frames_index(1,:))
                      color = sum(abs(mic_data(frames_index(1,frame_ind):frames_index(2,frame_ind),1)).^2);
                      color_maps_movie(frame_ind,freq_ind).color_data_matrix(y_ind,x_ind) = color;
                  end
              end
          end
    end

end

%% Validation check
% Gives a colormap of the actual positions of the sources
if signal_type == "emulated"
    xy_val_check = zeros(length(y_listen),length(x_listen));
    
    for x_ind = 1:length(x_listen)
        x = x_listen(x_ind);
        for y_ind = 1:length(y_listen)
            y = y_listen(y_ind);
            temp_val = 0;
            for source_ind = 1:length(sources)
                x_s = r_scan*sin(sources(source_ind).theta)*cos(sources(source_ind).phi);
                y_s = r_scan*sin(sources(source_ind).theta)*sin(sources(source_ind).phi);
                temp_val = temp_val + 1/((x_s-x)^2 + (y_s-y)^2)^(1/2);
            end
            xy_val_check(y_ind,x_ind) = temp_val;
        end
    end

    figure(3)
    imagesc(x_listen,y_listen,xy_val_check);
    set(gca,'YDir','normal')
    set(gca,'TickLabelInterpreter','latex','FontSize',12)
    title('Location of the sources',...
            'Interpreter','latex','FontSize',15);
    % export figure
    %exportgraphics(gcf,'figures/two_sources_one_array/intensity_check_one_soruce.png','Resolution',300)
end

%% Cartesian color map calculations
% Get the maximum intensity
max_intensity = 0;
for freq_ind = 1:f_bands_N
    intensity = max(max(color_maps_complete(freq_ind).color_data_matrix));
    if intensity > max_intensity
        max_intensity = intensity;
    end
end

% Sum all the colormaps for every frequency to see intensity for total spectrum
color_map_intensity = zeros(length(y_listen),length(x_listen));  
for freq_ind = 1:f_bands_N
    color_map_intensity =  color_map_intensity + color_maps_complete(freq_ind).color_data_matrix;
end

% Absolute value of the gradient of the spectrum map to obtain the sources locations

% Calculate the gradient of the intensity (gradient_x, gradient_y), and the absolute value of the
% gradient of the intensity (color_map_intensity_grad)
color_map_intensity_grad = zeros(length(y_listen),length(x_listen));  
gradient_x = zeros(length(y_listen),length(x_listen));
gradient_y = zeros(length(y_listen),length(x_listen));

for x_ind = 2:length(x_listen)-1
    for y_ind = 2:length(y_listen)-1
        % ( f(x+1) - f(x-1) )/2
        gradient_x(x_ind,y_ind) = (color_map_intensity(x_ind+1,y_ind) - ...
            color_map_intensity(x_ind-1,y_ind))/2;

        % ( f(y+1) - f(y-1) )/2
        gradient_y(x_ind,y_ind) = (color_map_intensity(x_ind,y_ind+1) - ...
            color_map_intensity(x_ind,y_ind-1))/2;

        gradient = [gradient_x(x_ind,y_ind); gradient_y(x_ind,y_ind);0];
        color_map_intensity_grad(x_ind,y_ind) = norm(gradient);

    end
end

% Calculate the divergence of the gradient (laplace operator) (color_map_intensity_grad2)
color_map_intensity_grad2 = zeros(length(y_listen),length(x_listen));  
for x_ind = 3:length(x_listen)-2
    for y_ind = 3:length(y_listen)-2
        % ( f(x+1) - f(x-1) )/2
        gradient_x_temp = (gradient_x(x_ind+1,y_ind) - ...
            gradient_x(x_ind-1,y_ind))/2;

        % ( f(y+1) - f(y-1) )/2
        gradient_y_temp = (gradient_y(x_ind,y_ind+1) - ...
            gradient_y(x_ind,y_ind-1))/2;

        color_map_intensity_grad2(x_ind,y_ind) = gradient_x_temp + gradient_y_temp;
    end
end

max_amp = max(max(color_map_intensity)); % max amplitude in intensity of the color map

grad2_max = max(max(color_map_intensity_grad2));
grad_max = max(max(color_map_intensity_grad));
intensity_max = max(max(color_map_intensity));


% Precision map combines the intensity map, absolute derivative map and the
% second derivative map to distinguish peaks
%
% By applying some threshold function to the precision_map, the location of
% the sources seen by the audio_warfare algorithm can be obtained
%

precision_map_temp = ((1./(color_map_intensity_grad/grad_max) )).* -color_map_intensity_grad2/grad2_max.*color_map_intensity/intensity_max;
precision_map = zeros(length(y_listen),length(x_listen));
precision_map(2:end-1,2:end-1) = precision_map_temp(2:end-1,2:end-1);
precision_map = precision_map./(max(max(precision_map)));

locations = find_sources(precision_map,x_listen,y_listen,0.3);
sources_found = length(locations(:,1));

if signal_type == "emulated"
    validation_locations = find_sources(xy_val_check,x_listen,y_listen,0.3);
end

locations_angles = zeros(sources_found,2);

% Convert the locations to theta and phi
for source_ind = 1:sources_found
    x = locations(source_ind,1);
    y = locations(source_ind,2);
    z_0 = sqrt(r_scan^2-x^2-y^2);
    
    if signal_type == "emulated"
        x_val = validation_locations(source_ind,1);
        y_val = validation_locations(source_ind,2);
        z_0_val = sqrt(r_scan^2-x^2-y^2);
        theta_validation = 180/pi*acos(z_0_val/(sqrt(x_val^2 + y_val^2 + z_0_val^2)));   %Get theta from our x,y coordinates
        phi_validation = 180/pi*atan2(y_val,x_val);
        disp(strcat('Validation location at theta =  ', int2str(theta_validation), ', phi = ',int2str(phi_validation)));
    end

    theta = 180/pi*acos(z_0/(sqrt(x^2 + y^2 + z_0^2)));   %Get theta from our x,y coordinates
    phi = 180/pi*atan2(y,x);  

    locations_angles(source_ind,:) = [theta,phi];
    disp(strcat('Source found at theta =  ', int2str(theta), ', phi = ',int2str(phi)));
end


%% COLOR MAP plos
% Cartesian color map plot
title_size = 18;
subtitle_size = 12;

% Generate filenames of the .pgn figures to save of the plots,
% and subtitles to the figures
if signal_type == "emulated"
    if N_sources == 1
        sub_title = "$$f_{source}$$="+f_start1+"Hz, $$\theta$$="+theta_deg1+"$$^\circ$$, $$\phi$$="+phi_deg1+"$$^\circ$$";
        filename = "f="+f_start1+"_theta="+theta_deg1+"_phi="+phi_deg1+"_A"+N_arrays;
    elseif N_sources == 2
        sub_title = {"$$f_{source1}$$="+f_start1+"Hz, $$\theta_1$$="+theta_deg1+"$$^\circ$$, $$\phi_1$$="+phi_deg1+"$$^\circ$$",...
            "$$f_{source2}$$="+f_start2+"Hz, $$\theta_2$$="+theta_deg2+"$$^\circ$$, $$\phi_2$$ = "+phi_deg2+"$$^\circ$$"};
        filename = "f="+f_start1+"_"+f_start2+"_theta="+theta_deg1+"_"+theta_deg2+"_phi="+phi_deg1+"_"+phi_deg2+"_A"+N_arrays;
    end
elseif signal_type == "recorded"
    % If the signals is recorded, a filename already exists
    sub_title = '';
end

clims = [0 max_intensity]; % color limits for plots

% Plot the color map of the beamforming for every frequency-band, 
% before the signals of every frequency band have been summed up
figure(4); clf(figure(4));
for plot_ind = 1:f_bands_N
    nexttile
    imagesc(x_listen,y_listen,color_maps_complete(plot_ind).color_data_matrix,clims);
    set(gca,'YDir','normal');
    title(strcat('f @ ', int2str(frequency_bands(plot_ind)), ' Hz'),...
        'Interpreter','latex','FontSize',subtitle_size);
    set(gca,'TickLabelInterpreter','latex','FontSize',15)
    if plot_ind == f_bands_N
        break;
    end 
end

% Plot of beamforming results
figure(5)
imagesc(x_listen,y_listen,color_map_intensity);
set(gca,'YDir','normal')
set(gca,'TickLabelInterpreter','latex','FontSize',12)
title('Beamforming results',...
        'Interpreter','latex','FontSize',title_size);
subtitle(sub_title,...
        'Interpreter','latex','FontSize',subtitle_size);
% export figure
%exportgraphics(gcf,"figures/real_data/"+filename+"_MIMO.png",'Resolution',300)

if plot_intensity_grad
    figure(6)
    imagesc(x_listen,y_listen,color_map_intensity_grad);
    set(gca,'YDir','normal')
    set(gca,'TickLabelInterpreter','latex','FontSize',12)
    title('Beamforming results, grad 1',...
            'Interpreter','latex','FontSize',title_size);
    subtitle(sub_title,...
        'Interpreter','latex','FontSize',subtitle_size);
    % export figure
    %exportgraphics(gcf,'long_dist_test.png','Resolution',300)
    
    figure(7)
    imagesc(x_listen,y_listen,-color_map_intensity_grad2);
    set(gca,'YDir','normal')
    set(gca,'TickLabelInterpreter','latex','FontSize',12)
    title('Beamforming results, grad 2',...
            'Interpreter','latex','FontSize',title_size);
    subtitle(sub_title,...
        'Interpreter','latex','FontSize',subtitle_size);
    % export figure
    %exportgraphics(gcf,'long_dist_test.png','Resolution',300)
end

% Plot of enhanced beamforming results
figure(8)
clim = [0 1];
imagesc(x_listen,y_listen,precision_map,clim);
set(gca,'YDir','normal')
set(gca,'TickLabelInterpreter','latex','FontSize',12)
title('Beamforming results enhanced',...
        'Interpreter','latex','FontSize',title_size);
subtitle(sub_title,...
        'Interpreter','latex','FontSize',subtitle_size);
% export figure
%exportgraphics(gcf,"figures/real_data/"+filename+"_MIMO_precision.png",'Resolution',300)


%% Movie Time!
movietest = movie_function(max_intensity,color_maps_movie,fps_rate,x_listen,y_listen);
