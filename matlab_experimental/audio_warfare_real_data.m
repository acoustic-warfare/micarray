% %% load real data
% array_audio_signals(1) = audio_data;
% [array_audio_signals(1).audio_signals,f_sampling] = load_data(); 
% array_audio_signals(1).audio_signals = array_audio_signals(1).audio_signals/(max(max(array_audio_signals(1).audio_signals)));
% samples = length(array_audio_signals(1).audio_signals(:,1));


%% Load emulated data
array_audio_signals(1) = audio_data;
%array_audio_signals(1).audio_signals = CsigURA2(:,:);
%samples = length(CsigURA2(:,1));
array_audio_signals(1).audio_signals = CsigURA2(1:80000,:);
samples = length(CsigURA2(1:80000,1));
f_sampling = 44100;

%% Initialization
c = 343;
t_start = 0;
t_end = samples/f_sampling;
t_total = t_end - t_start;
t = linspace(t_start,t_end+1/f_sampling,samples); 

%% Movie settings
fps_rate =5;
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

%% Audio Array Geometry
row_elements = 8;
column_elements = 8;
uni_distance = 20*10^-3;
%r_a1 = 10^-3 * [80;80;0];
r_a1 = 10^-3 * [0;0;0];
r_a2 = 10^-3 * [-80;80;0];
r_a3 = 10^-3 * [80;-80];
r_a4 = 10^-3 * [-80;-80];

array_matrix = matrix_array(r_a1,uni_distance,row_elements,column_elements);
array_matrix2 = matrix_array(r_a2,uni_distance,row_elements,column_elements);
array_matrix3 = matrix_array(r_a3,uni_distance,row_elements,column_elements);
array_matrix4 = matrix_array(r_a4,uni_distance,row_elements,column_elements);

%array_matrices = [array_matrix,array_matrix2,array_matrix3,array_matrix4];
array_matrices = [array_matrix];

sub_arrays = length(array_matrices);

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

%% Plot generated signals

figure(2)
samples_max = 100;
for i = 1:length(array_audio_signals(1).audio_signals(1,:))
    plot(1:samples_max,array_audio_signals(1).audio_signals(1:samples_max,i))
    hold on
end
set(gca,'TickLabelInterpreter','latex','FontSize',18)
xlabel('n ','Interpreter','latex','FontSize',18);
ylabel('x[n]','Interpreter','latex','FontSize',18)
%legend('MATLAB−sim','CST−sim','Interpreter','latex','location','southeast')
%set(gca,'GridALpha',0.1,'LineWidth',.3);
%exportgraphics(gcf,'generated_signals_close.png','Resolution',300)

% 
% 
% 
%                   All signals generated after this point  
% 
% 
% 
%
%
%
%
%                   Begin signal processing on generated data
%
%

%% Beamforming CLEAN
tic
x_res = 20;                         %Resolution in x
y_res = 20;                         %Resolution in y
x_listen = linspace(-1,1,x_res);    %Our scanning window, x coordinates
y_listen = linspace(-1,1,y_res);    %Our scanning window, y coordinates
r_scan = sqrt(2);                   %radius of scanning window. r_scan^2 = x^2 + y^2 + z^2

% Which bands to check
f_bands_N = 55;          %Linearly spaced N bands,
frequency_bands = linspace(100,3800,f_bands_N);

% Create N-audio complete audio signals for every band
audio_filtered_complete(sub_arrays,f_bands_N) = audio_data;
samples = length(t);
filter_order = 200;
filter_coefficients = zeros(f_bands_N,filter_order+1);

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

tic
for array = 1:sub_arrays
    % Filter the signals generated on the arrays
    Audio_signal = array_audio_signals(array).audio_signals;
    elements = array_matrices(array).elements;

    for freq_ind = 1:length(frequency_bands)
        % Filter design for each band
        nu_0 = 2*frequency_bands(freq_ind)/f_sampling;
        scale_factor = 10000;
        cut_off = [nu_0 - nu_0/scale_factor, nu_0  + nu_0/scale_factor];
        b = fir1(filter_order,cut_off);
        filter_coefficients(freq_ind,:) = b;
    
        audio_temp = zeros(samples,elements); 

        for mic_ind = 1:elements
            % Apply filter on every signal recorded from the elements
            audio_temp(:,mic_ind) = filter(b,1,Audio_signal(:,mic_ind));
            %audio_temp(:,mic_ind) = Audio_signal(:,mic_ind);
        end
    
        % Add the complete set of element-signals
        audio_filtered_complete(array,freq_ind) = audio_data(audio_temp);
    end

end

toc

% Weight matrix
weight_m = weight_matrix(array_matrices(1),7);
tic

for x_ind = 1:length(x_listen)
    x = x_listen(x_ind);
    x_ind
    for y_ind = 1:length(y_listen)
         y = y_listen(y_ind);
         z_0 = sqrt(r_scan^2-x^2-y^2);
         theta = acos(z_0/(sqrt(x^2 + y^2 + z_0^2)));   %Get theta from our x,y coordinates
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
                  % weight = adaptive_array_config_gen2(array_matrix,frequency,theta,phi,c);
                  %weight = adaptive_array_config(array_matrices(array),frequency,c);
                  %weight = ones(1,elements);
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

              % Movie-Time!!
               if fps_rate > 0
                  for frame_ind = 2:length(frames_index(1,:))
                      color = sum(abs(mic_data(frames_index(1,frame_ind):frames_index(2,frame_ind),1)).^2);
                      %color = sum(abs(fft(mic_data(frames_index(1,frame_ind):frames_index(2,frame_ind),1))).^2);
                      color_maps_movie(frame_ind,freq_ind).color_data_matrix(y_ind,x_ind) = color;
                  end
              end
          end
    end

end

toc

% Get the maximum intensity
max_intensity = 0;
for freq_ind = 1:f_bands_N
    intensity = max(max(color_maps_complete(freq_ind).color_data_matrix));
    if intensity > max_intensity
        max_intensity = intensity;
    end
end

%% Cartesian color map plot
clims = [0 max_intensity];

% Plot the colormap for every frequency-band
figure(20)
for plot_ind = 1:f_bands_N
    % figure(plot_ind);
    nexttile
    imagesc(x_listen,y_listen,color_maps_complete(plot_ind).color_data_matrix,clims);
    set(gca,'YDir','normal');
    title(strcat('Frequency @ ', int2str(frequency_bands(plot_ind)), ' Hz'),...
        'Interpreter','latex','FontSize',12);
    set(gca,'TickLabelInterpreter','latex','FontSize',15)
    
    if plot_ind == f_bands_N
        break;
    end
    
end

% Sum all the colormaps for every frequency to see intensity for total
% spectrum
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

max_amp = max(max(color_map_intensity));

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

locations = find_sources(precision_map,x_listen,y_listen,0.34);
sources_found = length(locations(:,1));

locations_angles = zeros(sources_found,2);

% Convert the locations to theta and phi
for source_ind = 1:sources_found
    x = locations(source_ind,1);
    y = locations(source_ind,2);
    z_0 = sqrt(r_scan^2-x^2-y^2);

    theta = 180/pi*acos(z_0/(sqrt(x^2 + y^2 + z_0^2)));   %Get theta from our x,y coordinates
    phi = 180/pi*atan2(y,x);  

    locations_angles(source_ind,:) = [theta,phi];
    disp(strcat('Source found at theta =  ', int2str(theta), ', phi = ',int2str(phi)));
end

clim = [0 1];

figure(21)
imagesc(x_listen,y_listen,color_map_intensity);
set(gca,'YDir','normal')
set(gca,'TickLabelInterpreter','latex','FontSize',12)
title('Beamforming results',...
        'Interpreter','latex','FontSize',15);
%exportgraphics(gcf,'intensity_test.png','Resolution',300)

figure(22)
imagesc(x_listen,y_listen,color_map_intensity_grad);
set(gca,'YDir','normal')
set(gca,'TickLabelInterpreter','latex','FontSize',12)
title('Beamforming results',...
        'Interpreter','latex','FontSize',15);
%exportgraphics(gcf,'long_dist_test.png','Resolution',300)

figure(23)
imagesc(x_listen,y_listen,-color_map_intensity_grad2);
set(gca,'YDir','normal')
set(gca,'TickLabelInterpreter','latex','FontSize',12)
title('Beamforming results',...
        'Interpreter','latex','FontSize',15);
%exportgraphics(gcf,'long_dist_test.png','Resolution',300)

figure(24)
imagesc(x_listen,y_listen,precision_map,clim);
set(gca,'YDir','normal')
set(gca,'TickLabelInterpreter','latex','FontSize',12)
title('Beamforming results enhanced',...
        'Interpreter','latex','FontSize',15);
%exportgraphics(gcf,'intensity_precision.png','Resolution',300)

%% Filter plot

colors = [0 0.4470 0.7410; 0.8500 0.3250 0.0980;0.9290 0.6940 0.1250;0.4940 0.1840 0.5560;...
    0.4660 0.6740 0.1880;0.3010 0.7450 0.9330];

figure(23)
for freq_ind = 1:f_bands_N
    color_ind = mod(freq_ind-1,6) + 1;
    [h,f] = freqz(filter_coefficients(freq_ind,:),1,1000,'whole',f_sampling);
    plot(f,20*log10(abs(h)),'color',colors(color_ind,:),'linewidth',0.8);
    set(gca,'TickLabelInterpreter','latex','FontSize',11)
    xlim([0 f_sampling/2])
    ylim([-60 0])
    xlabel('Frequency (Hz)','Interpreter','latex','FontSize',20);
    ylabel('Magnitude (dB)','Interpreter','latex','FontSize',20)
    grid on
    hold on
end


%% Post processing 
% Testing the AW_listening function to apply spatial filtering and obtain a
% signal in the wanted direction
frames_index_AW = [1;samples];
tic
audio_out = AW_listening(c,f_sampling,array_matrices,45,-45,array_audio_signals);
toc

%% post processing improved

tic
f_bands_N = 300;          %Linearly spaced N bands,

%Exponentiall spaced bands
f_min = 100;
f_max = 20000;
frequency_bands = exp(linspace(log(f_min),log(f_max),f_bands_N));

%Linearly spaced bands
%frequency_bands = linspace(f_min,f_max,f_bands_N);
f_coefficients = get_filter_coefficients(f_sampling,frequency_bands);
audio_out2 = AW_listening_improved(c,f_sampling,array_matrices,47,-45,array_audio_signals,f_coefficients,frequency_bands);
toc
%% Plot results 
Audio_signal = array_audio_signals(1).audio_signals;
figure(6)
plot(t,Audio_signal(:,2)/max(Audio_signal(:,2)))
set(gca,'TickLabelInterpreter','latex','FontSize',18)
ylim([-1 1])
xlabel('t (s) ','Interpreter','latex','FontSize',18);
ylabel('$y[f_s t]$','Interpreter','latex','FontSize',18)
%exportgraphics(gcf,'time_domain_signal.png','Resolution',300)

figure(7)
plot(t,audio_out2)
set(gca,'TickLabelInterpreter','latex','FontSize',18)
%ylim([-1 1])
xlabel('t (s) ','Interpreter','latex','FontSize',18);
ylabel('$y[f_s t]$','Interpreter','latex','FontSize',18)
%exportgraphics(gcf,'time_domain_signal_AW.png','Resolution',300)

[spectrum_original,nu] = tdftfast(Audio_signal(:,2));
[spectrum_bm,nu2] = tdftfast(audio_out2);

figure(8)
plot(nu,abs(spectrum_bm));
title('Spectrum of spatially filtered signal')

figure(9)
plot(nu2,abs(spectrum_original));
title('Spectrum of original signal')


%% Movie Time!

movietest = movie_function(max_intensity,color_maps_movie,fps_rate,x_listen,y_listen);




