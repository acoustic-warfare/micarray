%% Initialization
f_sampling = 16000;     %sampling frequency in Hz
t_start = 0;
t_end = 2;
t_total = t_end - t_start;
t = linspace(t_start,t_end,t_total*f_sampling); 
samples = length(t);

%% Movie settings
fps_rate =0;
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
c = 343;
away_distance = 10;    % The distance between the array and the sources
%sources = [audio_source(400,500,100,45,20,0,1),audio_source(450,600,200,0,0,0,1.5)];
sources = [audio_source(3500,3600,40,35,180,away_distance,0,1.5),audio_source(3100,3200,40,35,0,away_distance,0.5,2)];
%sources = [audio_source(3000,3400,50,90,180,away_distance,0,1)];

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

%% Generate signals on each element of the audio array
%Audio_signal = generate_array_signals(array_matrix,sources,t,c);

% Create a vector containing the audio_data for each sub_array
array_audio_signals(sub_arrays) = audio_data;

for array = 1:sub_arrays

    % Generate the audio signals on each array-element for each sub-array
    temp_signal = generate_array_signals(array_matrices(array),sources,t,c);
    array_audio_signals(array).audio_signals = temp_signal;
    disp(strcat('Audio signal for array ', int2str(array), ' generated'));
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
%                   Begin signal processing on generated data
%
%
%% Beamforming CLEAN
x_res = 20;                         %Resolution in x
y_res = 20;                         %Resolution in y
x_listen = linspace(-1,1,x_res);    %Our scanning window, x coordinates
y_listen = linspace(-1,1,y_res);    %Our scanning window, y coordinates
r_scan = sqrt(2);                   %radius of scanning window. r_scan^2 = x^2 + y^2 + z^2

f_bands_N = 55;          %Linearly spaced N bands,
frequency_bands = linspace(1000,6700,f_bands_N);
f_coefficients = get_filter_coefficients(f_sampling,frequency_bands);

scan_angles = zeros(2,x_res*y_res);
scan_index = 1;
for x_ind = 1:length(x_listen)
    x = x_listen(x_ind);
    for y_ind = 1:length(y_listen)
         y = y_listen(y_ind);
         z_0 = sqrt(r_scan^2-x^2-y^2);
         theta = acos(z_0/(sqrt(x^2 + y^2 + z_0^2)));   %Get theta from our x,y coordinates
         phi = atan2(y,x);                              %Get phi from our x,y coordinates

         scan_angles(1,scan_index) = theta;
         scan_angles(2,scan_index) = phi;
         scan_index = scan_index + 1;
    end
end

tic
scan_length = length(scan_angles);
theta_listen = scan_angles(1,:);
phi_listen = scan_angles(2,:);
color_intensity = zeros(1,scan_length);
audio_signals = array_audio_signals(1).audio_signals;
weight_m = weight_matrix(array_matrices(1),7);

c =  parallel.pool.Constant(c);
f_sampling =  parallel.pool.Constant(f_sampling);
r_prime =  parallel.pool.Constant(array_matrices(1).r_prime);
audio_signals = parallel.pool.Constant(audio_signals);
f_coefficients = parallel.pool.Constant(f_coefficients);
frequency_bands = parallel.pool.Constant(frequency_bands);
weight_m = parallel.pool.Constant(weight_m);

parfor scan_ind = 1:scan_length
    scan_ind

    audio_out = AW_listening_parallel(c.Value,f_sampling.Value,r_prime.Value...
        ,theta_listen(scan_ind),phi_listen(scan_ind),...
        audio_signals.Value,f_coefficients.Value,frequency_bands.Value,weight_m.Value);

    color = sum(abs(audio_out).^2)/samples;
    color_intensity(scan_ind) = color;
end
toc

%% Generate colormap

color_intensity_map = zeros(y_res,x_res);
scan_ind = 1;
for x_ind = 1:x_res
    for y_ind = 1:y_res
        color_intensity_map(y_ind,x_ind) = color_intensity(scan_ind);
        scan_ind = scan_ind + 1;
    end
end

% x_ind = 1;
% y_ind = 1;
% for scan_ind = 1:scan_length
%     x_ind = 1 + floor(scan_ind/x_res);
%     y_ind = 1 + mod(scan_ind-1,y_res);
%     color_intensity_map(y_ind,x_ind) = color_intensity(scan_ind);
% end

%% Colormap plotting
figure(21)
imagesc(x_listen,y_listen,color_intensity_map);
set(gca,'YDir','normal')
set(gca,'TickLabelInterpreter','latex','FontSize',12)
title('Beamforming results',...
        'Interpreter','latex','FontSize',15);
%exportgraphics(gcf,'intensity_test.png','Resolution',300)

