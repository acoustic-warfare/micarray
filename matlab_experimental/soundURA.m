%% ACOUSTIC ARRAY TEST SCRIPT
% Inspiration and examples: https://www.mathworks.com/help/phased/sonar-systems.html?s_tid=CRUX_lftnav
%clear all

%% OPTIONS
% Plot / illustrative options
plotPatternULA = 0; % plot response pattern of URL
viewArrayULA = 0;   % view array of ULA

plotPatternURA = 1; % plot response pattern of URA
viewArrayURA = 1;   % view array of URA     

% frequency span of microphone
f_min = 50;         % min frequency of microphone
f_max = 20e3;       % max frequency of microphone

% Incident angle of signal
sig1_ang = [50;50];    % Angel of signal, [AzimuthAngle; ElevationAngle]
sig2_ang = [-50;-50];    % Angel of signal, [AzimuthAngle; ElevationAngle]

% Array dimensions
d = 0.02;           % distance between elements (in m)
Nele = 8;           % Number of array elements

% sampling
sample_rate = 40100; %Sample rate of 10kHz

% constants
c = 343;            % sound speed (in m/s)
lambda = c/f_max;   % wavelength

%% MICROPHONE ELEMENT
% models a microphone element with an omnidirectional response pattern
microphone = ...
    phased.OmnidirectionalMicrophoneElement('FrequencyRange', [f_min f_max]);

%% Uniform Linear Array (ULA)
% Uniform linear array of microphones
% https://www.mathworks.com/help/phased/ref/phased.ula-system-object.html
% creates uniform linear array of microphone elements
ula = phased.ULA('NumElements',Nele,'ElementSpacing',d,'Element', microphone, 'ArrayAxis', 'y'); % array axis y

if viewArrayULA
    viewArray(ula);
end

%% Uniform rectangular array (URA)
% Uniform rectangular array of microphones
ura = phased.URA('Size', [Nele Nele], 'ElementSpacing',d,'Element', microphone, 'ArrayNormal','x');

if viewArrayURA
    viewArray(ura,'orientation',[0,1,1]');
end


%% COLLECTOR
% ULA
% https://www.mathworks.com/help/phased/ref/phased.widebandcollector-system-object.html
collectorULA = phased.WidebandCollector('Sensor',ula,'SampleRate',sample_rate,...
    'PropagationSpeed',c,'ModulatedInput',false);

% URA
% https://www.mathworks.com/help/phased/ref/phased.widebandcollector-system-object.html
collectorURA = phased.WidebandCollector('Sensor',ura,'SampleRate',sample_rate,...
    'PropagationSpeed',c,'ModulatedInput',false,'NumSubbands',500);


%% AUDIO

load handel.mat
audiowrite('handel.wav',y,Fs);
%[sig2,Fs2] = audioread('handel.wav', [1,52000]);
[temp_sig2,Fs2] = audioread('28 - Black Box - Ride on Time (79 Disco Mix).mp3', [1,1000000]);

load laughter.mat
audiowrite('laughter.wav',y,Fs);
%[sig1,Fs1] = audioread('laughter.wav', [1,52000]);
[temp_sig1,Fs1] = audioread('119 - Kenny Loggins - Danger Zone.mp3', [1,1000000]);

sig2 = temp_sig2(:,1);
sig1 = temp_sig1(:,1);

% add noise
% noise = 0.1*(randn(size(x)) + 1j*randn(size(x)));
% rx = x + noise;

padding = zeros(8000,1);
sig2 = [padding; sig2];
sig1 = [sig1; padding];


%% COLLECT SIGNAL
% ULA or URA collecting a signal originated in a specific angle
% collector(signal,angle)


CsigULA = collectorULA(sig1,sig1_ang);
CsigURA = collectorURA(sig1,sig1_ang);

% Collect 2 signals
CsigURA2 = collectorURA([sig1 sig2],...
        [sig1_ang sig2_ang]); % Collect two sound sources from different angles

%soundsc(CsigURA2)



%% PLOTS
%% RESPONS PATTERN (ULA)
if plotPatternULA
    fc = 1e3; % frequency to plot response
    figure();
    pattern(ula,fc,-180:180,0,'CoordinateSystem','polar',...
            'PropagationSpeed',c);
end

%% RESPONS PATTERN (URA)
if plotPatternURA
    fc = 4e3; % frequency to plot response
    figure();
    pattern(ura,fc,-45:45,0,'CoordinateSystem','rectangular',...
            'PropagationSpeed',c);
end

%% MAGNITUDE and PHASE, FFT of URA


%% Plot signals

n_x1 = 1:length(sig1);
n_x2 = 1:length(sig2);

figure(10)
plot(n_x1,sig1);

figure(11)
plot(n_x2,sig2);