function [data, fs] = load_data(filename)
% LOAD_DATA loads data from a .txt file.
%   In this program, the input file is assumed to only contain data from
%   the microphones (and not any other information). Therefore, the sampling
%   frequency needs to be entered manually here.
%
%   It is possible to reduce the number of samples that should be processed
%   by chaning the value of the variable "samples". 
%
%   The order of the microphones in the .txt file (sampled by the FPGA) is
%   different from what the algorithm is working with, so the microphone
%   order is rearranged to reflect the real array (the array is "mirrored" in reality)

    data = readmatrix(filename);
    fs = 48828;                 % sampling frequency

    % number of samples to process, change the value of the variable
    % "samples" if you want to process less samples
    samples = floor(length(data(:,1))/2);   % all samples

    % state the new order of microphones
    order = zeros(1,64);
    for i = 1:64
        order(i) = i;
    end
    for i = 1:8:64
        order(i:i+7) = flip(order(i:i+7));
    end
    
    data = data(1:samples,order); % take out samples, and rearrange mics

end