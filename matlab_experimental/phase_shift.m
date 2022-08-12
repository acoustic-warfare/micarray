function y = phase_shift(x,ny,phase)
%PHASE Summary of this function goes here
%INPUT x(n) = sin(2piny*n + arbitrary phase)     
%OUTPUT y(n) = sin(2piny*n + arbitrary phase + phase)
%   Detailed explanation goes here
y = zeros(length(x),1);
A = cos(phase);
B = sin(phase)/(4*pi*ny);

% for i = 2:length(x)-1
%     y(i,1) =  A*x(i) + B*(x(i+1)-x(i-1));
% end


for i = 3:length(x)
    y(i,1) =  A*x(i-1) + B*(x(i)-x(i-2));
end






