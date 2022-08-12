function f_mega_coefficients = mega_filter_coefficients(array_matrices,f_sampling,center_frequencies,matrix_array,theta,phi,c)

% filter coefficients matrix
filter_order = 199;
f_coefficients = zeros(length(center_frequencies),filter_order+1);
scale_factor = 10000;

for freq_ind = 1:length(center_frequencies)
    % Filter design for each band
    nu_0 = 2*center_frequencies(freq_ind)/f_sampling;
    cut_off = [nu_0 - nu_0/scale_factor, nu_0  + nu_0/scale_factor];
    b = fir1(filter_order,cut_off);
    f_coefficients(freq_ind,:) = b;
end

% final filter coefficients matrix 
elements = matrix_array.elements;

rows = length(center_frequencies)*elements;
columns = filter_order+1 +2;

f_mega_coefficients = zeros(rows,columns);

r_prime = matrix_array.r_prime;

x_factor = sin(theta)*cos(phi);
y_factor = sin(theta)*sin(phi);

x = r_prime(1,:);
y = r_prime(2,:);

a_0 = 1;

P = filter_order;

% Weight matrix
weight_m = weight_matrix(array_matrices(1),7);

for freq_ind = 1:length(center_frequencies)
    frequency = center_frequencies(freq_ind);
    w_index = weight_index(array_matrices(1),frequency,c);
    weight = weight_m(w_index,:);
    for mic_ind = 1:elements
        if weight(mic_ind) ==1
            % Filter coefficients for each band
            b = f_coefficients(freq_ind,:);
    
            % Row index
            row_index = (freq_ind-1)*elements + mic_ind;
    
            % Center frequency
            frequency = center_frequencies(freq_ind);
    
            % The narrowband frequency
            k = 2*pi*frequency/c;
    
            % The normalized frequency
            ny = frequency/f_sampling;
    
            % Phase shift value that is dependent on the frequency and
            % the location of the element (x,y)
            phi_0 = -k*(x(mic_ind)*x_factor + ...
                        y(mic_ind)*y_factor);
            
            f_mega_coefficients(row_index,1) = sin(phi_0)/(4*pi*ny*a_0)*b(1);
    
            f_mega_coefficients(row_index,2) = cos(phi_0)/a_0*b(1) + sin(phi_0)/(4*pi*ny*a_0)*b(2);
    
            f_mega_coefficients(row_index,3:P+1) = cos(phi_0)/a_0.*b(2:P) + sin(phi_0)/(4*pi*ny*a_0).*(...
                b(3:P+1) - b(1:P-1));
    
            f_mega_coefficients(row_index,P+2) = (cos(phi_0)/a_0 * b(P+1) - sin(phi_0)/(4*pi*ny*a_0) * b(P));
    
            f_mega_coefficients(row_index,P+3) = -sin(phi_0)/(4*pi*ny*a_0)*b(P+1);

        end
    end
end

end 
