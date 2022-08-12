function f_coefficients = get_filter_coefficients(f_sampling,center_frequencies)

% filter coefficients matrix
filter_order = 200;
f_coefficients = zeros(length(center_frequencies),filter_order+1);
scale_factor = 10000;


for freq_ind = 1:length(center_frequencies)
    % Filter design for each band
    nu_0 = 2*center_frequencies(freq_ind)/f_sampling;
    cut_off = [nu_0 - nu_0/scale_factor, nu_0  + nu_0/scale_factor];
    b = fir1(filter_order,cut_off);
    f_coefficients(freq_ind,:) = b;
end

end
