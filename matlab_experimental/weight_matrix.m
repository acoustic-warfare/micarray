function weight_m = weight_matrix(matrix_array,max_mode)
%
%   Decice how many elements to use in order to obtain the best
%   directivity. Very crude function. Weighting-chart desgined from
%   beamforming simulations listening in the broadside direction. More
%   optimal weights exist for a different direction than broadside
%
%
row_elements = matrix_array.row_elements;
column_elements = matrix_array.column_elements;
elements = matrix_array.elements;

weight_m = zeros(max_mode,elements);
for mode = 1:max_mode
    weight = zeros(1,row_elements*column_elements);
    row_lim = ceil((row_elements)/mode);
    column_lim = ceil((column_elements)/mode);
    test = 0;
    for i = 1:row_lim
        for j = 1:column_lim
            element_index = (mode*(i-1))*row_elements + mode*(j-1) +1;
            weight(element_index) = 1;
        end
    end

    weight_m(mode,:) = weight;

end


end

